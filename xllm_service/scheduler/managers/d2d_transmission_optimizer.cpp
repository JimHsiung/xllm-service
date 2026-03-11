/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm-service/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "d2d_transmission_optimizer.h"

#include <glog/logging.h>

#include <algorithm>
#include <array>
#include <limits>
#include <sstream>
#include <unordered_set>

namespace xllm_service {
namespace {

bool global_npu_less(const D2DTransmissionOptimizer::GlobalNpu& lhs,
                     const D2DTransmissionOptimizer::GlobalNpu& rhs) {
  if (lhs.instance != rhs.instance) {
    return lhs.instance < rhs.instance;
  }
  return lhs.local_npu < rhs.local_npu;
}

std::string format_global_npu(
    const D2DTransmissionOptimizer::GlobalNpu& source) {
  std::ostringstream oss;
  oss << source.instance << "#" << source.local_npu;
  return oss.str();
}

bool contains_source(
    const std::vector<D2DTransmissionOptimizer::GlobalNpu>& sources,
    const D2DTransmissionOptimizer::GlobalNpu& source) {
  return std::binary_search(
      sources.begin(), sources.end(), source, global_npu_less);
}

int get_source_load(
    const std::unordered_map<D2DTransmissionOptimizer::GlobalNpu,
                             int,
                             D2DTransmissionOptimizer::GlobalNpuHasher>&
        source_loads,
    const D2DTransmissionOptimizer::GlobalNpu& source) {
  auto it = source_loads.find(source);
  if (it == source_loads.end()) {
    return 0;
  }
  return it->second;
}

int count_max_source_load(
    const std::unordered_map<D2DTransmissionOptimizer::GlobalNpu,
                             int,
                             D2DTransmissionOptimizer::GlobalNpuHasher>&
        source_loads) {
  int max_load = 0;
  for (const auto& kv : source_loads) {
    max_load = std::max(max_load, kv.second);
  }
  return max_load;
}

int compute_single_change_boundary_delta(
    const std::vector<D2DTransmissionOptimizer::GlobalNpu>& assignment,
    size_t idx,
    const D2DTransmissionOptimizer::GlobalNpu& new_source) {
  if (idx >= assignment.size() || assignment[idx] == new_source) {
    return 0;
  }
  const auto& old_source = assignment[idx];
  int before = 0;
  int after = 0;
  if (idx > 0) {
    before += assignment[idx - 1] == old_source ? 0 : 1;
    after += assignment[idx - 1] == new_source ? 0 : 1;
  }
  if (idx + 1 < assignment.size()) {
    before += old_source == assignment[idx + 1] ? 0 : 1;
    after += new_source == assignment[idx + 1] ? 0 : 1;
  }
  return after - before;
}

int compute_swap_boundary_delta(
    const std::vector<D2DTransmissionOptimizer::GlobalNpu>& assignment,
    size_t i,
    const D2DTransmissionOptimizer::GlobalNpu& source_i_after,
    size_t j,
    const D2DTransmissionOptimizer::GlobalNpu& source_j_after) {
  if (i >= assignment.size() || j >= assignment.size() || i == j) {
    return 0;
  }
  const int n = static_cast<int>(assignment.size());
  std::unordered_set<int> affected_right_indices;
  std::array<int, 4> candidates = {static_cast<int>(i),
                                   static_cast<int>(i) + 1,
                                   static_cast<int>(j),
                                   static_cast<int>(j) + 1};
  for (int p : candidates) {
    if (p > 0 && p < n) {
      affected_right_indices.insert(p);
    }
  }
  auto source_after_at =
      [&](size_t idx) -> const D2DTransmissionOptimizer::GlobalNpu& {
    if (idx == i) {
      return source_i_after;
    }
    if (idx == j) {
      return source_j_after;
    }
    return assignment[idx];
  };
  int before = 0;
  int after = 0;
  for (int right_idx : affected_right_indices) {
    size_t left = static_cast<size_t>(right_idx - 1);
    size_t right = static_cast<size_t>(right_idx);
    before += assignment[left] == assignment[right] ? 0 : 1;
    after += source_after_at(left) == source_after_at(right) ? 0 : 1;
  }
  return after - before;
}

void normalize_expert_to_sources(
    const std::vector<int>& sorted_experts,
    const std::unordered_map<int,
                             std::vector<D2DTransmissionOptimizer::GlobalNpu>>&
        expert_to_src,
    std::unordered_map<int, std::vector<D2DTransmissionOptimizer::GlobalNpu>>*
        normalized_expert_to_src) {
  CHECK(normalized_expert_to_src != nullptr);
  normalized_expert_to_src->clear();
  for (int expert_id : sorted_experts) {
    auto it = expert_to_src.find(expert_id);
    if (it == expert_to_src.end()) {
      continue;
    }
    auto sources = it->second;
    std::sort(sources.begin(), sources.end(), global_npu_less);
    sources.erase(std::unique(sources.begin(), sources.end()), sources.end());
    if (!sources.empty()) {
      normalized_expert_to_src->emplace(expert_id, std::move(sources));
    }
  }
}

}  // namespace

void D2DTransmissionOptimizer::MaxFlow::init(int n, int ss, int tt) {
  g.assign(n, {});
  level.assign(n, 0);
  it.assign(n, 0);
  s = ss;
  t = tt;
}

void D2DTransmissionOptimizer::MaxFlow::add_edge(int u, int v, int c) {
  Edge a{v, (int)g[v].size(), c};
  Edge b{u, (int)g[u].size(), 0};
  g[u].push_back(a);
  g[v].push_back(b);
}

bool D2DTransmissionOptimizer::MaxFlow::bfs() {
  std::fill(level.begin(), level.end(), -1);
  std::vector<int> q;
  q.reserve(g.size());
  level[s] = 0;
  q.push_back(s);
  for (size_t i = 0; i < q.size(); ++i) {
    int v = q[i];
    for (const auto& e : g[v]) {
      if (e.cap > 0 && level[e.to] < 0) {
        level[e.to] = level[v] + 1;
        q.push_back(e.to);
      }
    }
  }
  return level[t] >= 0;
}

int D2DTransmissionOptimizer::MaxFlow::dfs(int v, int f) {
  if (v == t) return f;
  for (int& i = it[v]; i < (int)g[v].size(); ++i) {
    Edge& e = g[v][i];
    if (e.cap > 0 && level[v] < level[e.to]) {
      int d = dfs(e.to, std::min(f, e.cap));
      if (d > 0) {
        e.cap -= d;
        g[e.to][e.rev].cap += d;
        return d;
      }
    }
  }
  return 0;
}

int D2DTransmissionOptimizer::MaxFlow::dinic() {
  int flow = 0;
  while (bfs()) {
    std::fill(it.begin(), it.end(), 0);
    int f;
    while ((f = dfs(s, 1e9)) > 0) {
      flow += f;
    }
  }
  return flow;
}

bool D2DTransmissionOptimizer::feasible(
    int K,
    const std::vector<int>& required_per_target,
    const std::unordered_map<int, std::vector<GlobalNpu>>& expert_to_src,
    std::vector<GlobalNpu>& npu_index_map,
    std::vector<int>& req_to_expert_id,
    MaxFlow& mf) {
  // 1. Establish mapping from NPU to index (fixed order)
  std::unordered_map<GlobalNpu, int, GlobalNpuHasher> npu_to_idx;
  npu_index_map.clear();
  std::vector<GlobalNpu> all_npus;
  for (int expert_id : required_per_target) {
    auto it = expert_to_src.find(expert_id);
    if (it == expert_to_src.end()) {
      continue;
    }
    all_npus.insert(all_npus.end(), it->second.begin(), it->second.end());
  }
  std::sort(all_npus.begin(), all_npus.end(), global_npu_less);
  all_npus.erase(std::unique(all_npus.begin(), all_npus.end()), all_npus.end());
  npu_index_map = std::move(all_npus);
  for (size_t i = 0; i < npu_index_map.size(); ++i) {
    npu_to_idx.emplace(npu_index_map[i], static_cast<int>(i));
  }

  int npu_num = npu_index_map.size();
  int total_req = (int)required_per_target.size();

  // 2. Initialize the flow graph
  // S: 0 | NPU nodes: 1..npu_num | Req nodes: npu_num+1..npu_num+total_req | T:
  // npu_num+total_req+1
  int s = 0;
  int npu_offset = 1;
  int req_offset = npu_offset + npu_num;
  int t = req_offset + total_req;
  mf.init(t + 1, s, t);

  // S -> NPU (Capacity: K, limit maximum output per card)
  for (int i = 0; i < npu_num; ++i) {
    mf.add_edge(s, npu_offset + i, K);
  }

  // NPU -> Req & Req -> T
  req_to_expert_id.clear();
  for (int i = 0; i < total_req; ++i) {
    int expert_id = required_per_target[i];
    int cur_req_node = req_offset + i;
    req_to_expert_id.push_back(expert_id);

    if (expert_to_src.count(expert_id)) {
      for (const auto& gn : expert_to_src.at(expert_id)) {
        mf.add_edge(npu_offset + npu_to_idx[gn], cur_req_node, 1);
      }
    }
    mf.add_edge(cur_req_node, t, 1);
  }

  return mf.dinic() >= total_req;
}

std::vector<D2DTransmissionOptimizer::Step>
D2DTransmissionOptimizer::extract_plan(
    const MaxFlow& mf,
    const std::vector<GlobalNpu>& npu_index_map,
    const std::vector<int>& req_to_expert_id,
    int npu_offset,
    int req_offset) {
  std::vector<Step> plan;
  // Iterate through all request nodes
  for (int i = 0; i < (int)req_to_expert_id.size(); ++i) {
    int req_node = req_offset + i;
    int expert_id = req_to_expert_id[i];
    bool found = false;

    // Find which NPU node flows to it
    for (const auto& edge : mf.g[req_node]) {
      // In the residual network, edge.cap > 0 for reverse edges indicates that
      // the forward edge flow > 0
      int u = edge.to;
      if (u >= npu_offset && u < npu_offset + (int)npu_index_map.size()) {
        // Check if the corresponding forward edge is out of capacity
        for (const auto& forward_e : mf.g[u]) {
          if (forward_e.to == req_node && forward_e.cap == 0) {
            plan.push_back({npu_index_map[u - npu_offset], expert_id});
            found = true;
            break;
          }
        }
      }
      if (found) {
        break;
      }
    }
    if (!found) {
      LOG(ERROR) << "Failed to extract source for expert_id=" << expert_id
                 << ", req_node=" << req_node;
    }
  }
  return plan;
}

bool D2DTransmissionOptimizer::build_initial_assignment_from_plan(
    const std::vector<int>& sorted_experts,
    const std::vector<Step>& plan,
    const std::unordered_map<int, std::vector<GlobalNpu>>& expert_to_src,
    std::vector<GlobalNpu>* assignment,
    std::unordered_map<GlobalNpu, int, GlobalNpuHasher>* source_loads) const {
  CHECK(assignment != nullptr);
  CHECK(source_loads != nullptr);
  assignment->clear();
  source_loads->clear();
  assignment->reserve(sorted_experts.size());

  std::unordered_map<int, GlobalNpu> selected_source_by_expert;
  selected_source_by_expert.reserve(plan.size());
  for (const auto& step : plan) {
    auto it = selected_source_by_expert.find(step.expert_id);
    if (it != selected_source_by_expert.end()) {
      if (!(it->second == step.src)) {
        LOG(WARNING) << "Expert has multiple selected sources in initial plan. "
                     << "expert_id=" << step.expert_id
                     << ", keep=" << format_global_npu(it->second)
                     << ", drop=" << format_global_npu(step.src);
      }
      continue;
    }
    selected_source_by_expert.emplace(step.expert_id, step.src);
  }

  for (int expert_id : sorted_experts) {
    auto candidates_it = expert_to_src.find(expert_id);
    if (candidates_it == expert_to_src.end() || candidates_it->second.empty()) {
      LOG(ERROR) << "No candidate source for expert " << expert_id;
      return false;
    }
    const auto& candidates = candidates_it->second;
    GlobalNpu selected_source = candidates.front();

    auto selected_it = selected_source_by_expert.find(expert_id);
    if (selected_it == selected_source_by_expert.end()) {
      LOG(WARNING) << "Missing selected source in initial plan for expert "
                   << expert_id << ", fallback to "
                   << format_global_npu(selected_source);
    } else if (!contains_source(candidates, selected_it->second)) {
      LOG(WARNING) << "Initial plan chose invalid source for expert "
                   << expert_id
                   << ", selected=" << format_global_npu(selected_it->second)
                   << ", fallback=" << format_global_npu(selected_source);
    } else {
      selected_source = selected_it->second;
    }
    assignment->push_back(selected_source);
    ++(*source_loads)[selected_source];
  }
  return true;
}

int D2DTransmissionOptimizer::count_source_boundaries(
    const std::vector<GlobalNpu>& assignment) const {
  int boundaries = 0;
  for (size_t i = 1; i < assignment.size(); ++i) {
    boundaries += assignment[i] == assignment[i - 1] ? 0 : 1;
  }
  return boundaries;
}

bool D2DTransmissionOptimizer::try_single_move_improve(
    const std::vector<int>& sorted_experts,
    const std::unordered_map<int, std::vector<GlobalNpu>>& expert_to_src,
    int best_k,
    std::vector<GlobalNpu>* assignment,
    std::unordered_map<GlobalNpu, int, GlobalNpuHasher>* source_loads,
    int* improved_moves) const {
  CHECK(assignment != nullptr);
  CHECK(source_loads != nullptr);
  CHECK(improved_moves != nullptr);

  bool changed = false;
  for (size_t i = 0; i < assignment->size(); ++i) {
    int expert_id = sorted_experts[i];
    auto candidates_it = expert_to_src.find(expert_id);
    if (candidates_it == expert_to_src.end() || candidates_it->second.empty()) {
      continue;
    }
    const auto& candidates = candidates_it->second;
    const GlobalNpu old_source = (*assignment)[i];

    bool found_better = false;
    int best_delta = 0;
    int best_candidate_load = std::numeric_limits<int>::max();
    GlobalNpu best_candidate = old_source;
    for (const auto& candidate : candidates) {
      if (candidate == old_source) {
        continue;
      }
      int candidate_load = get_source_load(*source_loads, candidate);
      if (candidate_load + 1 > best_k) {
        continue;
      }
      int delta =
          compute_single_change_boundary_delta(*assignment, i, candidate);
      if (delta >= 0) {
        continue;
      }
      if (!found_better || delta < best_delta ||
          (delta == best_delta && candidate_load < best_candidate_load) ||
          (delta == best_delta && candidate_load == best_candidate_load &&
           global_npu_less(candidate, best_candidate))) {
        found_better = true;
        best_delta = delta;
        best_candidate_load = candidate_load;
        best_candidate = candidate;
      }
    }
    if (!found_better) {
      continue;
    }

    auto old_it = source_loads->find(old_source);
    if (old_it == source_loads->end() || old_it->second <= 0) {
      LOG(ERROR) << "Invalid old source load state when applying single move. "
                 << "source=" << format_global_npu(old_source)
                 << ", expert_id=" << expert_id;
      continue;
    }
    --old_it->second;
    if (old_it->second == 0) {
      source_loads->erase(old_it);
    }
    ++(*source_loads)[best_candidate];
    (*assignment)[i] = best_candidate;
    ++(*improved_moves);
    changed = true;
  }
  return changed;
}

bool D2DTransmissionOptimizer::try_pair_swap_improve(
    const std::vector<int>& sorted_experts,
    const std::unordered_map<int, std::vector<GlobalNpu>>& expert_to_src,
    int best_k,
    std::vector<GlobalNpu>* assignment,
    std::unordered_map<GlobalNpu, int, GlobalNpuHasher>* source_loads,
    int* improved_swaps) const {
  CHECK(assignment != nullptr);
  CHECK(source_loads != nullptr);
  CHECK(improved_swaps != nullptr);

  bool changed = false;
  for (size_t i = 0; i < assignment->size(); ++i) {
    int expert_id = sorted_experts[i];
    auto i_candidates_it = expert_to_src.find(expert_id);
    if (i_candidates_it == expert_to_src.end() ||
        i_candidates_it->second.empty()) {
      continue;
    }
    const auto& i_candidates = i_candidates_it->second;
    const GlobalNpu old_source_i = (*assignment)[i];

    bool found_better = false;
    int best_delta = 0;
    size_t best_swap_idx = 0;
    GlobalNpu best_source_i_after = old_source_i;

    for (const auto& target_source : i_candidates) {
      if (target_source == old_source_i) {
        continue;
      }
      int target_load = get_source_load(*source_loads, target_source);
      if (target_load < best_k) {
        continue;
      }
      int move_delta =
          compute_single_change_boundary_delta(*assignment, i, target_source);
      if (move_delta >= 0) {
        continue;
      }

      for (size_t j = 0; j < assignment->size(); ++j) {
        if (j == i || !((*assignment)[j] == target_source)) {
          continue;
        }
        int expert_j = sorted_experts[j];
        auto j_candidates_it = expert_to_src.find(expert_j);
        if (j_candidates_it == expert_to_src.end() ||
            j_candidates_it->second.empty()) {
          continue;
        }
        if (!contains_source(j_candidates_it->second, old_source_i)) {
          continue;
        }

        int delta = compute_swap_boundary_delta(
            *assignment, i, target_source, j, old_source_i);
        if (delta >= 0) {
          continue;
        }
        if (!found_better || delta < best_delta ||
            (delta == best_delta &&
             global_npu_less(target_source, best_source_i_after)) ||
            (delta == best_delta && target_source == best_source_i_after &&
             j < best_swap_idx)) {
          found_better = true;
          best_delta = delta;
          best_swap_idx = j;
          best_source_i_after = target_source;
        }
      }
    }

    if (!found_better) {
      continue;
    }
    (*assignment)[best_swap_idx] = old_source_i;
    (*assignment)[i] = best_source_i_after;
    ++(*improved_swaps);
    changed = true;
  }
  return changed;
}

std::vector<D2DTransmissionOptimizer::Step>
D2DTransmissionOptimizer::apply_contiguity_refinement_under_k(
    const std::vector<int>& sorted_experts,
    const std::unordered_map<int, std::vector<GlobalNpu>>& expert_to_src,
    const std::vector<Step>& plan,
    int best_k,
    ContiguityRefineStats* stats) const {
  ContiguityRefineStats local_stats;
  if (stats == nullptr) {
    stats = &local_stats;
  }
  *stats = ContiguityRefineStats{};

  std::vector<GlobalNpu> assignment;
  std::unordered_map<GlobalNpu, int, GlobalNpuHasher> source_loads;
  if (!build_initial_assignment_from_plan(
          sorted_experts, plan, expert_to_src, &assignment, &source_loads)) {
    LOG(ERROR)
        << "Failed to build initial assignment for contiguity refinement.";
    return plan;
  }

  stats->boundary_before = count_source_boundaries(assignment);
  int improved_moves = 0;
  int improved_swaps = 0;
  constexpr int k_max_rounds = 3;
  for (int round = 0; round < k_max_rounds; ++round) {
    bool round_changed = false;
    round_changed |= try_single_move_improve(sorted_experts,
                                             expert_to_src,
                                             best_k,
                                             &assignment,
                                             &source_loads,
                                             &improved_moves);
    round_changed |= try_pair_swap_improve(sorted_experts,
                                           expert_to_src,
                                           best_k,
                                           &assignment,
                                           &source_loads,
                                           &improved_swaps);
    if (!round_changed) {
      break;
    }
  }
  stats->boundary_after = count_source_boundaries(assignment);
  stats->improved_moves = improved_moves;
  stats->improved_swaps = improved_swaps;
  stats->max_source_load_after = count_max_source_load(source_loads);

  std::vector<Step> refined_plan;
  refined_plan.reserve(sorted_experts.size());
  for (size_t i = 0; i < sorted_experts.size(); ++i) {
    refined_plan.push_back(Step{assignment[i], sorted_experts[i]});
  }
  return refined_plan;
}

std::vector<D2DTransmissionOptimizer::Step>
D2DTransmissionOptimizer::optimize_layer(
    const std::vector<int>& required_per_target,
    const std::unordered_map<int, std::vector<GlobalNpu>>& expert_to_src) {
  int total_req = (int)required_per_target.size();
  if (total_req == 0) {
    return {};
  }

  std::vector<int> sorted_required_experts = required_per_target;
  std::sort(sorted_required_experts.begin(), sorted_required_experts.end());

  std::unordered_map<int, std::vector<GlobalNpu>> normalized_expert_to_src;
  normalize_expert_to_sources(
      sorted_required_experts, expert_to_src, &normalized_expert_to_src);
  for (int expert_id : sorted_required_experts) {
    auto it = normalized_expert_to_src.find(expert_id);
    if (it == normalized_expert_to_src.end() || it->second.empty()) {
      LOG(ERROR) << "No available source for required expert " << expert_id;
      return {};
    }
  }

  int low = 1, high = total_req;
  int bestK = high;

  // Cache the data required for the final result
  std::vector<GlobalNpu> final_npu_map;
  std::vector<int> final_req_to_expert;
  MaxFlow final_mf;

  while (low <= high) {
    int mid = low + (high - low) / 2;
    std::vector<GlobalNpu> tmp_npu_map;
    std::vector<int> tmp_req_expert;
    MaxFlow tmp_mf;

    if (feasible(mid,
                 sorted_required_experts,
                 normalized_expert_to_src,
                 tmp_npu_map,
                 tmp_req_expert,
                 tmp_mf)) {
      bestK = mid;
      high = mid - 1;
      // Record the state of the current optimal solution
      final_npu_map = std::move(tmp_npu_map);
      final_req_to_expert = std::move(tmp_req_expert);
      final_mf = std::move(tmp_mf);
    } else {
      low = mid + 1;
    }
  }

  if (final_req_to_expert.size() != sorted_required_experts.size()) {
    LOG(ERROR) << "Failed to find feasible assignment for all experts. "
               << "required=" << sorted_required_experts.size()
               << ", assigned=" << final_req_to_expert.size();
    return {};
  }

  auto initial_plan = extract_plan(final_mf,
                                   final_npu_map,
                                   final_req_to_expert,
                                   1,
                                   1 + (int)final_npu_map.size());
  std::sort(initial_plan.begin(),
            initial_plan.end(),
            [](const Step& lhs, const Step& rhs) {
              if (lhs.expert_id != rhs.expert_id) {
                return lhs.expert_id < rhs.expert_id;
              }
              if (lhs.src.instance != rhs.src.instance) {
                return lhs.src.instance < rhs.src.instance;
              }
              return lhs.src.local_npu < rhs.src.local_npu;
            });

  ContiguityRefineStats refine_stats;
  auto refined_plan =
      apply_contiguity_refinement_under_k(sorted_required_experts,
                                          normalized_expert_to_src,
                                          initial_plan,
                                          bestK,
                                          &refine_stats);
  std::sort(refined_plan.begin(),
            refined_plan.end(),
            [](const Step& lhs, const Step& rhs) {
              if (lhs.expert_id != rhs.expert_id) {
                return lhs.expert_id < rhs.expert_id;
              }
              if (lhs.src.instance != rhs.src.instance) {
                return lhs.src.instance < rhs.src.instance;
              }
              return lhs.src.local_npu < rhs.src.local_npu;
            });

  return refined_plan;
}

D2DTransmissionOptimizer::NonExpertStep
D2DTransmissionOptimizer::optimize_non_expert(
    const std::vector<Step>& expert_steps,
    const std::unordered_map<std::string, InstanceConfig>& instance_configs) {
  // 1. Calculate the expert transfer load for each individual NPU
  std::unordered_map<std::string, std::vector<int>> inst_npu_loads;
  for (const auto& [name, config] : instance_configs) {
    inst_npu_loads[name].assign(config.device_size, 0);
  }

  for (const auto& s : expert_steps) {
    if (inst_npu_loads.count(s.src.instance)) {
      inst_npu_loads[s.src.instance][s.src.local_npu]++;
    }
  }

  // 2. Find the DP group with optimal load (minimize the maximum expert load
  // within the group)
  std::string best_inst;
  int best_group_idx = -1;
  int best_dp_group_num = 0;
  int best_npu_per_group = 0;
  int min_max_load = 1e9;

  for (const auto& [inst_name, config] : instance_configs) {
    if (config.device_size <= 0 || config.dp_size <= 0) continue;
    if (config.device_size % config.dp_size != 0) continue;

    int num_dp_groups = config.dp_size;
    int npu_per_group = config.device_size / config.dp_size;
    const auto& loads = inst_npu_loads[inst_name];

    for (int g = 0; g < num_dp_groups; ++g) {
      int current_group_max_load = 0;

      // Calculate the maximum load of NPUs within this DP group
      // Load of the k-th NPU in the group: Load_{inst, g * npu_per_group + k}
      for (int i = 0; i < npu_per_group; ++i) {
        int local_npu = g * npu_per_group + i;
        if (local_npu < (int)loads.size()) {
          current_group_max_load =
              std::max(current_group_max_load, loads[local_npu]);
        }
      }

      // Update global optimum: Find the group with the smallest max(load)
      if (current_group_max_load < min_max_load) {
        min_max_load = current_group_max_load;
        best_inst = inst_name;
        best_group_idx = g;
        best_dp_group_num = config.dp_size;
        best_npu_per_group = npu_per_group;
      }
    }
  }

  // 3. Construct the result
  NonExpertStep result;
  if (best_group_idx != -1) {
    result.src_instance = best_inst;
    result.dp_group_index = best_group_idx;
    result.start_npu_index = best_group_idx * best_npu_per_group;
    result.dp_size = best_dp_group_num;
  }

  return result;
}
}  // namespace xllm_service
