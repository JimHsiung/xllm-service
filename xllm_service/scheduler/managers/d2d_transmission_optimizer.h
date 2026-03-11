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

#pragma once

#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace xllm_service {

class D2DTransmissionOptimizer {
 public:
  struct InstanceConfig {
    int device_size = 0;
    int dp_size = 0;
  };

  struct GlobalNpu {
    std::string instance;
    int local_npu;

    bool operator==(const GlobalNpu& o) const {
      return instance == o.instance && local_npu == o.local_npu;
    }
  };

  struct GlobalNpuHasher {
    std::size_t operator()(const GlobalNpu& gn) const {
      return std::hash<std::string>{}(gn.instance) ^
             (std::hash<int>{}(gn.local_npu) << 1);
    }
  };

  struct Step {
    GlobalNpu src;
    int expert_id;
  };

  struct NonExpertStep {
    std::string src_instance;
    int dp_group_index = -1;
    int start_npu_index = -1;
    int dp_size = 0;
  };

  std::vector<Step> optimize_layer(
      const std::vector<int>& required_per_target,
      const std::unordered_map<int, std::vector<GlobalNpu>>& expert_to_src);

  NonExpertStep optimize_non_expert(
      const std::vector<Step>& expert_steps,
      const std::unordered_map<std::string, InstanceConfig>& instance_configs);

 private:
  struct Edge {
    int to;
    int rev;
    int cap;
  };

  struct MaxFlow {
    std::vector<std::vector<Edge>> g;
    std::vector<int> level;
    std::vector<int> it;
    int s;
    int t;

    void init(int n, int ss, int tt);
    void add_edge(int u, int v, int c);
    bool bfs();
    int dfs(int v, int f);
    int dinic();
  };

  struct ContiguityRefineStats {
    int boundary_before = 0;
    int boundary_after = 0;
    int improved_moves = 0;
    int improved_swaps = 0;
    int max_source_load_after = 0;
  };

  bool feasible(
      int K,
      const std::vector<int>& required_per_target,
      const std::unordered_map<int, std::vector<GlobalNpu>>& expert_to_src,
      std::vector<GlobalNpu>& npu_index_map,
      std::vector<int>& req_to_expert_id,
      MaxFlow& mf);

  std::vector<Step> extract_plan(const MaxFlow& mf,
                                 const std::vector<GlobalNpu>& npu_index_map,
                                 const std::vector<int>& req_to_expert_id,
                                 int npu_offset,
                                 int req_offset);

  bool build_initial_assignment_from_plan(
      const std::vector<int>& sorted_experts,
      const std::vector<Step>& plan,
      const std::unordered_map<int, std::vector<GlobalNpu>>& expert_to_src,
      std::vector<GlobalNpu>* assignment,
      std::unordered_map<GlobalNpu, int, GlobalNpuHasher>* source_loads) const;

  int count_source_boundaries(const std::vector<GlobalNpu>& assignment) const;

  bool try_single_move_improve(
      const std::vector<int>& sorted_experts,
      const std::unordered_map<int, std::vector<GlobalNpu>>& expert_to_src,
      int best_k,
      std::vector<GlobalNpu>* assignment,
      std::unordered_map<GlobalNpu, int, GlobalNpuHasher>* source_loads,
      int* improved_moves) const;

  bool try_pair_swap_improve(
      const std::vector<int>& sorted_experts,
      const std::unordered_map<int, std::vector<GlobalNpu>>& expert_to_src,
      int best_k,
      std::vector<GlobalNpu>* assignment,
      std::unordered_map<GlobalNpu, int, GlobalNpuHasher>* source_loads,
      int* improved_swaps) const;

  std::vector<Step> apply_contiguity_refinement_under_k(
      const std::vector<int>& sorted_experts,
      const std::unordered_map<int, std::vector<GlobalNpu>>& expert_to_src,
      const std::vector<Step>& plan,
      int best_k,
      ContiguityRefineStats* stats) const;
};

}  // namespace xllm_service
