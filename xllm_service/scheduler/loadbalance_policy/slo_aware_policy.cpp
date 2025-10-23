/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include "slo_aware_policy.h"

#include "common/global_gflags.h"

namespace xllm_service {

SloAwarePolicy::SloAwarePolicy(const Options& options,
                               std::shared_ptr<InstanceMgr> instance_mgr)
    : options_(options), LoadBalancePolicy(instance_mgr) {
  if (FLAGS_enable_instance_update) {
    update_instance_thread_ =
        std::make_unique<std::thread>(&SloAwarePolicy::update_instance, this);
  }
}

SloAwarePolicy::~SloAwarePolicy() {
  exited_ = true;
  if (update_instance_thread_ && update_instance_thread_->joinable()) {
    update_instance_thread_->join();
  }
}

bool SloAwarePolicy::select_instances_pair(std::shared_ptr<Request> request) {
  if (request->token_ids.empty()) {
    return instance_mgr_->get_next_instance_pair(&request->routing);
  }

  // select instances pair based on slo
  if (!instance_mgr_->select_instance_pair_on_slo(&request->routing)) {
    LOG(ERROR) << "Select instances based on the SLO failed!";
    return false;
  }

  // update estimated ttft
  auto& ttft_predictor =
      instance_mgr_->get_ttft_predictor(request->routing.prefill_name);
  request->estimated_ttft =
      ttft_predictor.predict_ttft(request->token_ids.size());

  return true;
}

void SloAwarePolicy::update_instance() {
  while (!exited_) {
    std::this_thread::sleep_for(std::chrono::milliseconds(
        static_cast<int64_t>(options_.instance_update_interval() * 1000)));

    InstanceMetrics metrics;
    instance_mgr_->get_instance_metrics(metrics);

    // When the number of prefill instances or decode instances is insufficient,
    // skip the update.
    if (metrics.prefill_instance_num == 0 || metrics.decode_instance_num == 0) {
      continue;
    }

    // Calculate the current instance's load based on the instance's current
    // lantency
    InstanceLoad prefill_load;
    float ttft_threshold =
        (metrics.prefill_instance_num - 1) / metrics.prefill_instance_num;
    if (metrics.avg_recent_ttft > FLAGS_target_ttft) {
      prefill_load = InstanceLoad::HIGH;
    } else if (metrics.avg_recent_ttft <= FLAGS_target_ttft &&
               metrics.avg_recent_ttft > FLAGS_target_ttft * ttft_threshold) {
      prefill_load = InstanceLoad::MEDIUM;
    } else if (metrics.avg_recent_ttft <= FLAGS_target_ttft * ttft_threshold &&
               metrics.avg_recent_ttft > 0) {
      prefill_load = InstanceLoad::LOW;
    } else {
      prefill_load = InstanceLoad::IDLE;
    }

    InstanceLoad decode_load;
    float tbt_threshold =
        (metrics.decode_instance_num - 1) / metrics.decode_instance_num;
    if (metrics.avg_recent_tbt > FLAGS_target_tpot) {
      decode_load = InstanceLoad::HIGH;
    } else if (metrics.avg_recent_tbt <= FLAGS_target_tpot &&
               metrics.avg_recent_tbt > FLAGS_target_tpot * tbt_threshold) {
      decode_load = InstanceLoad::MEDIUM;
    } else if (metrics.avg_recent_tbt <= FLAGS_target_tpot * tbt_threshold &&
               metrics.avg_recent_tbt > 0) {
      decode_load = InstanceLoad::LOW;
    } else {
      decode_load = InstanceLoad::IDLE;
    }

    // recode decode load
    decode_load_sliding_window_.emplace_back(decode_load);
    if (decode_load_sliding_window_.size() >
        FLAGS_decode_load_sliding_window_size) {
      decode_load_sliding_window_.pop_front();
    }
    bool all_decode_load_high = std::all_of(
        decode_load_sliding_window_.begin(),
        decode_load_sliding_window_.end(),
        [](const InstanceLoad& load) { return load == InstanceLoad::HIGH; });

    // check if instance needs to be flipped
    if (prefill_load == InstanceLoad::HIGH &&
        decode_load <= InstanceLoad::LOW &&
        prefill_to_decode_cooldown_count_ == 0) {
      // If prefill instances have high load and decode instances have low load,
      // flip the decode instance with the lowest load to become a prefill
      // instance.
      prefill_to_decode_cooldown_count_ = FLAGS_instance_flip_cooldown_count;
      instance_mgr_->flip_decode_to_prefill(metrics.min_recent_ttft_instance);
    } else if (prefill_load <= InstanceLoad::LOW && all_decode_load_high &&
               decode_to_prefill_cooldown_count_ == 0) {
      // If decode instances have high load, directly flip the prefill instance
      // with the lowest load to decode.
      decode_to_prefill_cooldown_count_ = FLAGS_instance_flip_cooldown_count;
      instance_mgr_->flip_prefill_to_decode(metrics.min_recent_tbt_instance);
    } else {
      // update cooldown count
      if (prefill_to_decode_cooldown_count_ > 0) {
        prefill_to_decode_cooldown_count_--;
      }

      if (decode_to_prefill_cooldown_count_ > 0) {
        decode_to_prefill_cooldown_count_--;
      }
    }
  }
}

}  // namespace xllm_service