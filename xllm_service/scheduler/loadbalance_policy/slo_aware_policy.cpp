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

  std::string prefill_instance;
  int64_t prefill_time;
  if (!instance_mgr_->get_min_prefill_time_instance(prefill_instance,
                                                    prefill_time)) {
    LOG(ERROR) << "Get min prefill time instance failed!";
    return false;
  }

  std::string decode_instance;
  int64_t decode_length;
  if (!instance_mgr_->get_min_decode_length_instance(decode_instance,
                                                     decode_length)) {
    LOG(ERROR) << "Get min decode length instance failed!";
    return false;
  }

  auto& ttft_predictor = instance_mgr_->get_ttft_predictor(prefill_instance);
  request->estimated_ttft =
      ttft_predictor.predict_ttft(request->token_ids.size());
  request->routing.prefill_name = prefill_instance;
  request->routing.decode_name = decode_instance;

  return true;
}

void SloAwarePolicy::update_instance() {
  while (!exited_) {
    std::this_thread::sleep_for(std::chrono::milliseconds(
        static_cast<int64_t>(options_.instance_update_interval() * 1000)));

    // When the number of prefill instances or decode instances is insufficient,
    // skip the update.
    if (!(instance_mgr_->get_prefill_instance_num() > 0 &&
          instance_mgr_->get_decode_instance_num() > 0)) {
      continue;
    }

    std::string min_load_prefill_instance;
    InstanceLoad min_prefill_load;
    if (!get_min_load_prefill_instance(min_load_prefill_instance,
                                       min_prefill_load)) {
      // Prefill instance is empty.
      continue;
    }

    std::string min_load_decode_instance;
    InstanceLoad min_decode_load;
    if (!get_min_load_decode_instance(min_load_decode_instance,
                                      min_decode_load)) {
      // Decode instance is empty.
      continue;
    }

    // recode decode load
    decode_load_sliding_window_.emplace_back(min_decode_load);
    if (decode_load_sliding_window_.size() >
        FLAGS_decode_load_sliding_window_size) {
      decode_load_sliding_window_.pop_front();
    }
    bool all_decode_load_high = std::all_of(
        decode_load_sliding_window_.begin(),
        decode_load_sliding_window_.end(),
        [](const InstanceLoad& load) { return load == InstanceLoad::HIGH; });

    // check if instance needs to be flipped
    if (min_prefill_load == InstanceLoad::HIGH &&
        min_decode_load <= InstanceLoad::LOW &&
        prefill_to_decode_cooldown_count_ == 0) {
      // If prefill instances have high load and decode instances have low load,
      // flip the decode instance with the lowest load to become a prefill
      // instance.
      prefill_to_decode_cooldown_count_ = FLAGS_instance_flip_cooldown_count;
      instance_mgr_->flip_decode_to_prefill(min_load_decode_instance);
    } else if (min_prefill_load <= InstanceLoad::LOW && all_decode_load_high &&
               decode_to_prefill_cooldown_count_ == 0) {
      // If decode instances have high load, directly flip the prefill instance
      // with the lowest load to decode.
      decode_to_prefill_cooldown_count_ = FLAGS_instance_flip_cooldown_count;
      instance_mgr_->flip_prefill_to_decode(min_load_prefill_instance);
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

bool SloAwarePolicy::get_min_load_prefill_instance(
    std::string& min_load_prefill_instance,
    InstanceLoad& min_prefill_load) {
  std::string min_prefill_time_instance;
  int64_t min_prefill_time;
  if (!instance_mgr_->get_min_prefill_time_instance(min_prefill_time_instance,
                                                    min_prefill_time)) {
    // Prefill instance is empty.
    return false;
  }

  std::string min_recent_ttft_instance;
  int64_t min_recent_ttft_time;
  if (!instance_mgr_->get_min_recent_ttft_instance(min_recent_ttft_instance,
                                                   min_recent_ttft_time)) {
    // Prefill instance is empty.
    return false;
  }

  // Set the larger value between the simulated prefill time and the recorded
  // TTFT as the current TTFT for the instance.
  int64_t instance_ttft;
  if (min_prefill_time < min_recent_ttft_time) {
    min_load_prefill_instance = min_recent_ttft_instance;
    instance_ttft = min_recent_ttft_time;
  } else {
    min_load_prefill_instance = min_prefill_time_instance;
    instance_ttft = min_prefill_time;
  }

  // Calculate the current instance's load based on the instance's current TTFT
  // and the target TTFT.
  if (instance_ttft > FLAGS_target_ttft) {
    min_prefill_load = InstanceLoad::HIGH;
  } else if (instance_ttft <= FLAGS_target_ttft &&
             instance_ttft > FLAGS_target_ttft * 0.5) {
    min_prefill_load = InstanceLoad::MEDIUM;
  } else if (instance_ttft <= FLAGS_target_ttft * 0.5 && instance_ttft > 0) {
    min_prefill_load = InstanceLoad::LOW;
  } else {
    min_prefill_load = InstanceLoad::IDLE;
  }

  return true;
}

bool SloAwarePolicy::get_min_load_decode_instance(
    std::string& min_load_decode_instance,
    InstanceLoad& min_decode_load) {
  int64_t min_recent_tbt_time;
  if (!instance_mgr_->get_min_recent_tbt_instance(min_load_decode_instance,
                                                  min_recent_tbt_time)) {
    // Decode instance is empty.
    return false;
  }

  if (min_recent_tbt_time > FLAGS_target_tpot) {
    min_decode_load = InstanceLoad::HIGH;
  } else if (min_recent_tbt_time <= FLAGS_target_tpot &&
             min_recent_tbt_time > FLAGS_target_tpot * 0.5) {
    min_decode_load = InstanceLoad::MEDIUM;
  } else if (min_recent_tbt_time <= FLAGS_target_tpot * 0.5 &&
             min_recent_tbt_time > 0) {
    min_decode_load = InstanceLoad::LOW;
  } else {
    min_decode_load = InstanceLoad::IDLE;
  }

  return true;
}

}  // namespace xllm_service