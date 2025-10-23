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

#pragma once

#include "common/options.h"
#include "common/types.h"
#include "loadbalance_policy.h"

namespace xllm_service {

class SloAwarePolicy final : public LoadBalancePolicy {
 public:
  SloAwarePolicy(const Options& options,
                 std::shared_ptr<InstanceMgr> instance_mgr);

  virtual ~SloAwarePolicy();

  bool select_instances_pair(std::shared_ptr<Request> request) override;

 private:
  DISALLOW_COPY_AND_ASSIGN(SloAwarePolicy);

  void update_instance();

  Options options_;

  bool exited_ = false;

  std::unique_ptr<std::thread> update_instance_thread_;

  std::deque<InstanceLoad> decode_load_sliding_window_;

  int32_t prefill_to_decode_cooldown_count_ = 0;

  int32_t decode_to_prefill_cooldown_count_ = 0;
};

}  // namespace xllm_service