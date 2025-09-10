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

namespace xllm_service {

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

}  // namespace xllm_service