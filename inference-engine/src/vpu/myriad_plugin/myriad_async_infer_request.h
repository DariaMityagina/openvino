// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpp_interfaces/impl/ie_infer_async_request_thread_safe_default.hpp"
#include "myriad_infer_request.h"

namespace vpu {
namespace MyriadPlugin {

class MyriadAsyncInferRequest : public InferenceEngine::AsyncInferRequestThreadSafeDefault {
public:
    MyriadAsyncInferRequest(const InferenceEngine::IInferRequestInternal::Ptr &inferRequest,
                            const InferenceEngine::ITaskExecutor::Ptr &taskExecutor,
                            const InferenceEngine::ITaskExecutor::Ptr &callbackExecutor);

    ~MyriadAsyncInferRequest();
private:
    InferenceEngine::ITaskExecutor::Ptr _taskExecutorGetResult;
};

}  // namespace MyriadPlugin
}  // namespace vpu
