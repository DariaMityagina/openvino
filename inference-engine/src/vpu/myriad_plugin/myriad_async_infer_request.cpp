// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include "myriad_async_infer_request.h"
#include <vpu/utils/profiling.hpp>

using namespace vpu::MyriadPlugin;
using namespace InferenceEngine;

MyriadAsyncInferRequest::MyriadAsyncInferRequest(const InferenceEngine::IInferRequestInternal::Ptr &inferRequest,
                                                 const InferenceEngine::ITaskExecutor::Ptr &taskExecutor,
                                                 const InferenceEngine::ITaskExecutor::Ptr &callbackExecutor)
: InferenceEngine::AsyncInferRequestThreadSafeDefault(inferRequest, taskExecutor, callbackExecutor) {
        static_cast<MyriadInferRequest*>(inferRequest.get())->SetAsyncRequest(this);
}

MyriadAsyncInferRequest::~MyriadAsyncInferRequest() {
    StopAndWait();
}
