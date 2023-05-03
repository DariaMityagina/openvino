// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/utils/containers.hpp"
#include "vpu/configuration/options/myriad_timeout.hpp"
#include "vpu/configuration/plugin_configuration.hpp"
#include "vpu/utils/error.hpp"
#include <vpu/myriad_config.hpp>

namespace vpu {

void TimeoutMyriadValue::validate(const std::string& value) {
    if (value == defaultValue()) {
        return;
    }

    int intValue;
    try {
        intValue = std::stoi(value);
    } catch (const std::exception& e) {
        VPU_THROW_FORMAT(R"(unexpected {} option value "{}", must be a number)", key(), value);
    }

    VPU_THROW_UNLESS(intValue >= 0,
        R"(unexpected {} option value "{}", only not negative numbers are supported)", key(), value);
}

void TimeoutMyriadValue::validate(const PluginConfiguration& configuration) {
    validate(configuration[key()]);
}

std::string TimeoutMyriadValue::key() {
    return InferenceEngine::MYRIAD_TIMEOUT;
}

details::Access TimeoutMyriadValue::access() {
    return details::Access::Public;
}

details::Category TimeoutMyriadValue::category() {
    return details::Category::CompileTime;
}

std::string TimeoutMyriadValue::defaultValue() {
    return 0;
}

TimeoutMyriadValue::value_type TimeoutMyriadValue::parse(const std::string& value) {
    if (value == defaultValue()) {
        return TimeoutMyriadValue::value_type();
    }

    int intValue;
    try {
        intValue = std::stoi(value);
    } catch (const std::exception& e) {
        VPU_THROW_FORMAT(R"(unexpected {} option value "{}", must be a number)", key(), value);
    }

    VPU_THROW_UNSUPPORTED_OPTION_UNLESS(intValue >= 0,
        R"(unexpected {} option value "{}", only not negative numbers are supported)", key(), value);
    return intValue;
}

}  // namespace vpu
