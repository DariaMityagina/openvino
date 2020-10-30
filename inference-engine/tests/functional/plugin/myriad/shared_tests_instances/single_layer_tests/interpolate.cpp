// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/interpolate.hpp"
#include <vpu/private_plugin_config.hpp>
#include "common_test_utils/test_constants.hpp"
#include "common/myriad_common_test_utils.hpp"

#include <vector>

using namespace LayerTestsDefinitions;

namespace {

typedef std::map<std::string, std::string> Config;

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP16,
};

std::vector<size_t> axes = {2, 3};
std::vector<float> scales = {1.25f, 1.5f};
std::vector<std::vector<size_t>> layerParams = {
    // {1, 1, 1, 1},
    // {12, 4, 3, 2},
    // {128, 128, 12, 1}
    {1, 1, 40, 40}
    // {3, 1, 12, 4},
    // {12, 1, 128, 128}
};
std::vector<std::vector<size_t>> layerParamsOut = {
    // {1, 1, 1, 1},
    // {24, 16, 3, 2},
    // {256, 512, 12, 1}
    {1, 1, 50, 60}
    // {3, 1, 12, 4},
    // {12, 1, 128, 128}
};

// INSTANTIATE_TEST_CASE_P(smoke_interpolate,
//                         InterpolateLayerTest1,
//                         ::testing::Combine(
//                             ::testing::ValuesIn(netPrecisions),
//                             ::testing::Values(InferenceEngine::Layout::ANY),
//                             ::testing::Values(InferenceEngine::Layout::ANY),
//                             ::testing::ValuesIn(layerParams),
//                             ::testing::ValuesIn(layerParamsOut),
//                             ::testing::Values(CommonTestUtils::DEVICE_MYRIAD)),
//                         InterpolateLayerTest1::getTestCaseName);

}  // namespace
