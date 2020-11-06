// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include "myriad_layers_tests.hpp"

VPU_DECLARE_ENUM(InterpolateCoordTransMode,
    half_pixel = 0,
    pytorch_half_pixel = 1,
    asymmetric = 2,
    tf_half_pixel_for_nn = 3,
    align_corners = 4
)
VPU_DECLARE_ENUM(InterpolateNearestMode,
    round_prefer_floor = 0,
    round_prefer_ceil = 1,
    floor = 2,
    ceil = 3,
    simple = 4
)

using namespace InferenceEngine;

#define ERROR_BOUND 1e-3

PRETTY_PARAM(Factor, float)
PRETTY_PARAM(Antialias, int)
PRETTY_PARAM(CoordinateTransformMode, InterpolateCoordTransMode)
PRETTY_PARAM(NearestMode, InterpolateNearestMode)
PRETTY_PARAM(HwOptimization, bool);
PRETTY_PARAM(CustomConfig, std::string);

typedef myriadLayerTestBaseWithParam<std::tuple<SizeVector, Factor, Antialias, CoordinateTransformMode, NearestMode, HwOptimization, CustomConfig>>
	myriadResampleLayerTests_smoke;

static inline float triangleCoeff(float x)
{
    return (1.0f - fabsf(x));
}

void refResample(const Blob::Ptr src, Blob::Ptr dst, int antialias, InterpolateCoordTransMode coordTransMode, InterpolateNearestMode nearestMode) {
    ie_fp16 *src_data = static_cast<ie_fp16*>(src->buffer());
    ie_fp16 *output_sequences = static_cast<ie_fp16*>(dst->buffer());
    ASSERT_NE(src_data, nullptr);
    ASSERT_NE(output_sequences, nullptr);

    const auto& src_dims = src->getTensorDesc().getDims();
    const auto& dst_dims = dst->getTensorDesc().getDims();
    int OH = dst_dims[2];
    int OW = dst_dims[3];

    int C  = src_dims[1];
    int IH = src_dims[2];
    int IW = src_dims[3];

    if (IH == OH && IW == OW)
    {
    	std::copy(src_data, src_data + C*IH*IW, output_sequences);
        return;
    }

    const float fy = static_cast<float>(IH) / static_cast<float>(OH);
    const float fx = static_cast<float>(IW) / static_cast<float>(OW);

    float ax = 1.0f / fx; // scale
    float ay = 1.0f / fy;

    int rx = (fx < 1.0f) ? 2 : ceil((1.0f)/ax);
    int ry = (fy < 1.0f) ? 2 : ceil((1.0f)/ay);

    for (int c = 0; c < C; c++)
    {
        const ie_fp16* in_ptr = src_data + IW*IH*c;
        ie_fp16* out_ptr = output_sequences + OW*OH*c;

        for (int oy = 0; oy < OH; oy++)
        {
            for (int ox = 0; ox < OW; ox++)
            {
                float ix = ox*fx + fx / 2.0f - 0.5f; // half pixel
                float iy = oy*fy + fy / 2.0f - 0.5f;

                int ix_r = (int)(round(ix)); // round prefer ceil
                int iy_r = (int)(round(iy));

                if (coordTransMode == InterpolateCoordTransMode::asymmetric) { // asymmetric
                    ix = ox*fx;
                    iy = oy*fy;
                }
                if (nearestMode == InterpolateNearestMode::floor) { // floor
                    ix_r = (int)(floor(ix));
                    iy_r = (int)(floor(iy));
                }

                float sum=0;
                float wsum=0;

                if(antialias){
                    for (int y = iy_r - ry; y <= iy_r + ry; y++)
                    {
                        for (int x = ix_r - rx; x <= ix_r + rx; x++)
                        {
                            if (y < 0 || x < 0) continue;
                            if (y >= (int)IH || x >= (int)IW) continue;

                            float dx = ix - x;
                            float dy = iy - y;

                            float w = ax*triangleCoeff(ax*dx) * ay * triangleCoeff(ay*dy);

                            sum += w * PrecisionUtils::f16tof32(in_ptr[y*IW + x]);
                            wsum += w;
                        }
                    }
                    out_ptr[oy * OW + ox] = PrecisionUtils::f32tof16((!wsum) ? 0.0f : (sum / wsum));
                }
                else{
                    out_ptr[oy * OW + ox] = in_ptr[iy_r * IW + ix_r];
                }
            }
        }
    }
}

TEST_P(myriadResampleLayerTests_smoke, Resample) {
    const SizeVector inputDims = std::get<0>(GetParam());
    const float factor = std::get<1>(GetParam());
    const bool antialias = std::get<2>(GetParam());
    const InterpolateCoordTransMode coordTransMode = std::get<3>(GetParam());
    const InterpolateNearestMode nearestMode = std::get<4>(GetParam());
    const bool hwOptimization = std::get<5>(GetParam());
    const std::string customConfig = std::get<6>(GetParam());

    ASSERT_GT(factor, 0);

    if (customConfig.empty() && antialias) {
        GTEST_SKIP() << "Native Resample with antialiasing is not supported";
    }

    if (!customConfig.empty() && !CheckMyriadX()) {
        GTEST_SKIP() << "Custom layers for MYRIAD2 not supported";
    }

    _config[InferenceEngine::MYRIAD_CUSTOM_LAYERS] = customConfig;

    const auto outputDims = SizeVector{inputDims[0],
                                       inputDims[1],
                                       (size_t)(inputDims[2] * factor),
                                       (size_t)(inputDims[3] * factor)};

    SetInputTensors({inputDims});
    SetOutputTensors({outputDims});

    std::map<std::string, std::string> params;
    params["antialias"] = std::to_string((int)antialias);
    params["coordTransMode"] = "half_pixel";
    if (coordTransMode == InterpolateCoordTransMode::asymmetric)
        params["coordTransMode"] = "asymmetric";
    params["nearestMode"] = "round_prefer_ceil";
    if (nearestMode == InterpolateNearestMode::floor)
        params["nearestMode"] = "floor";
    params["factor"] = std::to_string(factor);

    ASSERT_NO_FATAL_FAILURE(makeSingleLayerNetwork(LayerInitParams("Resample").params(params),
                                                   NetworkInitParams()
                                                        .useHWOpt(hwOptimization)
                                                        .lockLayout(true)));

    ASSERT_TRUE(Infer());

    ASSERT_NO_FATAL_FAILURE(refResample(_inputMap.begin()->second, _refBlob, antialias, coordTransMode, nearestMode));

    CompareCommonAbsolute(_outputMap.begin()->second, _refBlob, ERROR_BOUND);
}

static std::vector<SizeVector> s_ResampleInput = {
        {1, 128, 26, 26},
        {1, 64, 52, 52},
        {1, 23, 14, 14}
};

static std::vector<CustomConfig> s_CustomConfig = {
    {""},
#ifdef VPU_HAS_CUSTOM_KERNELS
   getIELibraryPath() + "/vpu_custom_kernels/customLayerBindings.xml"
#endif
};

