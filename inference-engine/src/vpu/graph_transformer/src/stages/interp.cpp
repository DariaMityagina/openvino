// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <vector>
#include <unordered_set>
#include <memory>
#include <set>

namespace vpu {

namespace {

class InterpStage final : public StageNode {
public:
    using StageNode::StageNode;

private:
    StagePtr cloneImpl() const override {
        return std::make_shared<InterpStage>(*this);
    }

    void propagateDataOrderImpl(StageDataInfo<DimsOrder>& orderInfo) override {
        auto input = inputEdge(0)->input();

        orderInfo.setOutput(outputEdge(0), input->desc().dimsOrder());
    }

    void getDataStridesRequirementsImpl(StageDataInfo<StridesRequirement>& stridesInfo) override {
    }

    void finalizeDataLayoutImpl() override {
    }

    void getBatchSupportInfoImpl(StageDataInfo<BatchSupport>& batchInfo) override {
        batchInfo.setInput(inputEdge(0), BatchSupport::Split);
        batchInfo.setOutput(outputEdge(0), BatchSupport::Split);
    }

    void initialCheckImpl() const override {
        assertInputsOutputsTypes(this, {{DataType::FP16}}, {{DataType::FP16}});
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        auto align_corners = attrs().get<bool>("align_corners");
        auto sampleType = attrs().get<InterpolateMode>("mode");
        auto coordinateTransformationMode = attrs().get<InterpolateCoordTransMode>("coordinate_transformation_mode");

        serializer.append(static_cast<int32_t>(align_corners));
        serializer.append(static_cast<uint32_t>(sampleType));
        serializer.append(static_cast<uint32_t>(coordinateTransformationMode));
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        auto input = inputEdge(0)->input();
        auto output = outputEdge(0)->output();

        input->serializeBuffer(serializer);
        output->serializeBuffer(serializer);
    }
};

}  // namespace

Stage StageBuilder::addInterpStage(
                        const Model& model,
                        const std::string& name,
                        const ie::CNNLayerPtr& layer,
                        bool align_corners,
                        InterpolateMode mode,
                        InterpolateCoordTransMode coordinateTransformationMode,
                        const Data& input,
                        const Data& output) {
    auto stage = model->addNewStage<InterpStage>(layer->name, StageType::Interp, layer, {input}, {output});
    stage->attrs().set<bool>("align_corners", align_corners);
    stage->attrs().set<InterpolateMode>("mode", mode);
    stage->attrs().set<InterpolateCoordTransMode>("coordinate_transformation_mode", coordinateTransformationMode);

    return stage;
}

void FrontEnd::parseInterp(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const {
    VPU_THROW_UNLESS(inputs.size() == 1,
                     "Interp stage with name {} must have only 1 input, "
                     "actually provided {}", layer->name, inputs.size());
    VPU_THROW_UNLESS(outputs.size() == 1,
                     "Interp stage with name {} must have only 1 output, "
                     "actually provided {}", layer->name, outputs.size());
    ie::details::CaselessEq<std::string> cmp;
    const auto coord = layer->GetParamAsString("coordinate_transformation_mode", "half_pixel");
    const auto interpMode = layer->GetParamAsString("mode", "linear");
    InterpolateCoordTransMode coordinateTransformationMode = InterpolateCoordTransMode::HalfPixel;
    InterpolateMode mode = InterpolateMode::Linear;

    if (cmp(coord, "asymmetric")) {
        coordinateTransformationMode = InterpolateCoordTransMode::Asymmetric;
    } else if (cmp(coord, "half_pixel")) {
        coordinateTransformationMode = InterpolateCoordTransMode::HalfPixel;
    } else if (cmp(coord, "pytorch_half_pixel")) {
        coordinateTransformationMode = InterpolateCoordTransMode::PytorchHalfPixel;
    } else if (cmp(coord, "tf_half_pixel_for_nn")) {
        coordinateTransformationMode = InterpolateCoordTransMode::TfHalfPixelForNn;
    } else if (cmp(coord, "align_corners")) {
        coordinateTransformationMode = InterpolateCoordTransMode::AlignCorners;
    }

    if (cmp(interpMode, "linear_onnx")) {
        mode = InterpolateMode::LinearOnnx;
    }

    _stageBuilder->addInterpStage(model, layer->name, layer, layer->GetParamAsInt("align_corners", 0), mode, coordinateTransformationMode, inputs[0], outputs[0]);
}

}  // namespace vpu
