// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

using namespace InferenceEngine;

namespace vpu {
// void FrontEnd::parseLogicalNot(const Model &model, const NodePtr& node, const DataVector &inputs, const DataVector &outputs) const {
//     auto logicalNot = ngraph::as_type_ptr<ngraph::opset1::LogicalNot>(node);
//     // LayerParams params = {logicalNot->get_friendly_name(), "Eltwise", Precision::FP16};
//     // auto res = std::make_shared<EltwiseOperation>(params);

//     parseEltwise(model, logicalNot, inputs, outputs);
// }

// void FrontEnd::parseAbs(const Model &model, const ie::CNNLayerPtr &layer, const DataVector &inputs, const DataVector &outputs) const {
//     LayerParams params = {layer->name, "Eltwise", layer->precision};
//     auto res = std::make_shared<InferenceEngine::EltwiseLayer>(params);
//     res->_operation = InferenceEngine::EltwiseLayer::Abs;

//     parseEltwise(model, res, inputs, outputs);
// }

void FrontEnd::parseActivation(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const {
    const ie::details::caseless_map<std::string, LayerParser> activationParsers {
        {"not", LAYER_PARSER(parseLogicalNot)},
        {"abs", LAYER_PARSER(parseAbs)},
    };

    const auto type = node->get_type_name();

    const auto activationParserIt = activationParsers.find(type);
    VPU_THROW_UNSUPPORTED_LAYER_UNLESS(activationParserIt != activationParsers.end(),
                                 "Failed to compile layer \"%v\"(type = %v) ", node->get_friendly_name(), type);


    activationParserIt->second(model, node, inputs, outputs);
}

} // namespace vpu
