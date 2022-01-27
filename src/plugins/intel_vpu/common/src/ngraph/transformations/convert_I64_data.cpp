// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/ngraph/transformations/convert_I64_data.hpp"
#include "vpu/ngraph/operations/dynamic_shape_resolver.hpp"
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <ngraph/opsets/opset6.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <memory>
#include <numeric>
#include "vpu/ngraph/utilities.hpp"
#include <vpu/utils/error.hpp>

#include "ngraph/graph_util.hpp"

// #define MORE
// #define MORE_NOT_ALL
#define MORE_ALL

NGRAPH_RTTI_DEFINITION(vpu::ConvertI64Data, "ConvertI64Data", 0);

namespace vpu {

ConvertI64Data::ConvertI64Data() {
    auto any_node = ngraph::pattern::any_input();

#ifdef MORE
    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        const auto& node = m.get_match_root();
        auto dst_type = ngraph::element::i32;
        auto src_type = ngraph::element::i64;
        auto new_in   = node->input_values();
        auto new_node = node;
        ov::NodeVector new_ops;
        std::string name = node->get_type_info().name;
        bool contains = name.find(std::string("Equal")) != std::string::npos;
        if (!contains) {
            if (!ngraph::op::is_output(node)) {
                for (int i = 0; i < node->input_values().size(); i++) {
                    auto input = node->input_value(i);
                    if (input.get_element_type() == src_type) {
                        std::cout << "[DEBUG] <Input> int64 = " << node->get_type_info().name << "\n";
                        auto convert = std::make_shared<ngraph::opset1::Convert>(input, dst_type);
                        new_in[i] = convert;
                        std::cout << "[DEBUG] <Input> Type after : " << new_in[i].get_element_type() << "\n";
                    }
                    if (name.find(std::string("DynamicShapeResolver")) != std::string::npos) {
                        auto dynamic = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(new_in[i],
                                        node->outputs()[0]);
                        new_in[i] = dynamic;
                    }
                }
                for (int i = 0; i < new_node->outputs().size(); i++) {
                    auto output = node->output(i);
                    if (output.get_element_type() == src_type) {
                        std::cout << "[DEBUG] <Output> int64 : " << node->get_name() << ", " << node->get_type_info().name << "\n";
                        if (name.find(std::string("Static")) == std::string::npos) {
                            if (output.get_partial_shape().rank().is_static()) {
                                std::cout << "[DEBUG]    <Output> static : " << output.get_partial_shape() << "\n";
                                new_node->set_output_type(i, dst_type, output.get_partial_shape());
                            } else {
                                std::cout << "[DEBUG]    <Output> dynamic : " << output.get_partial_shape() << "\n";
                                std::cout << "[DEBUG]    <Output> dynamic : " << output.get_partial_shape().rank() << "\n";
                                std::cout << "[DEBUG]    <Output> dynamic : " << output.get_partial_shape().rank().get_length() << "\n";
                                new_node->set_output_type(i, dst_type, ov::PartialShape::dynamic(output.get_partial_shape().rank()));
                            }
                        } else if (name.find(std::string("Static")) != std::string::npos) {
                            std::cout << "[DEBUG]    DynamicShapeResolver usage for StaticShape:\n";
                            auto shape = node->input_value(0).get_node_shared_ptr()->input(1).get_source_output();
                            auto dynamic = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(node->output(i),
                                           shape);
                            if (node->output(i).get_element_type() == src_type) {
                                auto convert = std::make_shared<ngraph::opset1::Convert>(dynamic, dst_type);
                                new_node->output(i).replace(convert);
                            } else {
                                new_node->output(i).replace(dynamic);
                            }
                        }
                        std::cout << "[DEBUG] <Output> Type after : " << new_node->outputs()[i].get_element_type() << "\n";
                    }
                }
                new_node = node->copy_with_new_inputs({new_in});
                new_node->set_friendly_name(node->get_friendly_name());
                ngraph::copy_runtime_info(node, new_node);
                ngraph::replace_node(node, {new_node});
            }
        } else if (name.find(std::string("Equal")) != std::string::npos) {
            for (int i = 1; i < node->input_values().size(); i++) {
                auto input = node->input_value(i);
                std::cout << "[DEBUG] <Input> int64 = " << node->get_type_info().name << "\n";
                auto convert = std::make_shared<ngraph::opset1::Convert>(input, node->get_input_element_type(0));
                new_in[i] = convert;
                std::cout << "[DEBUG] <Input> Type after : " << new_in[i].get_element_type() << "\n";
            }
            new_node = node->copy_with_new_inputs({new_in});
            new_node->set_friendly_name(node->get_friendly_name());
            ngraph::copy_runtime_info(node, new_node);
            ngraph::replace_node(node, {new_node});
        }
        return true;
    };
#endif
#ifdef MORE_ALL
    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        const auto& node = m.get_match_root();
        auto dst_type = ngraph::element::i32;
        auto src_type = ngraph::element::i64;
        auto new_in   = node->input_values();
        auto new_node = node;
        ov::NodeVector new_ops;
        std::string name = node->get_type_info().name;
        for (int i = 0; i < node->input_values().size(); i++) {
            auto input = node->input_value(i);
            if (input.get_element_type() == src_type) {
                std::cout << "[DEBUG] <Input> int64 = " << node->get_type_info().name << "\n";
                auto convert = std::make_shared<ngraph::opset1::Convert>(input, dst_type);
                new_in[i] = convert;
                std::cout << "[DEBUG] <Input> Type after : " << new_in[i].get_element_type() << "\n";
            }
        }
        for (int i = 0; i < new_node->outputs().size(); i++) {
            auto output = node->output(i);
            if (name.find(std::string("Shape")) == std::string::npos) {
                if (output.get_element_type() == src_type) {
                    std::cout << "[DEBUG] <Output> int64 : " << node->get_name() << ", " << node->get_type_info().name << "\n";
                    if (output.get_partial_shape().rank().is_static()) {
                        std::cout << "[DEBUG]    <Output> static : " << output.get_partial_shape() << "\n";
                        new_node->set_output_type(i, dst_type, output.get_partial_shape());
                    } else {
                        std::cout << "[DEBUG]    <Output> dynamic : " << output.get_partial_shape() << "\n";
                        std::cout << "[DEBUG]    <Output> dynamic : " << output.get_partial_shape().rank() << "\n";
                        std::cout << "[DEBUG]    <Output> dynamic : " << output.get_partial_shape().rank().get_length() << "\n";
                        new_node->set_output_type(i, dst_type, ov::PartialShape::dynamic(output.get_partial_shape().rank()));
                    }
                }
            } else if (name.find(std::string("Shape")) != std::string::npos) {
                if (output.get_element_type() == src_type || node->outputs()[i].get_partial_shape().is_dynamic()) {
                    std::cout << "[DEBUG]    DynamicShapeResolver usage for StaticShape:\n";
                    ngraph::Shape inputShape{node->outputs()[i].get_shape()};
                    for (int j = 0; j < node->inputs().size(); j++) {
                        if (node->input_value(j).get_shape().size() > 0) {
                            inputShape = node->input_value(j).get_shape();
                            break;
                        }
                    }
                    const auto data = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::i32, inputShape);
                    const auto dataShape = ngraph::opset5::Constant::create(node->outputs()[i].get_element_type(),
                                           ngraph::Shape{inputShape.size()}, inputShape);
                    auto dynamic = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(node->output(i), dataShape);
                    if (node->output(i).get_element_type() == src_type) {
                        auto convert = std::make_shared<ngraph::opset1::Convert>(dynamic, dst_type);
                        new_node->output(i).replace(convert);
                    } else {
                        new_node->output(i).replace(dynamic);
                    }
                }
                std::cout << "[DEBUG] <Output> Type after : " << new_node->outputs()[i].get_element_type() << "\n";
            }
        }
        new_node = node->copy_with_new_inputs({new_in});
        new_node->set_friendly_name(node->get_friendly_name());
        ngraph::copy_runtime_info(node, new_node);
        ngraph::replace_node(node, new_node);
        return true;
    };
#endif
#ifdef MORE_NOT_ALL
    auto stShTopK_pattern = ngraph::pattern::wrap_type<ngraph::vpu::op::StaticShapeTopK>();
    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        auto stShTopK_pattern_node = std::dynamic_pointer_cast<ngraph::vpu::op::StaticShapeTopK>(m.get_match_root());
        if (!stShTopK_pattern_node)
            return false;
        auto dst_type = ngraph::element::i32;
        auto src_type = ngraph::element::i64;
        auto new_in   = stShTopK_pattern_node->input_values();
        auto new_node = stShTopK_pattern_node;
        ov::NodeVector new_ops;
        std::string name = stShTopK_pattern_node->get_type_info().name;

        for (int i = 0; i < stShTopK_pattern_node->input_values().size(); i++) {
            auto input = stShTopK_pattern_node->input_value(i);
            if (input.get_element_type() == src_type) {
                std::cout << "[DEBUG] <Input> int64 = " << stShTopK_pattern_node->get_type_info().name << "\n";
                auto convert = std::make_shared<ngraph::opset1::Convert>(input, dst_type);
                new_in[i] = convert;
                std::cout << "[DEBUG] <Input> Type after : " << new_in[i].get_element_type() << "\n";
            }
        }
        for (int i = 0; i < new_node->outputs().size(); i++) {
            auto output = stShTopK_pattern_node->output(i);
            if (output.get_element_type() == src_type) {
                std::cout << "[DEBUG] <Output> int64 : " << stShTopK_pattern_node->get_name() << ", " << stShTopK_pattern_node->get_type_info().name << "\n";
                if (output.get_partial_shape().rank().is_static()) {
                    std::cout << "[DEBUG]    <Output> static : " << output.get_partial_shape() << "\n";
                    new_node->set_output_type(i, dst_type, output.get_partial_shape());
                } else {
                    std::cout << "[DEBUG]    <Output> dynamic : " << output.get_partial_shape() << "\n";
                    std::cout << "[DEBUG]    <Output> dynamic : " << output.get_partial_shape().rank() << "\n";
                    std::cout << "[DEBUG]    <Output> dynamic : " << output.get_partial_shape().rank().get_length() << "\n";
                    new_node->set_output_type(i, dst_type, ov::PartialShape::dynamic(output.get_partial_shape().rank()));
                }
                std::cout << "[DEBUG] <Output> Type after : " << new_node->outputs()[i].get_element_type() << "\n";
            }
        }
        auto new_topK = stShTopK_pattern_node->clone_with_new_inputs({new_in});
        new_topK->set_friendly_name(stShTopK_pattern_node->get_friendly_name());
        ngraph::copy_runtime_info(stShTopK_pattern_node, new_topK);
        ngraph::replace_node(stShTopK_pattern_node, {new_topK});
        return true;
    };
#endif

    auto m = std::make_shared<ngraph::pattern::Matcher>(any_node, "ConvertI64Data");
    register_matcher(m, callback);
}

}  // namespace vpu
