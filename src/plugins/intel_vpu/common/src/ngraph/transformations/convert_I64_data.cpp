// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/ngraph/transformations/convert_I64_data.hpp"
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <ngraph/opsets/opset6.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

NGRAPH_RTTI_DEFINITION(vpu::ConvertI64Data, "ConvertI64Data", 0);

namespace vpu {

ConvertI64Data::ConvertI64Data() {
    auto any_node = ngraph::pattern::any_input();

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        const auto& node = m.get_match_root();

        auto dst_type = ngraph::element::Type_t::i32;
        auto src_type = ngraph::element::Type_t::i64;
        auto new_in = node->input_values();
        auto new_node = node;

        if (!ngraph::op::is_output(node)) {
            for (int i = 0; i < node->input_values().size(); i++) {
                auto input = node->input_values()[i];
                if (input.get_element_type() == src_type) {
                    printf(" --- Input int64 = %s\n", node->get_type_info().name);
                    auto convert = std::make_shared<ngraph::opset8::Convert>(input, dst_type);
                    new_in[i] = convert->outputs()[0];
                }
            }
            for (int i = 0; i < node->outputs().size(); i++) {
                auto output = node->output(i);
                if (output.get_element_type() == src_type) {
                    printf(" --- Output int64 = %s\n", node->get_type_info().name);
                    if (output.get_partial_shape().rank().is_dynamic()) {
                        new_node->set_output_type(i, dst_type, ov::PartialShape::dynamic());
                    } else {
                        auto shape = output.get_partial_shape();
                        new_node->set_output_type(i, dst_type, shape);
                    }
                }
            }
            new_node->copy_with_new_inputs({new_in});
            new_node->set_friendly_name(node->get_friendly_name());
            ngraph::copy_runtime_info(node, new_node);
            ngraph::replace_node(node, new_node);
        }

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(any_node, "ConvertI64Data");
    register_matcher(m, callback);
}

}  // namespace vpu
