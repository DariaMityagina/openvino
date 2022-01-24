// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/ngraph/transformations/convert_I64_data.hpp"
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

NGRAPH_RTTI_DEFINITION(vpu::ConvertI64Data, "ConvertI64Data", 0);

namespace vpu {

ConvertI64Data::ConvertI64Data() {
    auto input_pattern = ngraph::pattern::any_input();
    auto const_pattern = ngraph::pattern::wrap_type<ngraph::opset5::Constant>();
    auto const_pattern = ngraph::pattern::wrap_type<ngraph::opset1::Constant>({input_pattern, const_pattern},
                                                               ngraph::pattern::consumers_count(1));

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        const auto& pattern_value_map = m.get_pattern_value_map();
        const auto& input = pattern_value_map.at(input_pattern);
        const auto& type = input.get_element_type();

        if (input.get_element_type() == ngraph::element::i64) {
            
        }
        
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(, "ConvertI64Data");
    register_matcher(m, callback);
}

}  // namespace vpu
