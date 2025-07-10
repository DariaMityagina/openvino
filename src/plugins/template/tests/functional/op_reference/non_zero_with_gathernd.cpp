// // Copyright (C) 2024 Intel Corporation
// // SPDX-License-Identifier: Apache-2.0
// //

// #include <memory>
// #include <openvino/op/gather_nd.hpp>
// #include <openvino/op/not_equal.hpp>
// #include <shared_test_classes/base/ov_subgraph.hpp>

// #include <common_test_utils/ov_tensor_utils.hpp>
// #include <openvino/opsets/opset3.hpp>

// using namespace ov::test;
// namespace {

// using NonZeroWithGatherNDTestParams = std::tuple<ov::test::InputShape, ov::element::Type>;

// class NonZeroWithGatherNDReproducerTest :
//         public testing::WithParamInterface<NonZeroWithGatherNDTestParams> {
// public:
//     void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
//         inputs.clear();
//         const auto& funcInputs = function->inputs();

//         const int32_t startFrom = -1;
//         const int32_t range = 5;

//         for (size_t i = 0; i < funcInputs.size(); ++i) {
//             const auto& funcInput = funcInputs[i];
//             ov::Tensor tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(),
//                                                                         targetInputStaticShapes[i], range, startFrom);
//             inputs.insert({funcInput.get_node_shared_ptr(), tensor});
//         }
//     }

// protected:
//     void SetUp() override {
//         const auto& [inputShape, type] = this->GetParam();

//         init_input_shapes({inputShape});
//         ov::ParameterVector inputParams;
//         for (auto&& shape : inputDynamicShapes) {
//             inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(type, shape));
//         }

//         auto constInput = ov::op::v0::Constant::create(type, {1, 1}, {-1});
//         auto notEqual = std::make_shared<ov::op::v1::NotEqual>(inputParams[0], constInput);

//         auto nonZero = std::make_shared<ov::op::v3::NonZero>(notEqual, ov::element::i64);
//         inputParams[0]->set_friendly_name("input");

//         const auto dimsOrder = {1, 0};
//         auto order = ov::op::v0::Constant::create(ov::element::i64, {dimsOrder.size()}, dimsOrder);
//         auto transpose = std::make_shared<ov::op::v1::Transpose>(nonZero, order);
//         auto gatherND = std::make_shared<ov::op::v8::GatherND>(inputParams[0], transpose);
//         auto convert = std::make_shared<ov::op::v0::Convert>(gatherND, type);

//         auto results = ov::ResultVector();
//         for (size_t i = 0; i < convert->get_output_size(); i++) {
//             results.push_back(std::make_shared<ov::opset3::Result>(convert->output(i)));
//         }

//         function = std::make_shared<ov::Model>(results, inputParams, "NonZeroWithGatherND");
//     }
// };

// const std::vector<ov::element::Type> inputPrecision = {ov::element::i32};

// const std::vector<ov::test::InputShape> inShapes = {staticShape(1, 88), staticShape(8, 32)};

// INSTANTIATE_TEST_SUITE_P(
//     smoke_NonZeroWithGatherND,
//     NonZeroWithGatherNDReproducerTest,
//     ::testing::Combine(::testing::Combine(::testing::ValuesIn(inShapes)
//                        ::testing::ValuesIn(inputPrecision)
//                        ::testing::Values(ov::test::utils::DEVICE_TEMPLATE)));

// }  // namespace
