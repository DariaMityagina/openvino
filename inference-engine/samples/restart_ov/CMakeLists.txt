# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

ie_add_sample(NAME restart_ov
              SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
              HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/sample.h"
              DEPENDENCIES format_reader ie_samples_utils
              OPENCV_DEPENDENCIES core)

find_package(ngraph REQUIRED)
target_link_libraries(restart_ov PRIVATE ${NGRAPH_LIBRARIES})
