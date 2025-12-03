cmake_minimum_required(VERSION 3.1)
project(ObsAvoid LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED)
find_package(pybind11 REQUIRED)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

add_subdirectory(zed-open-capture)
pybind11_add_module(depth_wrapper src/depth_wrapper.cpp)
add_executable(depth_blur src/zed_oc_depth_blur.cpp)

target_link_libraries(depth_wrapper
    PRIVATE zed_open_capture ${OpenCV_LIBS}
)

target_include_directories(depth_wrapper
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/zed-open-capture/include
    ${CMAKE_CURRENT_SOURCE_DIR}/zed-open-capture/examples/include
    ${OpenCV_INCLUDE_DIRS}
)

target_compile_definitions(depth_wrapper PRIVATE
    VIDEO_MOD_AVAILABLE
)

target_link_libraries(depth_blur
    PRIVATE zed_open_capture ${OpenCV_LIBS}
)

target_include_directories(depth_blur
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/zed-open-capture/include
    ${CMAKE_CURRENT_SOURCE_DIR}/zed-open-capture/examples/include
    ${OpenCV_INCLUDE_DIRS}
)

target_compile_definitions(depth_blur PRIVATE
    VIDEO_MOD_AVAILABLE
)

remove_definitions(-DSENSORS_MOD_AVAILABLE)