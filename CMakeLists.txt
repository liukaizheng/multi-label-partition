cmake_minimum_required(VERSION 3.24)

project(multi-label-partition VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(cmake/CPM.cmake)

find_package(Eigen3 REQUIRED)
set(CGAL_DO_NOT_WARN_ABOUT_CMAKE_BUILD_TYPE ON)
find_package(CGAL REQUIRED)

CPMAddPackage("gh:CLIUtils/CLI11#v2.6.1")
# CPMAddPackage("gh:liukaizheng/mesh#main")
CPMAddPackage(
    NAME mesh
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../mesh
)
CPMAddPackage(
    NAME libigl
    GIT_TAG main
    GITHUB_REPOSITORY liukaizheng/libigl
)

# CPMAddPackage(
#     NAME predicates
#     SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../predicates
# )
CPMAddPackage("gh:liukaizheng/predicates#main")

CPMAddPackage("gh:qnzhou/MshIO#main")

add_executable(multi-label-partition
    src/main.cpp
    src/material_interface.cpp
)

target_compile_features(multi-label-partition PRIVATE cxx_std_23)
target_link_libraries(multi-label-partition PRIVATE
    CGAL::CGAL
    CLI11::CLI11
    Eigen3::Eigen
    gpf::mesh
    igl::core
    predicates
    mshio::mshio
)
target_include_directories(multi-label-partition PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_options(multi-label-partition PRIVATE
    -fsanitize=address -fno-omit-frame-pointer
)

target_link_options(multi-label-partition PRIVATE
    -fsanitize=address
)
