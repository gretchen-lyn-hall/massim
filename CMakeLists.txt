cmake_minimum_required(VERSION 3.15)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

#add_compile_options("$<$<CONFIG:DEBUG>:-fsanitize=address>")
#add_link_options("$<$<CONFIG:DEBUG>:-fsanitize=address>")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -v -stdlib=libstdc++")

file(GLOB SOURCES "src/cpp/*.cpp")
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
add_library(cmassim STATIC ${SOURCES})
set_property(TARGET cmassim PROPERTY CXX_STANDARD 20)
target_compile_options(cmassim PRIVATE -fexperimental-library)
target_link_libraries (cmassim Eigen3::Eigen -fexperimental-library)

# Install the dyl
install(
  TARGETS cmassim
  EXPORT cmassimTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})


find_package(OpenMP REQUIRED)

find_package (Eigen3 3.3 REQUIRED)
include_directories( ${EIGEN3_INCLUDE_DIRS} )
pybind11_add_module(_core MODULE src/main.cpp)
target_include_directories(_core PRIVATE OpenMP::OpenMP_CXX_INCLUDE_DIRS)
set_property(TARGET _core PROPERTY CXX_STANDARD 20)
target_link_libraries(_core PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(_core PRIVATE cmassim)

install(TARGETS _core cmassim DESTINATION ${SKBUILD_PROJECT_NAME})

# find_package(cmassim CONFIG)

