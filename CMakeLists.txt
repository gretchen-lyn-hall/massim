cmake_minimum_required(VERSION 3.15)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

#add_compile_options("$<$<CONFIG:DEBUG>:-fsanitize=address>")
#add_link_options("$<$<CONFIG:DEBUG>:-fsanitize=address>")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -v -stdlib=libstdc++")

find_package(OpenMP REQUIRED)

file(GLOB SOURCES "src/cpp/*.cpp")
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
add_library(cmassim STATIC ${SOURCES})
target_include_directories(cmassim PRIVATE OpenMP::OpenMP_CXX_INCLUDE_DIRS)
set_property(TARGET cmassim PROPERTY CXX_STANDARD 23)
target_compile_options(cmassim PRIVATE -fexperimental-library)
target_link_libraries (cmassim Eigen3::Eigen -fexperimental-library OpenMP::OpenMP_CXX)

# Install the dyl
install(
  TARGETS cmassim
  EXPORT cmassimTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})



find_package (Eigen3 3.3 REQUIRED)
include_directories( ${EIGEN3_INCLUDE_DIRS} )
pybind11_add_module(_core MODULE src/main.cpp)
target_include_directories(_core PRIVATE OpenMP::OpenMP_CXX_INCLUDE_DIRS)
set_property(TARGET _core PROPERTY CXX_STANDARD 23)
target_link_libraries(_core PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(_core PRIVATE cmassim)

install(TARGETS _core cmassim DESTINATION ${SKBUILD_PROJECT_NAME})

add_executable(prof_test "src/cpp/tests/profile_prof.cpp")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-instr-generate -fcoverage-mapping")
set_property(TARGET prof_test PROPERTY CXX_STANDARD 23)
target_compile_options(prof_test PRIVATE -fexperimental-library)
target_link_libraries (prof_test cmassim Eigen3::Eigen -fexperimental-library OpenMP::OpenMP_CXX)

add_executable(smallmap_test "src/cpp/tests/smallmap_test.cpp")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set_property(TARGET smallmap_test PROPERTY CXX_STANDARD 23)
target_compile_options(smallmap_test PRIVATE -fexperimental-library)
target_link_libraries (smallmap_test cmassim Eigen3::Eigen -fexperimental-library OpenMP::OpenMP_CXX)


# find_package(cmassim CONFIG)

