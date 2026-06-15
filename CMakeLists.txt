cmake_minimum_required(VERSION 4.0)
set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "451f2fe2-a8a2-47c3-bc32-94786d8fc91b")
set(CXX_MODULE_STD 1)

project(massim LANGUAGES CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)


#find_package(OpenMP REQUIRED)

file(GLOB SOURCES "src/cpp/*.cpp")
find_package (Eigen3 5 REQUIRED NO_MODULE)
add_library(cmassim STATIC ${SOURCES})
target_include_directories(cmassim PRIVATE)
set_property(TARGET cmassim PROPERTY CXX_STANDARD 23)

if (CMAKE_HOST_APPLE)  
  target_compile_options(cmassim PRIVATE -mmacosx-version-min=13.3)
endif()

target_link_libraries (cmassim Eigen3::Eigen) 

# Install the dyl
install(
  TARGETS cmassim
  EXPORT cmassimTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})



find_package (Eigen3 5 REQUIRED)
include_directories( ${EIGEN3_INCLUDE_DIRS} )
pybind11_add_module(_core MODULE src/main.cpp)
target_include_directories(_core PRIVATE)
set_property(TARGET _core PROPERTY CXX_STANDARD 23)
if (CMAKE_HOST_APPLE)  
  target_compile_options(cmassim PRIVATE -mmacosx-version-min=13.3)
endif()
target_link_libraries(_core PRIVATE cmassim)
install(TARGETS _core cmassim DESTINATION massim)



# Extra test apps
add_executable(prof_test "src/cpp/tests/profile_prof.cpp")
set_property(TARGET prof_test PROPERTY CXX_STANDARD 23)
target_compile_options(prof_test PRIVATE)
target_compile_features(prof_test PRIVATE cxx_std_23 INTERFACE cxx_std_23)
target_link_libraries (prof_test cmassim Eigen3::Eigen)

add_executable(smallmap_test "src/cpp/tests/smallmap_test.cpp")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set_property(TARGET smallmap_test PROPERTY CXX_STANDARD 23)
target_compile_options(smallmap_test PRIVATE)
target_compile_features(smallmap_test PRIVATE cxx_std_23 INTERFACE cxx_std_23)
target_link_libraries (smallmap_test cmassim Eigen3::Eigen)


# find_package(cmassim CONFIG)

