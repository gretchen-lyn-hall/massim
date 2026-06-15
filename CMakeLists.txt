cmake_minimum_required(VERSION 4.0)

if (CMAKE_HOST_APPLE)  
 #set(COMMON_COMPILE_OPTIONS -arch arm64)
 #set(COMMON_LINK_OPTIONS -arch arm64)
 set(CMAKE_OSX_ARCHITECTURES arm64)
 set(CMAKE_OSX_DEPLOYMENT_TARGET "15")
endif()

# For debugging builds
set(CMAKE_VERBOSE_MAKEFILE ON)

if ( ${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR} )
message( FATAL_ERROR “In-source builds not allowed! Create a build directory and run CMake from there. ” )
endif()

set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "451f2fe2-a8a2-47c3-bc32-94786d8fc91b")
set(CXX_MODULE_STD 1)

project(massim LANGUAGES CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)


# For debugging Github Actions compilation
function(print_all_variables)
  message(STATUS "CMake Variables:")
  get_cmake_property(_variableNames VARIABLES)
  list(SORT _variableNames)
  foreach (_variableName ${_variableNames})
    message(STATUS "${_variableName}=${${_variableName}}")
  endforeach()
endfunction()






#find_package(OpenMP REQUIRED)

## Building the C++ Library - 'cmassim'
file(GLOB SOURCES "src/cpp/*.cpp")
find_package (Eigen3 5 REQUIRED NO_MODULE)
add_library(cmassim STATIC ${SOURCES})
target_include_directories(cmassim PRIVATE)
set_property(TARGET cmassim PROPERTY CXX_STANDARD 23)
target_compile_options(cmassim PUBLIC ${COMMON_COMPILE_OPTIONS})
target_link_libraries (cmassim Eigen3::Eigen ${COMMON_LINK_OPTIONS}) 

print_all_variables()
# Install the dyl
install(
  TARGETS cmassim
  EXPORT cmassimTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})


## Building the Python bindings: 'massim._core'
find_package (Eigen3 5 REQUIRED)
include_directories( ${EIGEN3_INCLUDE_DIRS} )
pybind11_add_module(_core MODULE src/main.cpp)
target_include_directories(_core PRIVATE)
set_property(TARGET _core PROPERTY CXX_STANDARD 23)
target_compile_options(_core PRIVATE ${COMMON_COMPILE_OPTIONS})
target_link_libraries(_core PRIVATE cmassim ${COMMON_LINK_OPTIONS})
install(TARGETS _core cmassim DESTINATION massim)



# Extra test apps
add_executable(prof_test "src/cpp/tests/profile_prof.cpp")
set_property(TARGET prof_test PROPERTY CXX_STANDARD 23)
target_compile_options(prof_test PRIVATE ${COMMON_COMPILE_OPTIONS})
target_compile_features(prof_test PRIVATE cxx_std_23 INTERFACE cxx_std_23)
target_link_libraries (prof_test cmassim Eigen3::Eigen ${COMMON_LINK_OPTIONS})

add_executable(smallmap_test "src/cpp/tests/smallmap_test.cpp")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set_property(TARGET smallmap_test PROPERTY CXX_STANDARD 23)
target_compile_options(smallmap_test PRIVATE ${COMMON_COMPILE_OPTIONS})
target_compile_features(smallmap_test PRIVATE cxx_std_23 INTERFACE cxx_std_23)
target_link_libraries (smallmap_test cmassim Eigen3::Eigen ${COMMON_LINK_OPTIONS})


# find_package(cmassim CONFIG)

