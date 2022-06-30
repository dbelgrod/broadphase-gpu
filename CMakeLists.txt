cmake_minimum_required(VERSION 3.14.0 FATAL_ERROR)
# set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
project(SweepAndTiniestQueue LANGUAGES CXX)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(STQ_TOPLEVEL_PROJECT ON)
else()
  set(STQ_TOPLEVEL_PROJECT OFF)
endif()

################################################################################
# Configuration
################################################################################

option(STQ_WITH_CPU  "Enable CPU Implementation"  ON)
option(STQ_WITH_CUDA "Enable CUDA Implementation" ON)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/stq")
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/recipes/")

include(stq_utils)

add_subdirectory(src)

################################################################################
# C++ code
################################################################################
if(STQ_WITH_CPU)
  prepend_current_path(STQ_CPU_SOURCES)
  stq_copy_headers_cpu(${STQ_CPU_SOURCES})
  stq_set_source_group(${STQ_CPU_SOURCES})

  add_library(STQ_CPU STATIC ${STQ_CPU_SOURCES})
  add_library(STQ::CPU ALIAS STQ_CPU)

  target_include_directories(STQ_CPU PUBLIC ${PROJECT_BINARY_DIR}/include)

  include(stq_warnings)
  target_link_libraries(STQ_CPU PRIVATE STQ::warnings)
  target_compile_definitions(STQ_CPU PUBLIC NOMINMAX) # CCD_USE_DOUBLE)

  target_compile_features(STQ_CPU PUBLIC cxx_std_17)

  #-------------------------------------------------------------------------------
  # Required Libraries
  #-------------------------------------------------------------------------------

  # libigl
  include(eigen)
  include(libigl)
  target_link_libraries(STQ_CPU PUBLIC Eigen3::Eigen igl::core)

  # tbb (for multiGPU)
  include(onetbb)
  target_link_libraries(STQ_CPU PUBLIC TBB::tbb)

  # JSON
  include(json)
  target_link_libraries(STQ_CPU PUBLIC nlohmann_json::nlohmann_json)

  # Logger
  include(spdlog)
  target_link_libraries(STQ_CPU PUBLIC spdlog::spdlog)

  #-------------------------------------------------------------------------------
  # Binary
  #-------------------------------------------------------------------------------

  if(STQ_TOPLEVEL_PROJECT)
    add_executable(cpusweep_bin app/main.cpp)
    target_link_libraries(cpusweep_bin PUBLIC STQ::CPU)
  endif()
endif()

################################################################################
# GPU Library
################################################################################
if(STQ_WITH_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
  else()
    message(FATAL_ERROR "No CUDA support found!")
  endif()

  prepend_current_path(STQ_CUDA_SOURCES)
  stq_copy_headers(${STQ_CUDA_SOURCES})
  stq_set_source_group(${STQ_CUDA_SOURCES})

  add_library(STQ_CUDA STATIC ${STQ_CUDA_SOURCES} )
  add_library(STQ::CUDA ALIAS STQ_CUDA)

  target_include_directories(STQ_CUDA PUBLIC ${PROJECT_BINARY_DIR}/include)
  target_compile_definitions(STQ_CUDA PUBLIC NOMINMAX)
  # target_compile_definitions(STQ_CUDA PUBLIC CCD_USE_DOUBLE)

  # Request that STQ_CUDA be built with -std=c++11
  # As this is a public compile feature anything that links to STQ_CUDA
  # will also build with -std=c++11
  target_compile_features(STQ_CUDA PUBLIC cxx_std_11)

  #-------------------------------------------------------------------------------
  # CUDA
  #-------------------------------------------------------------------------------
  # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g")
  # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")
  # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas -O3,-v -use_fast_math -maxrregcount 24")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -use_fast_math --expt-relaxed-constexpr  -gencode arch=compute_86,code=sm_86")# -DCCD_USE_DOUBLE")

  # We need to explicitly state that we need all CUDA files in the STQ_CUDA
  # library to be built with -dc as the member functions could be called by
  # other libraries and executables
  set_target_properties(STQ_CUDA PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

  # Nvidia RTX8000 -> compute_75
  # Nvidia V100 -> compute_70
  # Nvidia 1080/1080Ti -> compute_61
  # Nvidia 3080Ti -> compute_86
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 75 86)
  endif()
  set_target_properties(STQ_CUDA PROPERTIES CUDA_ARCHITECTURES "70;75;86")

  # if(APPLE)
  # # We need to add the default path to the driver (libcuda.dylib) as an rpath,
  # # so that the static cuda runtime can find it at runtime.
  # set_property(TARGET GPUBF_bin PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
  # endif()

  target_compile_options(STQ_CUDA PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                        --generate-line-info
                        --use_fast_math
                        --relocatable-device-code=true
                        # --ptxas-options=-v
                        #--maxrregcount=40
                        >)

  #-------------------------------------------------------------------------------
  # Required Libraries
  #-------------------------------------------------------------------------------

  # libigl include(eigen)
  include(libigl)
  target_link_libraries(STQ_CUDA PUBLIC Eigen3::Eigen igl::core)

  # tbb (for multiGPU)
  include(onetbb)
  target_link_libraries(STQ_CUDA PUBLIC TBB::tbb)

  # JSON
  include(json)
  target_link_libraries(STQ_CUDA PUBLIC nlohmann_json::nlohmann_json)

  # Logger
  include(spdlog)
  target_link_libraries(STQ_CUDA PUBLIC spdlog::spdlog)

  #-------------------------------------------------------------------------------
  # Binary
  #-------------------------------------------------------------------------------

  if(STQ_TOPLEVEL_PROJECT)
    # add_executable(GPUBF_bin app/main.cu)
    add_executable(GPUBF_bin app/main.cu)
    set_target_properties(GPUBF_bin PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_property(TARGET GPUBF_bin PROPERTY CUDA_ARCHITECTURES 70 75 86)
    target_link_libraries(GPUBF_bin PRIVATE STQ::CUDA)

    target_compile_options(GPUBF_bin PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                          --generate-line-info
                          --use_fast_math
                          --relocatable-device-code=true
                          --ptxas-options=-v
                          # --maxrregcount=7
                          >)
  endif()
endif()
