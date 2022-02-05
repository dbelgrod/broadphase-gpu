cmake_minimum_required(VERSION 3.14.0 FATAL_ERROR)
# set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
project(GPUBF LANGUAGES CXX CUDA)

SET(CUDA_SEPARABLE_COMPILATION ON)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(GPUBF_TOPLEVEL_PROJECT ON)
else()
  set(GPUBF_TOPLEVEL_PROJECT OFF)
endif()

################################################################################
# Configuration
################################################################################

# set(GPUBF_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
# set(GPUBF_EXTERNAL "${CMAKE_CURRENT_SOURCE_DIR}/external")
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/")
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/recipes/")
include(Warnings)
include(UseColors)
include(GPUBF_Utils)

################################################################################
# GPU Library
################################################################################

add_subdirectory(src)

prepend_current_path(GPUBF_SOURCES)
gpubf_copy_headers(${GPUBF_SOURCES})
GPUBF_set_source_group(${GPUBF_SOURCES})

add_library(GPUBF STATIC ${GPUBF_SOURCES} )
add_library(broadphase ALIAS GPUBF )

target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_BINARY_DIR}/include)
# target_link_libraries(GPUBF PRIVATE warnings::all)
target_compile_definitions(${PROJECT_NAME} PUBLIC NOMINMAX)
# target_compile_definitions(${PROJECT_NAME} PUBLIC CCD_USE_DOUBLE)

# Request that GPUBF be built with -std=c++11
# As this is a public compile feature anything that links to GPUBF
# will also build with -std=c++11
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_11)

# include directory
# target_include_directories(${PROJECT_NAME} SYSTEM INTERFACE
#   $<BUILD_INTERFACE:${GPUBF_SOURCE_DIR}>
#   $<INSTALL_INTERFACE:include>
# )

#-------------------------------------------------------------------------------
# CUDA
#-------------------------------------------------------------------------------
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas -O3,-v -use_fast_math -maxrregcount 24")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -use_fast_math --expt-relaxed-constexpr  -gencode arch=compute_86,code=sm_86")# -DCCD_USE_DOUBLE")

# We need to explicitly state that we need all CUDA files in the GPUBF
# library to be built with -dc as the member functions could be called by
# other libraries and executables

set_target_properties( ${PROJECT_NAME}
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                      )
                      
# Nvidia RTX8000 -> compute_75
# Nvidia V100 -> compute_70
# Nvidia 1080/1080Ti -> compute_61       
# Nvidia 3080Ti -> compute_86               
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 70 75 86)

# if(APPLE)
# # We need to add the default path to the driver (libcuda.dylib) as an rpath,
# # so that the static cuda runtime can find it at runtime.
# set_property(TARGET GPUBF_bin PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
# endif()

target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --generate-line-info
                       --use_fast_math
                       --relocatable-device-code=true
                       # --ptxas-options=-v
                       #--maxrregcount=40
                       >)

#-------------------------------------------------------------------------------
# Required Libraries
#-------------------------------------------------------------------------------

# libigl
include(eigen)
include(libigl)
target_link_libraries(${PROJECT_NAME} PUBLIC
  Eigen3::Eigen
  igl::core
)

# tbb (for multiGPU)
include(onetbb)
target_link_libraries(${PROJECT_NAME} PUBLIC TBB::tbb)

# JSON
include(json)
target_link_libraries(${PROJECT_NAME} PUBLIC nlohmann::json)

# Logger
include(spdlog)
target_link_libraries(${PROJECT_NAME} PUBLIC spdlog::spdlog)

#-------------------------------------------------------------------------------
# Binary
#-------------------------------------------------------------------------------

if(GPUBF_TOPLEVEL_PROJECT)
  # add_executable(GPUBF_bin app/main.cu)
  add_executable(GPUBF_bin app/main.cu)
  set_target_properties(GPUBF_bin PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  set_property(TARGET GPUBF_bin PROPERTY CUDA_ARCHITECTURES 70 75 86)
  target_link_libraries(GPUBF_bin PRIVATE ${PROJECT_NAME})

  target_compile_options(GPUBF_bin PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                        --generate-line-info
                        --use_fast_math
                        --relocatable-device-code=true
                        --ptxas-options=-v
                        # --maxrregcount=7
                        >)  
endif()

################################################################################
# C++ code
################################################################################
prepend_current_path(CPUSWEEP_SOURCES)

message(DEBUG "include dir: ${CPUSWEEP_SOURCES}")

gpubf_copy_headers_cpu(${CPUSWEEP_SOURCES})
gpubf_set_source_group(${CPUSWEEP_SOURCES})
add_library(CPUSWEEP STATIC ${CPUSWEEP_SOURCES} )
target_include_directories(CPUSWEEP PUBLIC ${PROJECT_BINARY_DIR}/include/cpu)

# target_include_directories(CPUSWEEP PUBLIC ${PROJECT_BINARY_DIR}/include)
target_link_libraries(CPUSWEEP PRIVATE warnings::all)
target_compile_definitions(CPUSWEEP PUBLIC NOMINMAX) # CCD_USE_DOUBLE)

# Request that GPUBF be built with -std=c++11
# As this is a public compile feature anything that links to GPUBF
# will also build with -std=c++11
target_compile_features(CPUSWEEP PUBLIC cxx_std_17)

#-------------------------------------------------------------------------------
# Required Libraries
#-------------------------------------------------------------------------------

# libigl
include(eigen)
include(libigl)
target_link_libraries(CPUSWEEP PUBLIC
  Eigen3::Eigen
  igl::core
)

# tbb (for multiGPU)
include(onetbb)
target_link_libraries(CPUSWEEP PUBLIC TBB::tbb)

# JSON
include(json)
target_link_libraries(CPUSWEEP PUBLIC nlohmann::json)

# Logger
include(spdlog)
target_link_libraries(CPUSWEEP PUBLIC spdlog::spdlog)

#-------------------------------------------------------------------------------
# Binary
#-------------------------------------------------------------------------------

if(GPUBF_TOPLEVEL_PROJECT)
  add_executable(cpusweep_bin app/main.cpp)
  target_link_libraries(cpusweep_bin PUBLIC CPUSWEEP)
endif()
