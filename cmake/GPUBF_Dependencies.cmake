# Prepare dependencies
#
# For each third-party library, if the appropriate target doesn't exist yet,
# download it via external project, and add_subdirectory to build it alongside
# this project.


# Download and update 3rd_party libraries
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR})
list(REMOVE_DUPLICATES CMAKE_MODULE_PATH)
include(GPUBF_DownloadExternal)

# libigl
if(NOT TARGET igl::core)
    gpubf_download_libigl()
    add_subdirectory(${GPUBF_EXTERNAL}/libigl EXCLUDE_FROM_ALL)
endif()

# JSON
if(NOT TARGET nlohmann_json::nlohmann_json)
  gpubf_download_nlohmann_json()
  add_subdirectory(${GPUBF_EXTERNAL}/nlohmann_json EXCLUDE_FROM_ALL)
endif()

# if(TIGHT_INCLUSION_WITH_GMP OR TIGHT_INCLUSION_WITH_TESTS)
#   #GMP
#   find_package(GMPECCD)
#   IF(NOT ${GMP_FOUND})
#           MESSAGE(FATAL_ERROR "Cannot find GMP")
#   ENDIF()
# endif()


# if(NOT TARGET Eigen3::Eigen)
#   gpu_download_eigen()
#   add_library(tccd_eigen INTERFACE)
#   target_include_directories(tccd_eigen SYSTEM INTERFACE
#     $<BUILD_INTERFACE:${TIGHT_INCLUSION_EXTERNAL}/eigen>
#     $<INSTALL_INTERFACE:include>
#   )
#   set_property(TARGET tccd_eigen PROPERTY EXPORT_NAME Eigen3::Eigen)
#   add_library(Eigen3::Eigen ALIAS tccd_eigen)
# endif()

# TBB
if(NOT TARGET tbb::tbb)
  gpubf_download_tbb()
	set(TBB_BUILD_STATIC ON CACHE BOOL " " FORCE)
	set(TBB_BUILD_SHARED OFF CACHE BOOL " " FORCE)
  set(TBB_BUILD_STATIC ON CACHE BOOL " " FORCE)
	set(TBB_BUILD_SHARED OFF CACHE BOOL " " FORCE)
	set(TBB_BUILD_TBBMALLOC OFF CACHE BOOL " " FORCE)
	set(TBB_BUILD_TBBMALLOC_PROXY OFF CACHE BOOL " " FORCE)
	set(TBB_BUILD_TESTS OFF CACHE BOOL " " FORCE)
	set(TBB_NO_DATE ON CACHE BOOL " " FORCE)

	add_subdirectory(${GPUBF_EXTERNAL}/tbb tbb)
	set_target_properties(tbb_static PROPERTIES
		INTERFACE_INCLUDE_DIRECTORIES "${GPUBF_EXTERNAL}/tbb/include"
	)
	if(NOT MSVC)
		set_target_properties(tbb_static PROPERTIES
			COMPILE_FLAGS "-Wno-implicit-fallthrough -Wno-missing-field-initializers -Wno-unused-parameter -Wno-keyword-macro"
		)
		set_target_properties(tbb_static PROPERTIES POSITION_INDEPENDENT_CODE ON)
	endif()
	add_library(tbb::tbb ALIAS tbb_static)
endif()