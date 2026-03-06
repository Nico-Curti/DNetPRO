cmake_minimum_required (VERSION 3.18)

# -------------------------------------------------------------------------
# Version / Description
# -------------------------------------------------------------------------
if (NOT DNetPRO_VERSION)
  set(DNetPRO_VERSION "1.0.0")
endif ()
if (NOT CMAKE_PROJECT_DESCRIPTION)
  set(CMAKE_PROJECT_DESCRIPTION "Discriminant Analysis with Network Processing")
endif ()
message (STATUS "DNetPRO VERSION: ${DNetPRO_VERSION}")

string(REPLACE "." ";" VERSION_LIST ${DNetPRO_VERSION})
if (NOT DNetPRO_MAJOR)
  list (GET VERSION_LIST 0 DNetPRO_MAJOR)
endif ()
if (NOT DNetPRO_MINOR)
  list (GET VERSION_LIST 1 DNetPRO_MINOR)
endif ()
if (NOT DNetPRO_REVISION)
  list (GET VERSION_LIST 2 DNetPRO_REVISION)
endif ()

# -------------------------------------------------------------------------
# VCPKG detection
# -------------------------------------------------------------------------
if(DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
  set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
  message(STATUS "VCPKG found: $ENV{VCPKG_ROOT}")
  message(STATUS "Using VCPKG integration")
  message(STATUS "VCPKG_MANIFEST_FEATURES: ${VCPKG_MANIFEST_FEATURES}")
  if(DEFINED ENV{VCPKG_DEFAULT_TRIPLET})
    message(STATUS "Setting default vcpkg target triplet to $ENV{VCPKG_DEFAULT_TRIPLET}")
    set(VCPKG_TARGET_TRIPLET $ENV{VCPKG_DEFAULT_TRIPLET})
  endif()
endif()

project (libDNetPRO
  LANGUAGES CXX
  VERSION ${DNetPRO_VERSION}
  DESCRIPTION ${CMAKE_PROJECT_DESCRIPTION}
)

# this is mandatory for scikit-build check
enable_language(C)
enable_language(CXX)

# required by parse_args library
if ( ( "${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU" AND ${CMAKE_CXX_COMPILER_VERSION} LESS 7) OR ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" AND ${CMAKE_CXX_COMPILER_VERSION} LESS_EQUAL 4) )
  set (CMAKE_CXX_STANDARD 14)
else ()
  set (CMAKE_CXX_STANDARD 17)
endif()
set (CMAKE_CXX_STANDARD_REQUIRED ON)

add_definitions (
  -DMAJOR=${DNetPRO_MAJOR}
  -DMINOR=${DNetPRO_MINOR}
  -DREVISION=${DNetPRO_REVISION}
)

# -------------------------------------------------------------------------
# ONLY allowed options
# -------------------------------------------------------------------------
option (OMP                   "Enable OpenMP support" OFF)
option (BUILD_DOCS            "Enable Documentation build support" OFF)
option (SKBUILD               "Build triggered by scikit-build/pip" OFF)
option (FORCE_BUILD_SUBMODULES "Force usage of modules/parse_args subdirectory" OFF)

# Backward compatibility (if old cache variable exists, map it)
if (DEFINED FORCE_USE_SUBMODULES AND FORCE_USE_SUBMODULES)
  set(FORCE_BUILD_SUBMODULES ON)
endif()

# -------------------------------------------------------------------------
# CMake modules path
# -------------------------------------------------------------------------
set(CMAKE_MODULE_PATH
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules/"
  ${CMAKE_MODULE_PATH}
)

include(GNUInstallDirs)

if ( NOT APPLE )
  set (CMAKE_SKIP_BUILD_RPATH             FALSE )
  set (CMAKE_BUILD_WITH_INSTALL_RPATH     FALSE )
  set (CMAKE_INSTALL_RPATH_USE_LINK_PATH  TRUE  )
endif ()

# make sure that the default is a RELEASE
set (default_build_type "Release")
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message (STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set (CMAKE_BUILD_TYPE "${default_build_type}" CACHE STRING "Choose the type of build." FORCE)
  set_property (CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif ()

# -------------------------------------------------------------------------
# Compile flags section
# -------------------------------------------------------------------------
if ( CMAKE_COMPILER_IS_GNUCXX )
  add_compile_options (-Wall -Wextra -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -Wno-narrowing -Wpedantic -Wno-dev)
  if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    string (REGEX REPLACE "-O[0123]" "-Og" CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG})
    set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG" )
    set (CMAKE_CXX_FLAGS_RELEASE "")
  elseif (CMAKE_BUILD_TYPE STREQUAL "Release")
    set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DNDEBUG" )
    set (CMAKE_CXX_FLAGS_DEBUG "")
  endif()
endif()

if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
  add_compile_options(-Wno-deprecated -Wno-writable-strings -Wno-dev)
  if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    string (REGEX REPLACE "-O0" "-Og" CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG})
    set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG" )
  elseif (CMAKE_BUILD_TYPE STREQUAL "Release")
    set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DNDEBUG" )
  endif()
endif()

if (MSVC)
  set (CMAKE_CXX_FLAGS "/wd4013 /wd4018 /wd4028 /wd4047 /wd4068 /wd4090 /wd4101 /wd4113 /wd4133 /wd4190 /wd4244 /wd4267 /wd4305 /wd4477 /wd4996 /wd4819 /fp:fast ${CMAKE_CXX_FLAGS}")
  string (REGEX REPLACE "/O2" "/Ox" CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
  add_definitions (-D_CRT_RAND_S)
  add_definitions (-DNOMINMAX)
  add_definitions (-D_USE_MATH_DEFINES)
  add_definitions (-D_CRT_SECURE_NO_DEPRECATE -D_SCL_SECURE_NO_WARNINGS)
  set (CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif ()

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if (COMPILER_SUPPORTS_MARCH_NATIVE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()

if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}" CACHE PATH "Install prefix" FORCE)
endif()

# -------------------------------------------------------------------------
# Parse options
# -------------------------------------------------------------------------
set(linked_libs "")

if (OMP)
  find_package(OpenMP REQUIRED)
  if (OPENMP_FOUND)
    message(STATUS "OpenMP found")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    if (APPLE)
      list(APPEND linked_libs OpenMP::OpenMP_CXX)
    endif()
  endif()
else()
  message(STATUS "OpenMP - disabled")
endif()

# -------------------------------------------------------------------------
# ParseArgs dependency (CLI only)
#   ParseArgs is required ONLY for the C++ executable (bin/DNetPRO_couples).
#   The Python/Cython build (SKBUILD=ON) must not pull this dependency.
# -------------------------------------------------------------------------
if (NOT SKBUILD)
  if (FORCE_BUILD_SUBMODULES)
    set(_DNETPRO_PREFIX_SAVED "${CMAKE_INSTALL_PREFIX}")

    message(STATUS "Use ParseArgs submodule (forced)")
    set(BUILD_TESTING OFF CACHE BOOL "Disable tests in subprojects" FORCE)

    # Keep subproject install dirs under our prefix
    set(INSTALL_BIN_DIR "${CMAKE_INSTALL_BINDIR}" CACHE PATH "Install dir for executables" FORCE)
    set(INSTALL_LIB_DIR "${CMAKE_INSTALL_LIBDIR}" CACHE PATH "Install dir for libraries" FORCE)
    set(INSTALL_INCLUDE_DIR "${CMAKE_INSTALL_INCLUDEDIR}/parseargs" CACHE PATH "Install dir for headers" FORCE)
    set(INSTALL_CMAKE_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/parseargs" CACHE PATH "Install dir for CMake package files" FORCE)

if (FORCE_USE_SUBMODULES)
  message (STATUS "Use ParseArgs submodule")
  add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/modules/parse_args")
  include_directories("${CMAKE_CURRENT_SOURCE_DIR}/modules/parse_args/include/")
  include_directories("${CMAKE_CURRENT_SOURCE_DIR}/modules/parse_args/hpp/")
  set (parseargslib parseargs)
else ()
  message (STATUS "Looking for ParseArgs library")
  find_package (ParseArgs HINTS ParseArgs_DIR)
  message (STATUS "----------------------------${ParseArgs_FOUND}")
  if (ParseArgs_FOUND)
    include_directories(${ParseArgs_INCLUDE_DIR})
    set (parseargslib ParseArgs::parseargs)
  else ()
    set (FORCE_USE_SUBMODULES TRUE)
    message (STATUS "ParseArgs NOT FOUND")
    message (STATUS "Use ParseArgs submodule")
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/modules/parse_args")
    include_directories("${CMAKE_CURRENT_SOURCE_DIR}/modules/parse_args/include/")
    include_directories("${CMAKE_CURRENT_SOURCE_DIR}/modules/parse_args/hpp/")
    set(parseargslib parseargs)

    # force output prefix
    set(CMAKE_INSTALL_PREFIX "${_DNETPRO_PREFIX_SAVED}" CACHE PATH "Install prefix" FORCE)
  else()
    set(_DNETPRO_PREFIX_SAVED "${CMAKE_INSTALL_PREFIX}")
    message(STATUS "Looking for ParseArgs library")
    find_package(ParseArgs CONFIG QUIET)

    if (ParseArgs_FOUND AND TARGET ParseArgs::parseargs)
      set(parseargslib ParseArgs::parseargs)
    else()
      message(STATUS "ParseArgs not usable (not found or missing target). Using submodule.")
      add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/modules/parse_args")
      include_directories("${CMAKE_CURRENT_SOURCE_DIR}/modules/parse_args/include/")
      include_directories("${CMAKE_CURRENT_SOURCE_DIR}/modules/parse_args/hpp/")
      set(parseargslib parseargs)
    endif()
    
    # force output prefix
    set(CMAKE_INSTALL_PREFIX "${_DNETPRO_PREFIX_SAVED}" CACHE PATH "Install prefix" FORCE)
  endif()

  list(APPEND linked_libs ${parseargslib})
else()
  message(STATUS "SKBUILD=ON: ParseArgs disabled (Python/Cython build)")
endif()


# -------------------------------------------------------------------------
# Install layout
# -------------------------------------------------------------------------
set(CMAKE_DEBUG_POSTFIX d)

set(DNETPRO_INSTALL_BIN_DIR     "${CMAKE_INSTALL_BINDIR}" CACHE PATH "Install dir for executables")
set(DNETPRO_INSTALL_LIB_DIR     "${CMAKE_INSTALL_LIBDIR}" CACHE PATH "Install dir for libraries")
set(DNETPRO_INSTALL_INCLUDE_DIR "${CMAKE_INSTALL_INCLUDEDIR}/DNetPRO" CACHE PATH "Install dir for headers")
set(DNETPRO_INSTALL_CMAKE_DIR   "${CMAKE_INSTALL_LIBDIR}/cmake/DNetPRO" CACHE PATH "Install dir for CMake package files")

# Variables used by .pc.in templates (keep compatibility)
set(INSTALL_BIN_DIR     "${DNETPRO_INSTALL_BIN_DIR}")
set(INSTALL_LIB_DIR     "${DNETPRO_INSTALL_LIB_DIR}")
set(INSTALL_INCLUDE_DIR "${DNETPRO_INSTALL_INCLUDE_DIR}")
set(INSTALL_CMAKE_DIR   "${DNETPRO_INSTALL_CMAKE_DIR}")

# Determine core target name based on SKBUILD
if (SKBUILD)
  find_package(Cython REQUIRED)
  find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module NumPy)
  #find_package(PythonExtensions REQUIRED) # invalid for scikit-build-core
  set (DNetPROlib libDNetPRO)
else ()
  set (DNetPROlib DNetPRO)
endif ()

# generate pkg-config + version header
configure_file(
  "${CMAKE_CURRENT_LIST_DIR}/DNetPRO.pc.in"
  "${CMAKE_CURRENT_BINARY_DIR}/DNetPRO.pc" @ONLY
)
message(STATUS "Pkg-config generated")

configure_file(
  "${CMAKE_CURRENT_LIST_DIR}/include/version.h.in"
  "${CMAKE_CURRENT_LIST_DIR}/include/version.h" @ONLY
)

file (GLOB DNetPRO_SRC    "${CMAKE_CURRENT_LIST_DIR}/src/*.cpp"  )
file (GLOB DNetPRO_HEADER "${CMAKE_CURRENT_LIST_DIR}/include/*.h")
file (GLOB DNetPRO_HPP    "${CMAKE_CURRENT_LIST_DIR}/hpp/*.hpp"  )

include_directories("${CMAKE_CURRENT_LIST_DIR}/include")
include_directories("${CMAKE_CURRENT_LIST_DIR}/hpp")

# Make relative paths absolute (needed for config)
foreach (p LIB BIN INCLUDE CMAKE)
  set (var INSTALL_${p}_DIR)
  if (NOT IS_ABSOLUTE "${${var}}")
    set (FULLPATH_${var} "${CMAKE_INSTALL_PREFIX}/${${var}}")
  else()
    set (FULLPATH_${var} "${${var}}")
  endif()
endforeach()

# -------------------------------------------------------------------------
# SUMMARY
# -------------------------------------------------------------------------
message(STATUS ""                                                                          )
message(STATUS "=================== DNetPRO configuration Summary =================="      )
message(STATUS "   DNetPRO version: ${DNETPRO_VERSION}"                                    )
message(STATUS ""                                                                          )
message(STATUS "   Build type      : ${CMAKE_BUILD_TYPE}"                                  )
message(STATUS "   Install prefix  : ${CMAKE_INSTALL_PREFIX}"                              )
message(STATUS "   Install bin dir : ${DNETPRO_INSTALL_BIN_DIR}"                           )
message(STATUS "   Install lib dir : ${DNETPRO_INSTALL_LIB_DIR}"                           )
message(STATUS "   Install inc dir : ${DNETPRO_INSTALL_INCLUDE_DIR}"                       )
message(STATUS "   Install cmake dir: ${DNETPRO_INSTALL_CMAKE_DIR}"                        )
message(STATUS ""                                                                          )
message(STATUS "   OpenMP support : ${OMP}"                                                )
message(STATUS "   Cython support : ${SKBUILD}"                                            )
message(STATUS "   Documentation support : ${BUILD_DOCS}"                                  )
message(STATUS ""                                                                          )

# print compiler flags
string(REPLACE " " ";" CMAKE_CXX_FLAGS_LIST "${CMAKE_CXX_FLAGS}")
message(STATUS "      C++ Compiler : ${CMAKE_CXX_COMPILER}"                                )
message(STATUS "      C++ flags    :"                                                      )
foreach(FLAG ${CMAKE_CXX_FLAGS_LIST})
  message(STATUS "                    * ${FLAG}"                                           )
endforeach()

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  message(STATUS "      C++ Debug flags:"                                                  )
  string(REPLACE " " ";" _DBG "${CMAKE_CXX_FLAGS_DEBUG}")
  foreach(FLAG ${_DBG})
    message(STATUS "                    * ${FLAG}"                                         )
  endforeach()
elseif (CMAKE_BUILD_TYPE STREQUAL "Release")
  message(STATUS "      C++ Release flags  :"                                              )
  string(REPLACE " " ";" _REL "${CMAKE_CXX_FLAGS_RELEASE}")
  foreach(FLAG ${_REL})
    message(STATUS "                    * ${FLAG}"                                         )
  endforeach()
endif()

message(STATUS "      Linker flags : "                                                     )
foreach(FLAG ${linked_libs})
  message(STATUS "                    * ${FLAG}"                                           )
endforeach(FLAG)
message(STATUS ""                                                                          )

# -------------------------------------------------------------------------
# MAIN RULES
# -------------------------------------------------------------------------
# Build static PIC core for SKBUILD (link into Python extension); shared library otherwise
if (SKBUILD)
  set(_DNETPRO_LIBTYPE STATIC)
else()
  set(_DNETPRO_LIBTYPE SHARED)
endif()

add_library(
  ${DNetPROlib}
  ${_DNETPRO_LIBTYPE}
  ${DNetPRO_SRC} ${DNetPRO_HEADER} ${DNetPRO_HPP}
)
set_target_properties(${DNetPROlib} PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Ensure the actual library name is always libDNetPRO (so pkg-config uses -lDNetPRO)
set_target_properties(${DNetPROlib} PROPERTIES OUTPUT_NAME DNetPRO)

target_include_directories(${DNetPROlib}
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/hpp>
  $<INSTALL_INTERFACE:${DNETPRO_INSTALL_INCLUDE_DIR}>
)

target_link_libraries(${DNetPROlib} ${linked_libs})

if (NOT SKBUILD)
  set_target_properties(${DNetPROlib} PROPERTIES PUBLIC_HEADER "${DNetPRO_HEADER}")

  install(TARGETS ${DNetPROlib} EXPORT DNetPROTargets
    RUNTIME DESTINATION "${DNETPRO_INSTALL_BIN_DIR}"
    LIBRARY DESTINATION "${DNETPRO_INSTALL_LIB_DIR}"
    ARCHIVE DESTINATION "${DNETPRO_INSTALL_LIB_DIR}"
    PUBLIC_HEADER DESTINATION "${DNETPRO_INSTALL_INCLUDE_DIR}"
    COMPONENT dev
  )

  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/DNetPRO.pc"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig"
  )

  add_custom_target(example_test)
  add_executable(DNetPRO_couples "${CMAKE_CURRENT_LIST_DIR}/example/DNetPRO_couples.cpp")
  add_dependencies(example_test DNetPRO_couples)
  target_link_libraries(DNetPRO_couples ${linked_libs} ${DNetPROlib})
  install(TARGETS DNetPRO_couples DESTINATION "${DNETPRO_INSTALL_BIN_DIR}")
endif ()

# -------------------------------------------------------------------------
# EXPORT DNetPRO
# -------------------------------------------------------------------------
if (NOT SKBUILD)

  install(EXPORT DNetPROTargets
    FILE DNetPROTargets.cmake
    NAMESPACE DNetPRO::
    DESTINATION "${DNETPRO_INSTALL_CMAKE_DIR}"
  )

  include(CMakePackageConfigHelpers)

  # Generate Config + Version files
  configure_package_config_file(
    "${CMAKE_CURRENT_LIST_DIR}/DNetPROConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/DNetPROConfig.cmake"
    INSTALL_DESTINATION "${DNETPRO_INSTALL_CMAKE_DIR}"
    PATH_VARS DNETPRO_INSTALL_INCLUDE_DIR DNETPRO_INSTALL_LIB_DIR
  )

  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/DNetPROConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY SameMajorVersion
  )

  install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/DNetPROConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/DNetPROConfigVersion.cmake"
    DESTINATION "${DNETPRO_INSTALL_CMAKE_DIR}"
  )

  # optional: register build-tree package
  export(PACKAGE DNetPRO)

endif()

# -------------------------------------------------------------------------
# Cython subdir (SKBUILD only)
# -------------------------------------------------------------------------
if (SKBUILD)
  add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/DNetPRO/source")
endif ()

# This must be the latest subdirectory included!!
if (BUILD_DOCS)
  add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/docs/")
endif ()
