cmake_minimum_required (VERSION 3.9.0)

# DNetPRO Version
set (DNETPRO_MAJOR    1)
set (DNETPRO_MINOR    0)
set (DNETPRO_REVISION 0)
set (DNETPRO_VERSION ${DNETPRO_MAJOR}.${DNETPRO_MINOR}.${DNETPRO_REVISION})

set (CMAKE_PROJECT_HOMEPAGE_URL "https://github.com/Nico-Curti/DNetPRO")
set (CMAKE_PROJECT_DESCRIPTION "Discriminant Analysis with Network Processing")

message (STATUS "DNetPRO VERSION: ${DNETPRO_VERSION}")

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

project (DNetPRO LANGUAGES CXX VERSION ${DNETPRO_VERSION} DESCRIPTION "Discriminant Network Processing - Feature Selection")

enable_language(CXX)
# required by parse_args library
if ( ( "${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU" AND ${CMAKE_CXX_COMPILER_VERSION} LESS 7) OR ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" AND ${CMAKE_CXX_COMPILER_VERSION} LESS_EQUAL 4) )
  set (CMAKE_CXX_STANDARD 14)
else ()
  set (CMAKE_CXX_STANDARD 17)
endif()

add_definitions (-DMAJOR=${DNETPRO_MAJOR} -DMINOR=${DNETPRO_MINOR} -DREVISION=${DNETPRO_REVISION})

#################################################################
#                         COMPILE OPTIONS                       #
#################################################################

option (OMP         "Enable OpenMP              support" OFF )
option (PYWRAP      "Enable Cython              support" OFF )
option (BUILD_DOCS  "Enable Documentaion builid support" OFF )

#################################################################
#                         SETTING VARIABLES                     #
#################################################################

set (CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules/" ${CMAKE_MODULE_PATH})

if ( NOT APPLE )
  set (CMAKE_SKIP_BUILD_RPATH             FALSE )
  set (CMAKE_BUILD_WITH_INSTALL_RPATH     FALSE )
  set (CMAKE_INSTALL_RPATH_USE_LINK_PATH  TRUE  )
endif ()

# make sure that the default is a RELEASE
set (default_build_type "Release")
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message (STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set (CMAKE_BUILD_TYPE "${default_build_type}" CACHE
    STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property (CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif ()

if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}" CACHE PATH "Install prefix" FORCE)
endif()

if ( CMAKE_COMPILER_IS_GNUCXX )
  add_compile_options (-Wall -Wextra -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -Wno-narrowing -Wpedantic)
  if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    string (REGEX REPLACE "-O[0123]" "-Og" CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG})
    set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG" )
    set (CMAKE_CXX_FLAGS_RELEASE "")
  elseif (CMAKE_BUILD_TYPE STREQUAL "Release")
    string (REGEX REPLACE "-O3" "-Ofast" CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
    set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DNDEBUG" )
    set (CMAKE_CXX_FLAGS_DEBUG "")
  endif ()
endif ()

if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
  add_compile_options (-Wno-deprecated -Wno-writable-strings)
  if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    string (REGEX REPLACE "-O0" "-Og" CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG})
    set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG" )
  elseif (CMAKE_BUILD_TYPE STREQUAL "Release")
    string (REGEX REPLACE "-O3" "-Ofast" CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
    set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DNDEBUG" )
  endif ()
endif ()

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


#################################################################
#                         PARSE OPTIONS                         #
#################################################################

if (OMP)
  find_package (OpenMP REQUIRED)
  if (OPENMP_FOUND)
    message (STATUS "OpenMP found")
    if (OpenMP_CXX_VERSION_MAJOR LESS_EQUAL 4 AND OpenMP_CXX_VERSION_MINOR LESS 5)
      message (FATAL_ERROR " Your OpenMP is too old. Required OpenMP 4.5. Please upgrade.")
    endif()
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    if (APPLE)
      list (APPEND linked_libs OpenMP::OpenMP_CXX)
    endif ()
    add_definitions (-DNUM_AVAILABLE_THREADS=${NUM_TH})
  endif ()
else ()
  message (STATUS "OpenMP disabled")
  set (NUM_TH 1)
  add_definitions (-DNUM_AVAILABLE_THREADS=1)
endif ()

if (FORCE_USE_SUBMODULES)
  message (STATUS "Use ParseArgs submodule")
  add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/modules/parse_args")
  include_directories("${CMAKE_CURRENT_SOURCE_DIR}/modules/parse_args/include/")
  include_directories("${CMAKE_CURRENT_SOURCE_DIR}/modules/parse_args/hpp/")
  set (parseargslib parseargs)
else ()
  message (STATUS "Looking for ParseArgs library")
  find_package (ParseArgs HINTS ParseArgs_DIR)
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
  endif ()
endif ()

#################################################################
#                         SETTING DIRECTORIES                   #
#################################################################

set(INSTALL_BIN_DIR "${CMAKE_CURRENT_LIST_DIR}/bin" CACHE PATH "Path where exe and dll will be installed")
set(INSTALL_LIB_DIR "${CMAKE_CURRENT_LIST_DIR}/lib" CACHE PATH "Path where lib will be installed")

set(INSTALL_INCLUDE_DIR "include/DNetPRO" CACHE PATH "Path where headers will be installed")
set(INSTALL_CMAKE_DIR   "share/DNetPRO"   CACHE PATH "Path where cmake configs will be installed")

set(CMAKE_DEBUG_POSTFIX d)

file (GLOB DNetPRO_SRC    "${CMAKE_CURRENT_LIST_DIR}/src/*.cpp"  )
file (GLOB DNetPRO_HEADER "${CMAKE_CURRENT_LIST_DIR}/include/*.h")
file (GLOB DNetPRO_HPP    "${CMAKE_CURRENT_LIST_DIR}/hpp/*.hpp"  )

include_directories("${CMAKE_CURRENT_LIST_DIR}/include")
include_directories("${CMAKE_CURRENT_LIST_DIR}/hpp")

if (PYWRAP)
  include( UseCython )
  find_package (Python REQUIRED COMPONENTS Interpreter)
  find_package(NumPy REQUIRED)
  include_directories(${NumPy_INCLUDE_DIRS})
  add_definitions (-DNPY_NO_DEPRECATED_API)
endif()

set (dnetprolib dnetpro)

configure_file(
  "${CMAKE_CURRENT_LIST_DIR}/DNetPRO.pc.in"
  "${CMAKE_CURRENT_LIST_DIR}/DNetPRO.pc" @ONLY)
message(STATUS "Pkg-config generated")

# allow the export of LD_LIBRARY_PATH env variable
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Make relative paths absolute (needed later on)
foreach (p LIB BIN INCLUDE CMAKE)
  set (var INSTALL_${p}_DIR)
  if (NOT IS_ABSOLUTE "${${var}}")
    set (FULLPATH_${var} "${CMAKE_INSTALL_PREFIX}/${${var}}")
  endif()
endforeach()

#################################################################
#                          SUMMARY                              #
#################################################################

message(STATUS ""                                                                          )
message(STATUS "=================== DNetPRO configuration Summary =================="      )
message(STATUS "   DNetPRO version: ${DNETPRO_VERSION}"                                    )
message(STATUS ""                                                                          )
message(STATUS "   Build type : ${CMAKE_BUILD_TYPE}"                                       )
message(STATUS "   C++ :"                                                                  )
message(STATUS "      C++ Compiler : ${CMAKE_CXX_COMPILER}"                                )
message(STATUS "      C++ flags    :"                                                      )
foreach(FLAG ${CMAKE_CXX_FLAGS_LIST})
  message(STATUS "                    * ${FLAG}"                                           )
endforeach(FLAG)
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  message(STATUS "      C++ Debug flags:"                                                  )
  foreach(FLAG ${CMAKE_CXX_FLAGS_DEBUG})
    message(STATUS "                    * ${FLAG}"                                         )
  endforeach(FLAG)
elseif (CMAKE_BUILD_TYPE STREQUAL "Release")
  message(STATUS "      C++ Release flags  :"                                              )
  foreach(FLAG ${CMAKE_CXX_FLAGS_RELEASE})
    message(STATUS "                    * ${FLAG}"                                         )
  endforeach(FLAG)
endif()
message(STATUS "      Linker flags : "                                                     )
foreach(FLAG ${linked_libs})
  message(STATUS "                    * ${FLAG}"                                           )
endforeach(FLAG)
message(STATUS ""                                                                          )
message(STATUS "   OpenMP support : ${OMP}"                                                )
message(STATUS "   Pythonic support : ${PYWRAP}"                                           )
message(STATUS "   Documentation support : ${BUILD_DOCS}"                                  )
message(STATUS ""                                                                          )

#################################################################
#                         MAIN RULES                            #
#################################################################

add_library(${dnetprolib} SHARED ${DNetPRO_SRC} ${DNetPRO_HEADER} ${DNetPRO_HPP})
set_property(TARGET ${dnetprolib} PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(${dnetprolib}
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/hpp>
  $<INSTALL_INTERFACE:${INSTALL_INCLUDE_DIR}>)
target_link_libraries(${dnetprolib} ${linked_libs})

list (APPEND ALL_DNetPRO_HEADER ${DNetPRO_HEADER})
list (APPEND ALL_DNetPRO_HEADER ${DNetPRO_HPP})
set_target_properties(${dnetprolib} PROPERTIES PUBLIC_HEADER "${ALL_DNetPRO_HEADER}")

install(TARGETS ${dnetprolib} EXPORT DNetPROTargets
  RUNTIME DESTINATION "${INSTALL_BIN_DIR}"
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}"
  PUBLIC_HEADER DESTINATION "${INSTALL_INCLUDE_DIR}"
  COMPONENT dev
)

add_executable(DNetPRO_couples "${CMAKE_CURRENT_LIST_DIR}/example/DNetPRO_couples.cpp")
target_link_libraries(DNetPRO_couples ${linked_libs} ${parseargslib} ${dnetprolib})
if (FORCE_USE_SUBMODULES)
  add_dependencies(DNetPRO_couples parseargs)
endif ()
install(TARGETS DNetPRO_couples DESTINATION "${INSTALL_BIN_DIR}")

if (PYWRAP)
  add_subdirectory ("${CMAKE_CURRENT_LIST_DIR}/DNetPRO/source")
endif ()

if (BUILD_DOCS)
  add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/docs/")
endif ()

#################################################################
#                          EXPORT DNetPRO                         #
#################################################################

install (EXPORT DNetPROTargets
  FILE DNetPROTargets.cmake
  NAMESPACE DNetPRO::
  DESTINATION "${INSTALL_CMAKE_DIR}"
)

# Export the package for use from the build-tree (this registers the build-tree with a global CMake-registry)
export (PACKAGE DNetPRO)

# Create the DNetPROConfig.cmake
# First of all we compute the relative path between the cmake config file and the include path
file (RELATIVE_PATH REL_INCLUDE_DIR "${FULLPATH_INSTALL_CMAKE_DIR}" "${FULLPATH_INSTALL_INCLUDE_DIR}")
set (CONF_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}" "${PROJECT_BINARY_DIR}")
configure_file (DNetPROConfig.cmake.in "${PROJECT_BINARY_DIR}/DNetPROConfig.cmake" @ONLY)
set (CONF_INCLUDE_DIRS "\${DNetPRO_CMAKE_DIR}/${REL_INCLUDE_DIR}")
configure_file (DNetPROConfig.cmake.in "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/DNetPROConfig.cmake" @ONLY)

# Create the DNetPROConfigVersion.cmake
include (CMakePackageConfigHelpers)
write_basic_package_version_file ("${PROJECT_BINARY_DIR}/DNetPROConfigVersion.cmake"
  COMPATIBILITY SameMajorVersion
)

install (FILES
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/DNetPROConfig.cmake"
  "${PROJECT_BINARY_DIR}/DNetPROConfigVersion.cmake"
  DESTINATION "${INSTALL_CMAKE_DIR}"
)
