#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "DNetPRO::DNetPRO" for configuration "Release"
set_property(TARGET DNetPRO::DNetPRO APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(DNetPRO::DNetPRO PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libDNetPRO.so"
  IMPORTED_SONAME_RELEASE "libDNetPRO.so"
  )

list(APPEND _cmake_import_check_targets DNetPRO::DNetPRO )
list(APPEND _cmake_import_check_files_for_DNetPRO::DNetPRO "${_IMPORT_PREFIX}/lib/libDNetPRO.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
