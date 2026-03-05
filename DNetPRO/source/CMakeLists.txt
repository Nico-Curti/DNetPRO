# Include public headers
set(DNETPRO_CORE_INCLUDE_DIRS
  "${CMAKE_SOURCE_DIR}/include"
  "${CMAKE_SOURCE_DIR}/hpp"
)

set(PYX_FILE "${CMAKE_CURRENT_SOURCE_DIR}/DNetPRO.pyx")
set(GEN_CPP  "${CMAKE_CURRENT_BINARY_DIR}/DNetPRO.cpp")

# Output dir for *built* python module — produce it directly in the source package DNetPRO/lib
# (so import DNetPRO.lib.DNetPRO will work for editable installs too)
set(MODULE_OUTPUT_DIR "${PROJECT_SOURCE_DIR}/DNetPRO/lib")
file(MAKE_DIRECTORY "${MODULE_OUTPUT_DIR}")

# Generate C++ from Cython
add_custom_command(
  OUTPUT "${GEN_CPP}"
  COMMAND "${CYTHON_EXECUTABLE}" -3 --cplus
        -I "${CMAKE_SOURCE_DIR}/DNetPRO/lib"
        -I "${CMAKE_SOURCE_DIR}/DNetPRO"
        -o "${GEN_CPP}" "${PYX_FILE}"
  DEPENDS "${PYX_FILE}"
  COMMENT "Cythonizing ${PYX_FILE}"
  VERBATIM
)

# Build python module
add_library(DNetPRO MODULE "${GEN_CPP}")
#target_include_directories(DNetPRO PRIVATE ${DNETPRO_CORE_INCLUDE_DIRS})
target_include_directories(DNetPRO PRIVATE
  "${Python3_INCLUDE_DIRS}"
  "${Python3_NumPy_INCLUDE_DIRS}"
  "${PROJECT_SOURCE_DIR}/include"
)

configure_file(
  "${CMAKE_SOURCE_DIR}/DNetPRO/__version__.py.in"
  "${CMAKE_SOURCE_DIR}/DNetPRO/__version__.py" @ONLY
)

# NumPy headers (needed because Cython-generated code includes numpy/arrayobject.h)
if (DEFINED Python3_NumPy_INCLUDE_DIRS)
  target_include_directories(DNetPRO PRIVATE "${Python3_NumPy_INCLUDE_DIRS}")
elseif (DEFINED Python_NumPy_INCLUDE_DIRS)
  target_include_directories(DNetPRO PRIVATE "${Python_NumPy_INCLUDE_DIRS}")
endif()

# Make sure the module has no 'lib' prefix
set_target_properties(DNetPRO PROPERTIES PREFIX "" OUTPUT_NAME "DNetPRO")

# Place the built module file into MODULE_OUTPUT_DIR during build
set_target_properties(DNetPRO PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${MODULE_OUTPUT_DIR}"
  RUNTIME_OUTPUT_DIRECTORY "${MODULE_OUTPUT_DIR}"
  ARCHIVE_OUTPUT_DIRECTORY "${MODULE_OUTPUT_DIR}"
)

# Link the module against Python module interface (avoids linking libpythonX.Y.so)
# and against the core C++ shared library target (DNetPROlib) which should be defined in root CMakeLists.txt
# If the core target is named e.g. DNetPRO (shared), link to that target; otherwise link to the installed lib.
target_link_libraries(DNetPRO Python3::Module ${DNetPROlib})

# Determine extension suffix (portable)
if (DEFINED Python3_EXTENSION_SUFFIX)
  set(_pyext_suffix "${Python3_EXTENSION_SUFFIX}")
elseif(DEFINED Python3_SOABI)
  set(_pyext_suffix ".${Python3_SOABI}.so")
else()
  set(_pyext_suffix ".so")
endif()
set_target_properties(DNetPRO PROPERTIES SUFFIX "${_pyext_suffix}")


# IMPORTANT: set rpath so at runtime the module can find the core shared library (lib/DNetPRO.so)
# The module lives in: <project>/DNetPRO/lib/
# The core shared lib lives in: <project>/lib/
# Relative from module dir -> core dir: "../../lib"
# Use $ORIGIN which is interpreted at runtime as directory of the module file
set_target_properties(DNetPRO PROPERTIES
  BUILD_RPATH "$ORIGIN/../../lib"
  INSTALL_RPATH "$ORIGIN/../../lib"
  INSTALL_RPATH_USE_LINK_PATH TRUE
)

# Also ensure proper suffix/extension for python module — python_extension_module sets it if available
#python_extension_module(DNetPRO)

# Normal install step: install the module into package subdir for non-editable installs
install(TARGETS DNetPRO
  LIBRARY DESTINATION "DNetPRO/lib"
  RUNTIME DESTINATION "DNetPRO/lib"
)

install(
  FILES "${CMAKE_SOURCE_DIR}/DNetPRO/__version__.py"
  DESTINATION "DNetPRO"
)

# Keep a small post-build message
add_custom_command(TARGET DNetPRO POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E echo "Built Python extension: $<TARGET_FILE:DNetPRO>"
)
