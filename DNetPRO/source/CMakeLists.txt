# If the pyx file is a C++ file, we should specify that here.

set_source_files_properties( "${CMAKE_CURRENT_SOURCE_DIR}/DNetPRO.pyx" PROPERTIES CYTHON_IS_CXX TRUE )

# Include *.pxd directory

include_directories("${CMAKE_SOURCE_DIR}/DNetPRO/lib")

# Multi-file cython modules do not appear to be working at the moment.
cython_add_module(DNetPRO "${CMAKE_CURRENT_SOURCE_DIR}/DNetPRO.pyx")
target_compile_definitions(DNetPRO PRIVATE -D__dnet__)
target_link_libraries(DNetPRO ${linked_libs} ${dnetprolib})

# Installation of targets
install(TARGETS DNetPRO                                 DESTINATION "${CMAKE_SOURCE_DIR}/DNetPRO/lib")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/DNetPRO.cpp" DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}")

add_custom_target(PyDNetPRO)
add_dependencies(PyDNetPRO dnetpro)
add_dependencies(PyDNetPRO DNetPRO)

# setup installation

configure_file(
  "${CMAKE_SOURCE_DIR}/setup.py.in"
  "${CMAKE_SOURCE_DIR}/setup_.py" @ONLY)
configure_file(
  "${CMAKE_SOURCE_DIR}/DNetPRO/__version__.py.in"
  "${CMAKE_SOURCE_DIR}/DNetPRO/__version__.py" @ONLY)
configure_file(
  "${CMAKE_SOURCE_DIR}/setup.cfg.in"
  "${CMAKE_SOURCE_DIR}/setup.cfg" @ONLY)

# NOTE: the current installation is performed in development mode
add_custom_target(setup_DNetPRO ALL
  DEPENDS PyDNetPRO
  COMMAND ${PYTHON_EXECUTABLE} "${CMAKE_SOURCE_DIR}/setup_.py" develop --user
  #COMMAND ${PYTHON_EXECUTABLE} "${CMAKE_SOURCE_DIR}/setup_.py" install --user
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  COMMENT "Setup Python installation"
)
