# build C++ APIs with Doxygen

find_package(Doxygen REQUIRED)

set(DOXYGEN_OUT "${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile")

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in"
  "${DOXYGEN_OUT}"
  @ONLY
)
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/source/conf.py.in"
  "${CMAKE_CURRENT_SOURCE_DIR}/source/conf.py"
  @ONLY
)

add_custom_target(cpp_docs
  DEPENDS ${DNetPROlib}
  COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  COMMENT "Generating C++ API documentation with Doxygen"
  VERBATIM
)

# build Python APIs with Sphinx
# PYWRAP is deprecated: build Python docs only if the cython build is enabled (SKBUILD=ON)
if (SKBUILD)

  find_package(Sphinx REQUIRED)

  set(SPHINX_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/source/")
  set(SPHINX_BUILD  "${CMAKE_CURRENT_SOURCE_DIR}/build")

  add_custom_target(py_docs
    DEPENDS cpp_docs DNetPRO
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMAND "${SPHINX_EXECUTABLE}" -b html "${SPHINX_SOURCE}" "${SPHINX_BUILD}"
    COMMENT "Generating Python API documentation with Sphinx"
    VERBATIM
  )

  # Workaround for the lack of post_install steps.
  install(CODE "set(CMAKE_BINARY_DIR \"${CMAKE_BINARY_DIR}\")")
  install(CODE "set(SPHINX_SOURCE \"${SPHINX_SOURCE}\")")
  install(CODE "set(SPHINX_BUILD \"${SPHINX_BUILD}\")")
  install(CODE "set(SPHINX_EXECUTABLE \"${SPHINX_EXECUTABLE}\")")
  install(CODE "set(DOXYGEN_OUTPUT_DIR \"${DOXYGEN_OUTPUT_DIR}\")")
  install(SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/RunSphinx.cmake")

endif ()