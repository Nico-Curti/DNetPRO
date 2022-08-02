
set(ENV{LD_LIBRARY_PATH} "${CMAKE_BINARY_DIR}")
execute_process(COMMAND ${SPHINX_EXECUTABLE} -b html
                        # Tell Breathe where to find the Doxygen output
                        -Dbreathe_projects.CatCutifier=${DOXYGEN_OUTPUT_DIR}
                        ${SPHINX_SOURCE} ${SPHINX_BUILD})
