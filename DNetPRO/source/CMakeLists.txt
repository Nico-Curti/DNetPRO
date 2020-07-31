# If the pyx file is a C++ file, we should specify that here.

set(PYC_SRC_DIR    ${PYC_DIR}/source       CACHE PATH "Path where find pyx files" )
set(PYC_INC_DIR    ${PYC_DIR}/lib          CACHE PATH "Path where find pxd files" )

set_source_files_properties( ${PYC_SRC_DIR}/DNetPRO.pyx PROPERTIES CYTHON_IS_CXX TRUE )

# Include *.pxd directory

include_directories(${PYC_INC_DIR})

# Multi-file cython modules do not appear to be working at the moment.
cython_add_module( DNetPRO DNetPRO.pyx
                  ${SRC_DIR}/dnetpro_couples.cpp ${INC_DIR}/dnetpro_couples.h
                  ${SRC_DIR}/score.cpp ${INC_DIR}/score.h
                  ${SRC_DIR}/utility.cpp ${INC_DIR}/utility.h
                  ${SRC_DIR}/misc.cpp ${INC_DIR}/misc.h
                  )
target_compile_definitions(DNetPRO PRIVATE -D__dnet__ -DNPY_NO_DEPRECATED_API NPY_1_7_API_VERSION)
target_link_libraries(DNetPRO ${linked_libs} ${dnetlib})

# Installation of targets
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/DNetPRO.cpp      DESTINATION ${PYC_SRC_DIR})
install(TARGETS DNetPRO          DESTINATION ${PYC_INC_DIR})
