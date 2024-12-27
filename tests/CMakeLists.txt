configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cmake.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cmake.cfg.py
)

set (MLIR_TEST_DEPENDS
    FileCheck count not
    bf-opt
)

add_lit_testsuite(check-bf "Running the brainfuck regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${MLIR_TEST_DEPENDS}
)
