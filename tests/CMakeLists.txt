
file(GLOB SRCS "*.cpp")
add_executable(matchem_tests ${SRCS})

target_include_directories(matchem_tests PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(matchem_tests matchemlib)

add_test(NAME full_test_1 COMMAND ./tests/matchem_tests test_one WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
