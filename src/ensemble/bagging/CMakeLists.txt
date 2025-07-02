add_library(Bagging bagging.cpp)

target_link_libraries(Bagging PRIVATE OpenMP::OpenMP_CXX)

target_include_directories(Bagging PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(Bagging PUBLIC NeuralNetwork)
target_link_libraries(Bagging PRIVATE LossFunction)
