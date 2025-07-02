add_library(NeuralNetwork STATIC
    neuralnet.cpp
    neuralnet.h
)
target_include_directories(NeuralNetwork PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(NeuralNetwork PRIVATE OpenMP::OpenMP_CXX)

set_target_properties(NeuralNetwork PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)
