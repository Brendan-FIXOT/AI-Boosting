add_library(Data_Clean
    data_clean.cpp
)

target_include_directories(Data_Clean PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(Data_Clean PUBLIC NeuralNetwork)
target_link_libraries(Data_Clean PUBLIC FunctionsMath)