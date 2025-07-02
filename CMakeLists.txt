cmake_minimum_required(VERSION 3.10)
project(DecisionTreeEnsembleModel)

set(CMAKE_CXX_STANDARD 17)

option(OPENMP "Enable OpenMP" ON)

if(OPENMP)
    find_package(OpenMP REQUIRED)
    if(OpenMP_CXX_FOUND)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        message(STATUS "OpenMP enabled")
    else()
        message(WARNING "OpenMP requested but not found")
    endif()
else()
    message(STATUS "OpenMP disabled")
endif()

if(OPENMP)
    set(OPENMP_LIB OpenMP::OpenMP_CXX)
else()
    set(OPENMP_LIB "")  # Empty if OpenMP is OFF
endif()

find_package(OpenMP REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")


add_subdirectory(src/functions/io)
add_subdirectory(src/functions/neuralnet)
add_subdirectory(src/functions/loss)
add_subdirectory(src/functions/math)
add_subdirectory(src/ensemble/bagging)
add_subdirectory(src/ensemble/boosting)
add_subdirectory(src/data_clean)
add_subdirectory(src/model_comparison)
add_subdirectory(src/pipeline)


# Option to enable/disable tests
option(ENABLE_TESTS "Enable building tests" OFF)

if (ENABLE_TESTS)
    include(FetchContent)
    FetchContent_Declare(
      googletest
      GIT_REPOSITORY https://github.com/google/googletest.git
      GIT_TAG release-1.12.1
    )
    FetchContent_MakeAvailable(googletest)

    enable_testing()

    # Tests
    add_executable(math_functions_test tests/math_functions_test.cpp)
    target_link_libraries(math_functions_test
        gtest_main
        NeuralNetwork
        FunctionsMath
    )

    add_executable(boosting_test tests/boosting_test.cpp)
    target_link_libraries(boosting_test
        gtest_main
        NeuralNetwork
        FunctionsMath
        Boosting
        LossFunction
    )

    add_executable(bagging_test tests/bagging_test.cpp)
    target_link_libraries(bagging_test
        gtest_main
        NeuralNetwork
        FunctionsMath
        Bagging
        Boosting
        LossFunction
    )

    add_executable(cross_validation_test tests/cross_validation_test.cpp)
    target_link_libraries(cross_validation_test
        gtest_main
        NeuralNetwork
        FunctionsMath
        Bagging
        Boosting
        LossFunction
    )

    add_executable(data_clean_test tests/data_clean_test.cpp)
    target_link_libraries(data_clean_test
        gtest_main
        FunctionsMath
        NeuralNetwork
        Data_Clean
    )
endif()

add_executable(MainDataClean src/main/main_data_clean.cpp)
add_executable(MainEnsemble src/main/main.cpp src/main/utility.cpp)
add_executable(MainComparison src/main/main_comparison.cpp)

target_link_libraries(MainDataClean PRIVATE Data_Clean)

target_link_libraries(MainEnsemble PRIVATE
    FunctionsIO
    NeuralNetwork
    FunctionsMath
    Bagging
    Boosting
    model_comparison
    Data_Clean
    LossFunction
    Pipeline
    ${OPENMP_LIB}
)

target_include_directories(MainDataClean PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/functions/io
    ${CMAKE_CURRENT_SOURCE_DIR}/src/functions/neuralnet
    ${CMAKE_CURRENT_SOURCE_DIR}/src/functions/math
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data_clean
)

target_include_directories(MainEnsemble PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/functions/io
    ${CMAKE_CURRENT_SOURCE_DIR}/src/functions/neuralnet
    ${CMAKE_CURRENT_SOURCE_DIR}/src/functions/math
    ${CMAKE_CURRENT_SOURCE_DIR}/src/functions/loss
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ensemble/bagging
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ensemble/boosting
    ${CMAKE_CURRENT_SOURCE_DIR}/src/model_comparison
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data_clean
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pipeline
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main
    
)

target_include_directories(MainComparison PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/functions/io
    ${CMAKE_CURRENT_SOURCE_DIR}/src/functions/neuralnet
    ${CMAKE_CURRENT_SOURCE_DIR}/src/functions/math
    ${CMAKE_CURRENT_SOURCE_DIR}/src/functions/loss
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ensemble/bagging
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ensemble/boosting
    ${CMAKE_CURRENT_SOURCE_DIR}/src/model_comparison
)