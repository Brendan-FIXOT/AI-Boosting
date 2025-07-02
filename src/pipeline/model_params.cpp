#include "model_params.h"

bool getNeuralNetworkParams(const ProgramOptions& options, NeuralNetworkParams& out_params,
                            int inferred_input_size, int inferred_output_size) {
    createDirectory("../saved_models/neural_networks");

    out_params.inputSize = inferred_input_size;
    out_params.outputSize = inferred_output_size;

    if (options.use_custom_params && options.params.size() >= 2) {
        out_params.hiddenSize = std::stoi(options.params[0]);
        out_params.learningRate = std::stod(options.params[1]);
    } else {
        out_params.hiddenSize = 8;
        out_params.learningRate = 0.01;
    }

    std::cout << "Neural Network Parameters:\n";
    std::cout << "Input size: " << out_params.inputSize << "\n";
    std::cout << "Output size: " << out_params.outputSize << "\n";
    std::cout << "Hidden size: " << out_params.hiddenSize << "\n";
    std::cout << "Learning rate: " << out_params.learningRate << "\n";

    return true;
}

bool getBaggingParams(const ProgramOptions& options,
                      BaggingParams& out_params,
                      int inferred_input_size,
                      int inferred_output_size) {
    createDirectory("../saved_models/bagging_models");

    out_params.inputSize = inferred_input_size;
    out_params.outputSize = inferred_output_size;

    if (options.use_custom_params && options.params.size() >= 3) {
        out_params.numModels = std::stoi(options.params[0]);
        out_params.hiddenSize = std::stoi(options.params[1]);
        out_params.learningRate = std::stod(options.params[2]);
    } else {
        out_params.numModels = 10;
        out_params.hiddenSize = 8;
        out_params.learningRate = 0.01;
    }

    std::cout << "Bagging Parameters:\n";
    std::cout << "Number of models: " << out_params.numModels << "\n";
    std::cout << "Input size: " << out_params.inputSize << "\n";
    std::cout << "Output size: " << out_params.outputSize << "\n";
    std::cout << "Hidden size: " << out_params.hiddenSize << "\n";
    std::cout << "Learning rate: " << out_params.learningRate << "\n";

    return true;
}

bool getBoostingParams(const ProgramOptions& options,
                       BoostingParams& out_params,
                       int inferred_input_size,
                       int inferred_output_size) {
    createDirectory("../saved_models/boosting_models");

    out_params.inputSize = inferred_input_size;
    out_params.outputSize = inferred_output_size;

    if (options.use_custom_params && options.params.size() >= 3) {
        out_params.numModels = std::stoi(options.params[0]);
        out_params.hiddenSize = std::stoi(options.params[1]);
        out_params.learningRate = std::stod(options.params[2]);
    } else {
        out_params.numModels = 10;
        out_params.hiddenSize = 8;
        out_params.learningRate = 0.01;
    }

    std::cout << "Boosting Parameters:\n";
    std::cout << "Number of models: " << out_params.numModels << "\n";
    std::cout << "Input size: " << out_params.inputSize << "\n";
    std::cout << "Output size: " << out_params.outputSize << "\n";
    std::cout << "Hidden size: " << out_params.hiddenSize << "\n";
    std::cout << "Learning rate: " << out_params.learningRate << "\n";

    return true;
}