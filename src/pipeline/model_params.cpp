#include "model_params.h"

bool getNeuralNetworkParams(const ProgramOptions& options, NeuralNetworkParams& out_params) {
    createDirectory("../saved_models/neural_networks");

    if (options.use_custom_params && options.params.size() >= 4) {
        out_params.inputSize = std::stoi(options.params[0]);
        out_params.outputSize = std::stoi(options.params[1]);
        out_params.hiddenSize = std::stoi(options.params[2]);
        out_params.learningRate = std::stod(options.params[3]);
    } else {
        out_params.inputSize = 4;
        out_params.outputSize = 1;
        out_params.hiddenSize = 8;
        out_params.learningRate = 0.01;

        std::cout << "Default Neural Network Parameters:\n";
        std::cout << "Input size: " << out_params.inputSize << "\n";
        std::cout << "Output size: " << out_params.outputSize << "\n";
        std::cout << "Hidden size: " << out_params.hiddenSize << "\n";
        std::cout << "Learning rate: " << out_params.learningRate << "\n";
    }
    return true;
}

bool getBaggingParams(const ProgramOptions& options, BaggingParams& out_params) {
    createDirectory("../saved_models/bagging_models");

    if (options.use_custom_params && options.params.size() >= 5) {
        out_params.numModels = std::stoi(options.params[0]);
        out_params.inputSize = std::stoi(options.params[1]);
        out_params.outputSize = std::stoi(options.params[2]);
        out_params.hiddenSize = std::stoi(options.params[3]);
        out_params.learningRate = std::stod(options.params[4]);
    } else {
        out_params.numModels = 10;
        out_params.inputSize = 4;
        out_params.outputSize = 1;
        out_params.hiddenSize = 8;
        out_params.learningRate = 0.01;

        std::cout << "Default Bagging Parameters:\n";
        std::cout << "Number of models: " << out_params.numModels << "\n";
        std::cout << "Input size: " << out_params.inputSize << "\n";
        std::cout << "Output size: " << out_params.outputSize << "\n";
        std::cout << "Hidden size: " << out_params.hiddenSize << "\n";
        std::cout << "Learning rate: " << out_params.learningRate << "\n";
    }
    return true;
}

bool getBoostingParams(const ProgramOptions& options, BoostingParams& out_params) {
    createDirectory("../saved_models/boosting_models");

    if (options.use_custom_params && options.params.size() >= 5) {
        out_params.numModels = std::stoi(options.params[0]);
        out_params.inputSize = std::stoi(options.params[1]);
        out_params.outputSize = std::stoi(options.params[2]);
        out_params.hiddenSize = std::stoi(options.params[3]);
        out_params.learningRate = std::stod(options.params[4]);
    } else {
        out_params.numModels = 10;
        out_params.inputSize = 4;
        out_params.outputSize = 1;
        out_params.hiddenSize = 8;
        out_params.learningRate = 0.01;

        std::cout << "Default Boosting NN Parameters:\n";
        std::cout << "Number of models: " << out_params.numModels << "\n";
        std::cout << "Input size: " << out_params.inputSize << "\n";
        std::cout << "Output size: " << out_params.outputSize << "\n";
        std::cout << "Hidden size: " << out_params.hiddenSize << "\n";
        std::cout << "Learning rate: " << out_params.learningRate << "\n";
    }
    return true;
}
