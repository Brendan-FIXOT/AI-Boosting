#include "run_models.h"
#include "../main/utility.h"
#include "../ensemble/bagging/bagging.h"
#include "../ensemble/boosting/boosting.h"
#include "../functions/neuralnet/neuralnet.h"
#include "../model_comparison/model_comparison.h"

void runSingleNeuralNetwork(NeuralNetworkParams params, DataParams data_params) {
    std::cout << "Training a Neural Network model...\n";

    NeuralNetwork model(params.inputSize, params.outputSize, params.hiddenSize, params.learningRate);

    double train_time = 0.0;
    double eval_time = 0.0;
    double score = 0.0;

    trainAndEvaluateModel(model, data_params.X_train, data_params.rowLength,
      data_params.y_train, data_params.X_test, data_params.y_test,
      score, train_time, eval_time);

    //saveModel(model);

    ModelResults results;
    results.model_name = "Neural Network";
    results.mse_or_mae = score;
    results.training_time = train_time;
    results.evaluation_time = eval_time;

    results.parameters["input_size"] = params.inputSize;
    results.parameters["output_size"] = params.outputSize;
    results.parameters["hidden_size"] = params.hiddenSize;
    results.parameters["learning_rate"] = params.learningRate;

    ModelComparison::saveResults(results);
}

void runBaggingNeuralNetwork(BaggingParams params, DataParams data_params) {
    std::cout << "Training a Bagging Neural Network model...\n";

    Bagging model(params.numModels, params.inputSize, params.outputSize,
                    params.hiddenSize, params.learningRate);

    double train_time = 0.0;
    double eval_time = 0.0;
    double score = 0.0;

    trainAndEvaluateModel(model, data_params.X_train, data_params.rowLength,
      data_params.y_train, data_params.X_test, data_params.y_test,
      score, train_time, eval_time);

    //saveModel(model);

    ModelResults results;
    results.model_name = "Bagging";
    results.mse_or_mae = score;
    results.training_time = train_time;
    results.evaluation_time = eval_time;

    results.parameters["num_models"] = params.numModels;
    results.parameters["input_size"] = params.inputSize;
    results.parameters["output_size"] = params.outputSize;
    results.parameters["hidden_size"] = params.hiddenSize;
    results.parameters["learning_rate"] = params.learningRate;

    ModelComparison::saveResults(results);
}

void runBoostingNeuralNetwork(BoostingParams params, DataParams data_params) {
    std::cout << "Training a Boosting Neural Network model...\n";

    Boosting model(params.numModels, params.inputSize, params.outputSize,
                     params.hiddenSize, params.learningRate);

    double train_time = 0.0;
    double eval_time = 0.0;
    double score = 0.0;

    trainAndEvaluateModel(model,
      data_params.X_train, data_params.rowLength, data_params.y_train,
      data_params.X_test, data_params.y_test,
      score, train_time, eval_time);  

    //saveModel(model);

    ModelResults results;
    results.model_name = "Boosting";
    results.mse_or_mae = score;
    results.training_time = train_time;
    results.evaluation_time = eval_time;

    results.parameters["num_models"] = params.numModels;
    results.parameters["input_size"] = params.inputSize;
    results.parameters["output_size"] = params.outputSize;
    results.parameters["hidden_size"] = params.hiddenSize;
    results.parameters["learning_rate"] = params.learningRate;

    ModelComparison::saveResults(results);
}
