#ifndef UTILITY
#define UTILITY

#include "../functions/neuralnet/neuralnet.h"
#include "../functions/math/math_functions.h"
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>

int adjustNumThreads(int numThreads);

template <typename T>
T getInputWithDefault(const std::string &prompt, T defaultValue);

struct ProgramOptions {
  int choice = 0;
  bool use_custom_params = false;
  bool load_request = false;
  std::string path_model_filename;
  std::vector<std::string> params;
};

ProgramOptions parseCommandLineArguments(int argc, char *argv[]);

void createDirectory(const std::string &path);

template <typename ModelType> 
void saveModel(ModelType &model) {
  std::cout << "Would you like to save this model? (1 = Yes, 0 = No): ";
  int save_model;
  std::cin >> save_model;

  if (save_model) {
    std::cout << "Enter the filename to save the model: ";
    std::string filename;
    std::cin >> filename;
    std::string path = "../saved_models/" + filename;
    model.save(path);
    std::cout << "Model saved successfully as " << filename << "\n";
  }
}

template <typename NeuralModel>
typename std::enable_if<std::is_same<NeuralModel, NeuralNetwork>::value>::type
trainAndEvaluateModel(NeuralModel& model,
                      const std::vector<double>& X_train,
                      int rowLength,
                      const std::vector<double>& y_train,
                      const std::vector<double>& X_test,
                      const std::vector<double>& y_test,
                      double& score,
                      double& train_duration_count,
                      double& eval_duration_count) {
    std::cout << "Starting neural network training...\n";

    auto train_start = std::chrono::high_resolution_clock::now();
    model.train(X_train, rowLength, y_train);
    auto train_end = std::chrono::high_resolution_clock::now();
    train_duration_count = std::chrono::duration<double>(train_end - train_start).count();
    std::cout << "Training time: " << train_duration_count << " seconds\n";

    auto eval_start = std::chrono::high_resolution_clock::now();
    std::vector<double> y_pred = model.evaluate(X_test, rowLength, y_test);
    score = Math::computeLossMSE(y_test, y_pred);
    auto eval_end = std::chrono::high_resolution_clock::now();
    eval_duration_count = std::chrono::duration<double>(eval_end - eval_start).count();
    std::cout << "Evaluation time: " << eval_duration_count << " seconds\n";
    std::cout << "Model evaluation score (e.g. MSE): " << score << "\n";
}

// Cas pour Bagging et Boosting : evaluate() retourne directement un double
template <typename EnsembleModel>
typename std::enable_if<!std::is_same<EnsembleModel, NeuralNetwork>::value>::type
trainAndEvaluateModel(EnsembleModel& model,
                      const std::vector<double>& X_train,
                      int rowLength,
                      const std::vector<double>& y_train,
                      const std::vector<double>& X_test,
                      const std::vector<double>& y_test,
                      double& score,
                      double& train_duration_count,
                      double& eval_duration_count) {
    std::cout << "Starting ensemble model training...\n";

    auto train_start = std::chrono::high_resolution_clock::now();
    model.train(X_train, rowLength, y_train);
    auto train_end = std::chrono::high_resolution_clock::now();
    train_duration_count = std::chrono::duration<double>(train_end - train_start).count();
    std::cout << "Training time: " << train_duration_count << " seconds\n";

    auto eval_start = std::chrono::high_resolution_clock::now();
    score = model.evaluate(X_test, rowLength, y_test);
    auto eval_end = std::chrono::high_resolution_clock::now();
    eval_duration_count = std::chrono::duration<double>(eval_end - eval_start).count();
    std::cout << "Evaluation time: " << eval_duration_count << " seconds\n";
    std::cout << "Model evaluation score (e.g. MSE): " << score << "\n";
}

#endif