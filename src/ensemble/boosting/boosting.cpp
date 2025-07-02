#include "boosting.h"
#include <numeric>
#include <iostream>
#include <random>
#include <algorithm>
#include <stdexcept>
#include <fstream>

Boosting::Boosting(int num_models, int input_size, int output_size, int hidden_size, double learning_rate) {
    for (int i = 0; i < num_models; ++i) {
        models.emplace_back(std::make_unique<NeuralNetwork>(input_size, output_size, hidden_size, learning_rate));
    }
}

void Boosting::initializePrediction(const std::vector<double>& y) {
    initial_prediction = std::accumulate(y.begin(), y.end(), 0.0) / y.size();
}

void Boosting::train(const std::vector<double>& X, int rowLength,
                     const std::vector<double>& y) {
    if (X.empty() || y.empty()) {
        return;
    }

    size_t n_samples = y.size();
    initializePrediction(y);
    std::vector<double> y_pred(n_samples, initial_prediction);

    for (int i = 0; i < n_estimators; ++i) {
        std::vector<double> residuals = loss_function->negativeGradient(y, y_pred);

        auto model = std::make_unique<NeuralNetwork>(rowLength, 1, 32, learning_rate);

        for (int epoch = 0; epoch < 1; ++epoch) {
            for (size_t j = 0; j < n_samples; ++j) {
                std::vector<double> sample(X.begin() + j * rowLength, X.begin() + (j + 1) * rowLength);
                std::vector<double> y_true = { residuals[j] };
                std::vector<double> y_hat = model->forward(sample);
                model->backward(sample, y_true, y_hat);
            }
        }

        for (size_t j = 0; j < n_samples; ++j) {
            std::vector<double> sample(X.begin() + j * rowLength, X.begin() + (j + 1) * rowLength);
            y_pred[j] += learning_rate * model->forward(sample)[0];
        }

        models.push_back(std::move(model));

        double current_loss = loss_function->computeLoss(y, y_pred);
        std::cout << "Iteration " << i + 1 << ", Loss: " << current_loss << std::endl;
    }
}

double Boosting::predict(const std::vector<double>& x) const {
    double y_pred = initial_prediction;
    for (const auto& model : models) {
        y_pred += learning_rate * model->forward(x)[0];
    }
    return y_pred;
}

std::vector<double> Boosting::predict(const std::vector<double>& X, int rowLength) const {
    size_t n_samples = X.size() / rowLength;
    std::vector<double> y_pred(n_samples, initial_prediction);

    for (const auto& model : models) {
        for (size_t i = 0; i < n_samples; ++i) {
            std::vector<double> sample(X.begin() + i * rowLength, X.begin() + (i + 1) * rowLength);
            y_pred[i] += learning_rate * model->forward(sample)[0];
        }
    }
    return y_pred;
}

double Boosting::evaluate(const std::vector<double>& X_test, int rowLength, const std::vector<double>& y_test) const {
    std::vector<double> y_pred = predict(X_test, rowLength);
    return loss_function->computeLoss(y_test, y_pred);
}

void Boosting::save(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }

    file << n_estimators << " "
         << learning_rate << " "
         << initial_prediction << "\n";

    file.close();
    // Serialization des modèles non implémentée
}

void Boosting::load(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for reading: " + filename);
    }

    file >> n_estimators
         >> learning_rate
         >> initial_prediction;

    models.clear();
    models.resize(n_estimators);

    file.close();
    // Deserialization non implémentée
}

std::map<std::string, std::string> Boosting::getTrainingParameters() const {
    std::map<std::string, std::string> parameters;
    parameters["NumEstimators"] = std::to_string(n_estimators);
    parameters["LearningRate"] = std::to_string(learning_rate);
    parameters["InitialPrediction"] = std::to_string(initial_prediction);
    return parameters;
}

std::string Boosting::getTrainingParametersString() const {
    std::ostringstream oss;
    oss << "Training Parameters:\n";
    oss << "  - Number of Estimators: " << n_estimators << "\n";
    oss << "  - Learning Rate: " << learning_rate << "\n";
    oss << "  - Initial Prediction: " << initial_prediction << "\n";
    return oss.str();
}
