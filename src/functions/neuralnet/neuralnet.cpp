#include "neuralnet.h"
#include "../math/math_functions.h"
#include <random>
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>

NeuralNetwork::NeuralNetwork(int input_size, int output_size, int hidden_size, double learning_rate)
    : input_size(input_size), output_size(output_size), hidden_size(hidden_size), learning_rate(learning_rate) {
    init_weights(W1, input_size, hidden_size);
    b1.assign(hidden_size, 0.0);
    init_weights(W2, hidden_size, hidden_size);
    b2.assign(hidden_size, 0.0);
    init_weights(W3, hidden_size, output_size);
    b3.assign(output_size, 0.0);
}

void NeuralNetwork::init_weights(std::vector<double>& W, int in_size, int out_size) {
    std::default_random_engine gen;
    std::normal_distribution<double> dist(0.0, 0.01);
    W.resize(in_size * out_size);
    for (auto& w : W)
        w = dist(gen);
}

std::vector<double> NeuralNetwork::matvec(const std::vector<double>& W, const std::vector<double>& x,
                                         const std::vector<double>& b, int in_size, int out_size) {
    std::vector<double> result(out_size, 0.0);
    for (int j = 0; j < out_size; ++j) {
        for (int i = 0; i < in_size; ++i) {
            result[j] += W[i * out_size + j] * x[i];
        }
        result[j] += b[j];
    }
    return result;
}

std::vector<double> NeuralNetwork::matvecT(const std::vector<double>& W, const std::vector<double>& v, int in_size, int out_size) {
    std::vector<double> result(in_size, 0.0);
    for (int i = 0; i < in_size; ++i) {
        for (int j = 0; j < out_size; ++j) {
            result[i] += W[i * out_size + j] * v[j];
        }
    }
    return result;
}

std::vector<double> NeuralNetwork::forward(const std::vector<double>& x) {
    A1 = relu(matvec(W1, x, b1, input_size, hidden_size));
    A2 = relu(matvec(W2, A1, b2, hidden_size, hidden_size));
    Z3 = matvec(W3, A2, b3, hidden_size, output_size);
    double output = 1.0 / (1.0 + std::exp(-Z3[0]));  // sigmoid
    if (std::isnan(output) || std::isinf(output)) {
        std::cerr << "Warning: output is NaN or Inf. Z3[0] = " << Z3[0] << std::endl;
    }
    return {output};
}

void NeuralNetwork::backward(const std::vector<double>& x, const std::vector<double>& y_true, const std::vector<double>& y_pred) {
    double loss_mse = Math::computeLossMSE(y_true, y_pred);
    double loss_mae = Math::computeLossMAE(y_true, y_pred);
    std::cout << "Loss MSE: " << loss_mse << ", Loss MAE: " << loss_mae << std::endl;
    
    std::vector<double> dZ3 = Math::calculateMSEderivative(y_true, y_pred);
    std::vector<double> dW3 = outer_product(A2, dZ3);
    std::vector<double> db3 = dZ3;
    std::vector<double> dA2 = matvecT(W3, dZ3, hidden_size, output_size);

    std::vector<double> dZ2 = relu_derivative(A2, dA2);
    std::vector<double> dW2 = outer_product(A1, dZ2);
    std::vector<double> db2 = dZ2;
    std::vector<double> dA1 = matvecT(W2, dZ2, hidden_size, hidden_size);

    std::vector<double> dZ1 = relu_derivative(A1, dA1);
    std::vector<double> dW1 = outer_product(x, dZ1);
    std::vector<double> db1 = dZ1;

    update_weights(W3, b3, dW3, db3);
    update_weights(W2, b2, dW2, db2);
    update_weights(W1, b1, dW1, db1);
}

void NeuralNetwork::train(const std::vector<double>& X, int rowLength, const std::vector<double>& y) {
    int numSamples = y.size();
    std::vector<double> mse_history;
    std::vector<double> mae_history;

    const int window_size = 10;              // taille de la fenêtre glissante
    const double min_improvement = 1e-5;     // seuil de stagnation
    const int patience = 5;                  // nombre d'epochs consécutifs sans amélioration
    int stagnation_counter = 0;
    double last_avg_mse = std::numeric_limits<double>::max();

    for (int epoch = 0; epoch < epochs; ++epoch) {
        mse_history.clear();
        mae_history.clear();

        for (int i = 0; i < numSamples; ++i) {
            std::vector<double> input(X.begin() + i * rowLength, X.begin() + (i + 1) * rowLength);
            std::vector<double> target = { y[i] };

            std::vector<double> output = forward(input);

            if (std::isnan(output[0])) {
                std::cerr << "Warning: output[0] is NaN at sample " << i << std::endl;
                std::cerr << "Input: ";
                for (double v : input) std::cerr << v << " ";
                std::cerr << std::endl;
            }
            if (std::isnan(target[0])) {
                std::cerr << "Warning: target[0] is NaN at sample " << i << std::endl;
            }

            backward(input, target, output);

            double loss_mse = Math::computeLossMSE(target, output);
            double loss_mae = Math::computeLossMAE(target, output);
            mse_history.push_back(loss_mse);
            mae_history.push_back(loss_mae);
        }

        // Calcul des moyennes glissantes après chaque epoch
        double avg_mse = Math::movingAverage(mse_history, window_size);
        double avg_mae = Math::movingAverage(mae_history, window_size);
        std::cout << "[Epoch " << epoch << "] Moving Avg MSE: " << avg_mse
                  << ", Moving Avg MAE: " << avg_mae << std::endl;

        // Critère d'arrêt automatique
        double improvement = last_avg_mse - avg_mse;
        if (improvement < min_improvement) {
            ++stagnation_counter;
            std::cout << "No significant improvement (" << stagnation_counter << "/" << patience << ")\n";
            if (stagnation_counter >= patience) {
                std::cout << "Early stopping triggered. Training stopped.\n";
                break;
            }
        } else {
            stagnation_counter = 0;
        }

        last_avg_mse = avg_mse;
    }
}

std::vector<double> NeuralNetwork::relu(const std::vector<double>& x) {
    std::vector<double> result = x;
    for (double& val : result)
        val = std::max(0.0, val);
    return result;
}

std::vector<double> NeuralNetwork::relu_derivative(const std::vector<double>& act, const std::vector<double>& grad) {
    std::vector<double> result(act.size());
    for (size_t i = 0; i < act.size(); ++i)
        result[i] = (act[i] > 0) ? grad[i] : 0.0;
    return result;
}

std::vector<double> NeuralNetwork::softmax(const std::vector<double>& x) {
    double max_elem = *std::max_element(x.begin(), x.end());
    std::vector<double> exp_x(x.size());
    double sum = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        exp_x[i] = std::exp(x[i] - max_elem);
        sum += exp_x[i];
        if (sum == 0.0) sum = 1e-8;
    }
    for (double& val : exp_x)
        val /= sum;
    return exp_x;
}

std::vector<double> NeuralNetwork::outer_product(const std::vector<double>& a, const std::vector<double>& b) {
    std::vector<double> result(a.size() * b.size());
    for (size_t i = 0; i < a.size(); ++i)
        for (size_t j = 0; j < b.size(); ++j)
            result[i * b.size() + j] = a[i] * b[j];
    return result;
}

void NeuralNetwork::update_weights(std::vector<double>& W, std::vector<double>& b,
                                   const std::vector<double>& dW, const std::vector<double>& db) {
    if (W.size() != dW.size()) {
        std::cerr << "Dimension mismatch: W.size() = " << W.size() << ", dW.size() = " << dW.size() << "\n";
        return;
    }
    if (b.size() != db.size()) {
        std::cerr << "Dimension mismatch: b.size() = " << b.size() << ", db.size() = " << db.size() << "\n";
        return;
    }
    for (size_t i = 0; i < W.size(); ++i)
        W[i] -= learning_rate * dW[i];
    for (size_t i = 0; i < b.size(); ++i)
        b[i] -= learning_rate * db[i];
}

std::vector<double> NeuralNetwork::evaluate(const std::vector<double>& X, int rowLength, const std::vector<double>& y) {
    int numSamples = y.size();
    std::vector<double> predictions;

    for (int i = 0; i < numSamples; ++i) {
        std::vector<double> input(X.begin() + i * rowLength, X.begin() + (i + 1) * rowLength);
        std::vector<double> output = forward(input);

        predictions.push_back(output[0]);  // suppose un seul neurone de sortie
    }

    return predictions;
}

void NeuralNetwork::save(const std::string& path) const {
    std::ofstream out(path);
    if (!out.is_open()) {
        std::cerr << "Failed to open file for saving: " << path << std::endl;
        return;
    }

    out << "InputSize=" << input_size << "\n";
    out << "OutputSize=" << output_size << "\n";
    out << "HiddenSize=" << hidden_size << "\n";
    out << "LearningRate=" << learning_rate << "\n";

    out << "W1=";
    for (double w : W1) out << w << " ";
    out << "\n";

    out << "W2=";
    for (double w : W2) out << w << " ";
    out << "\n";

    out << "W3=";
    for (double w : W3) out << w << " ";
    out << "\n";

    out << "b1=";
    for (double b : b1) out << b << " ";
    out << "\n";

    out << "b2=";
    for (double b : b2) out << b << " ";
    out << "\n";

    out << "b3=";
    for (double b : b3) out << b << " ";
    out << "\n";

    out.close();
    std::cout << "Model saved to: " << path << "\n";
}
