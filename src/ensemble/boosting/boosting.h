#ifndef BOOSTING_H
#define BOOSTING_H

#include "../../functions/neuralnet/neuralnet.h"
#include "../../functions/loss/loss_function.h"
#include <vector>
#include <memory>
#include <map>
#include <string>
#include <sstream>

/**
 * @brief Boosting class using neural networks as weak learners
 */
class Boosting {
public:
    Boosting(int num_models, int input_size, int output_size, int hidden_size, double learning_rate);

    void train(const std::vector<double>& X, int rowLength,
               const std::vector<double>& y);

    double predict(const std::vector<double>& x) const;

    std::vector<double> predict(const std::vector<double>& X,
                                int rowLength) const;

    double evaluate(const std::vector<double>& X_test, int rowLength,
                    const std::vector<double>& y_test) const;

    void save(const std::string& filename) const;
    void load(const std::string& filename);

    double getInitialPrediction() const { return initial_prediction; }

    std::map<std::string, std::string> getTrainingParameters() const;
    std::string getTrainingParametersString() const;

private:
    int n_estimators;
    double learning_rate;
    double initial_prediction;

    std::unique_ptr<LossFunction> loss_function;
    std::vector<std::unique_ptr<NeuralNetwork>> models;

    void initializePrediction(const std::vector<double>& y);
};

#endif // BOOSTING_H
