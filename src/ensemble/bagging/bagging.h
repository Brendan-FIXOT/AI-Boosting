#ifndef BAGGING_H
#define BAGGING_H

#include <vector>
#include <memory>
#include <random>
#include <map>
#include <string>
#include <future>
#include <sstream>

#include "../../functions/neuralnet/neuralnet.h"
#include "../../functions/loss/loss_function.h"

class Bagging {
public:
    Bagging(int num_models, int input_size, int output_size, int hidden_size, double learning_rate);

    void train(const std::vector<double>& data, int rowLength,
               const std::vector<double>& labels);

    double predict(const std::vector<double>& sample) const;

    double evaluate(const std::vector<double>& test_data, int rowLength,
                    const std::vector<double>& test_labels) const;

    const std::vector<std::unique_ptr<NeuralNetwork>>& getModels() const { return models; }

    std::map<std::string, std::string> getTrainingParameters() const;
    std::string getTrainingParametersString() const;

private:
    int numModels;
    double learningRate;
    int numThreads;

    std::unique_ptr<LossFunction> loss_function;
    std::vector<std::unique_ptr<NeuralNetwork>> models;

    void bootstrapSample(const std::vector<double>& data, int rowLength,
                         const std::vector<double>& labels,
                         std::vector<double>& sampled_data,
                         std::vector<double>& sampled_labels);
};

#endif // BAGGING_H