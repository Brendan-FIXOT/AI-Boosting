#include "bagging.h"
#include <future>
#include <random>
#include <numeric>
#include <fstream>
#include <sstream>
#include <iostream>

Bagging::Bagging(int num_models, int input_size, int output_size, int hidden_size, double learning_rate) {
  
  loss_function = std::make_unique<LeastSquaresLoss>();
  
  for (int i = 0; i < num_models; ++i) {
      models.emplace_back(std::make_unique<NeuralNetwork>(input_size, output_size, hidden_size, learning_rate));
  }
}

void Bagging::bootstrapSample(const std::vector<double> &data, int rowLength,
                              const std::vector<double> &labels,
                              std::vector<double> &sampled_data,
                              std::vector<double> &sampled_labels) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, labels.size() - 1);

  size_t n_samples = labels.size();
  sampled_data.reserve(n_samples * rowLength);
  sampled_labels.reserve(n_samples);

  for (size_t i = 0; i < n_samples; ++i) {
    int idx = dis(gen);
    sampled_labels.push_back(labels[idx]);
    sampled_data.insert(sampled_data.end(), data.begin() + idx * rowLength,
                        data.begin() + (idx + 1) * rowLength);
  }
}

void Bagging::train(const std::vector<double>& data, int rowLength,
                    const std::vector<double>& labels) {
    if (numThreads <= 1) {
        for (int i = 0; i < numModels; ++i) {
            std::vector<double> sampled_data;
            std::vector<double> sampled_labels;
            bootstrapSample(data, rowLength, labels, sampled_data, sampled_labels);

            auto model = std::make_unique<NeuralNetwork>(rowLength, 1, 32, learningRate);
            model->train(sampled_data, rowLength, sampled_labels);  // ✅ entraînement propre
            models.push_back(std::move(model));
        }
    } else {
        std::vector<std::future<std::unique_ptr<NeuralNetwork>>> futures;

        for (int i = 0; i < numModels; ++i) {
            futures.push_back(std::async(std::launch::async, [this, &data, rowLength, &labels]() {
                std::vector<double> sampled_data;
                std::vector<double> sampled_labels;
                bootstrapSample(data, rowLength, labels, sampled_data, sampled_labels);

                auto model = std::make_unique<NeuralNetwork>(rowLength, 1, 32, learningRate);
                model->train(sampled_data, rowLength, sampled_labels);  // ✅ entraînement propre
                return model;
            }));

            // Récupération des modèles si nombre de threads atteint
            if (futures.size() >= static_cast<size_t>(numThreads)) {
                for (auto& future : futures) {
                    models.push_back(std::move(future.get()));
                }
                futures.clear();
            }
        }

        // Récupération finale
        for (auto& future : futures) {
            models.push_back(std::move(future.get()));
        }
    }
}

double Bagging::predict(const std::vector<double> &sample) const {
  double sum = 0.0;
  for (const auto &model : models) {
    sum += model->forward(sample)[0];
  }
  return sum / models.size();
}

double Bagging::evaluate(const std::vector<double> &test_data, int rowLength,
                         const std::vector<double> &test_labels) const {
  std::vector<double> predictions;
  size_t n_samples = test_labels.size();

  for (size_t i = 0; i < n_samples; ++i) {
    std::vector<double> sample(test_data.begin() + i * rowLength,
                               test_data.begin() + (i + 1) * rowLength);
    predictions.push_back(predict(sample));
  }
  return loss_function->computeLoss(test_labels, predictions);
}

std::map<std::string, std::string> Bagging::getTrainingParameters() const {
  std::map<std::string, std::string> parameters;
  parameters["NumModels"] = std::to_string(numModels);
  parameters["LearningRate"] = std::to_string(learningRate);
  parameters["NumThreads"] = std::to_string(numThreads);
  return parameters;
}

std::string Bagging::getTrainingParametersString() const {
  std::ostringstream oss;
  oss << "Training Parameters:\n";
  oss << "  - Number of Models: " << numModels << "\n";
  oss << "  - Learning Rate: " << learningRate << "\n";
  oss << "  - Number of Threads: " << numThreads << "\n";
  return oss.str();
}