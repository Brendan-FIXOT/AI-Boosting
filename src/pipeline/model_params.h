#pragma once

#include "../functions/neuralnet/neuralnet.h"
#include "../ensemble/bagging/bagging.h"
#include "../ensemble/boosting/boosting.h"
#include "../main/utility.h"
#include <string>
#include <iostream>
#include <filesystem>

// ===============================
// Paramètres pour un réseau de neurones simple
// ===============================

struct NeuralNetworkParams {
    int inputSize;
    int outputSize;
    int hiddenSize;
    double learningRate;
};

// ===============================
// Paramètres pour Bagging de réseaux de neurones
// ===============================

struct BaggingParams {
    int numModels;
    int inputSize;
    int outputSize;
    int hiddenSize;
    double learningRate;
};

// ===============================
// Paramètres pour Boosting de réseaux de neurones
// ===============================

struct BoostingParams {
    int numModels;
    int inputSize;
    int outputSize;
    int hiddenSize;
    double learningRate;
};

// ===============================
// Fonctions de récupération des paramètres
// ===============================

bool getNeuralNetworkParams(const ProgramOptions& options, NeuralNetworkParams& out_params);
bool getBaggingParams(const ProgramOptions& options, BaggingParams& out_params);
bool getBoostingParams(const ProgramOptions& options, BoostingParams& out_params);
