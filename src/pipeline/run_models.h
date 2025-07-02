#ifndef RUN_MODELS_H
#define RUN_MODELS_H

#include "model_params.h"
#include "../functions/neuralnet/neuralnet.h"
#include "data_split.h"

// Prototypes des fonctions d'entraînement
void runSingleNeuralNetwork(NeuralNetworkParams params, DataParams data_params);
void runBaggingNeuralNetwork(BaggingParams params, DataParams data_params);
void runBoostingNeuralNetwork(BoostingParams params, DataParams data_params);

#endif
