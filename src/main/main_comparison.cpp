#include "../model_comparison/model_comparison.h"
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>
#include <limits>
#include <thread>

void getModelParameters(int model_choice, std::string& parameters) {
    bool input = false;
    bool load_existing = false;
    std::cout << "Would you like to load an existing model? (1 = Yes (currently unused), 0 = No): ";
    std::cin >> load_existing;
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    if (load_existing) {
        parameters += " -l";
        std::string model_filename;

        switch (model_choice) {
            case 1:
                std::cout << "Enter the filename of the neural network model to load: ";
                std::cin >> model_filename;
                parameters += " ../saved_models/neural_networks/" + model_filename;
                return;
            case 2:
                std::cout << "Enter the filename of the bagging model to load: ";
                std::cin >> model_filename;
                parameters += " ../saved_models/bagging_models/" + model_filename;
                return;
            case 3:
                std::cout << "Enter the filename of the boosting model to load: ";
                std::cin >> model_filename;
                parameters += " ../saved_models/boosting_models/" + model_filename;
                return;
        }
    }

    std::cout << "\nDo you want to customize parameters? (1 = Yes, 0 = No): ";
    std::cin >> input;
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    if (!input) return;

    parameters += " -p";

    int input_size, output_size, hidden_size, num_models;
    double learning_rate;

    std::cout << "\nNeural Network Parameters:\n";
    std::cout << "Input size: ";
    std::cin >> input_size;
    std::cout << "Output size: ";
    std::cin >> output_size;
    std::cout << "Hidden layer size: ";
    std::cin >> hidden_size;
    std::cout << "Learning rate: ";
    std::cin >> learning_rate;
    std::cout << "Number of models (for ensemble): ";
    std::cin >> num_models;

    parameters += " " + std::to_string(input_size) +
                  " " + std::to_string(output_size) +
                  " " + std::to_string(hidden_size) +
                  " " + std::to_string(learning_rate) +
                  " " + std::to_string(num_models);
}

int main() {
    std::cout << "Neural Network Ensemble Program\n\n";

    int choice;
    std::cout << "Choose an option:\n";
    std::cout << "1. Run individual model\n";
    std::cout << "2. Run all tests\n";
    std::cout << "3. View models comparison\n";
    std::cin >> choice;

    switch (choice) {
        case 1: {
            std::cout << "\nChoose model to use:\n";
            std::cout << "1. Neural Network\n";
            std::cout << "2. Bagging Neural Networks\n";
            std::cout << "3. Boosting Neural Networks\n";

            int model_choice;
            std::cin >> model_choice;

            std::string parameters = std::to_string(model_choice);
            getModelParameters(model_choice, parameters);

            std::string command = "./MainNeuralNet " + parameters;
            std::cout << command << std::endl;
            system((command + " 2>&1").c_str());
            break;
        }
        case 2: {
            std::cout << "\nRunning all tests...\n\n";

            std::cout << "=== Math Functions Tests ===\n";
            system("./math_functions_test");

            std::cout << "\n=== Neural Network Tests ===\n";
            system("./neural_network_test");

            std::cout << "\n=== Bagging Tests ===\n";
            system("./bagging_test");

            std::cout << "\n=== Boosting Tests ===\n";
            system("./boosting_test");

            std::cout << "\n=== Cross Validation Tests ===\n";
            system("./cross_validation_test");

            std::cout << "\nAll tests completed.\n";
            break;
        }
        case 3: {
            std::cout << "\nDisplaying previous results...\n";
            std::ifstream file("../results/all_models_comparison.md");
            if (file.is_open()) {
                std::string line;
                while (std::getline(file, line)) {
                    std::cout << line << '\n';
                }
                file.close();
            } else {
                std::cout << "No previous results found. Please run tests first.\n";
            }
            break;
        }
        default:
            std::cout << "Invalid option\n";
            return 1;
    }

    return 0;
}
