#include "utility.h"

// For the single decision tree multithreading
int adjustNumThreads(int numThreads) {
  if (numThreads <= 0)
    return 1;
  if ((numThreads & (numThreads - 1)) == 0)
    return numThreads;

  int power = 1;
  while (power * 2 <= numThreads)
    power *= 2;

  return power;
}

// input function to set parameters with defaults
template <typename T>
T getInputWithDefault(const std::string &prompt, T defaultValue) {
  std::cout << prompt << " (Default: " << defaultValue << "): ";
  std::string input;
  std::getline(std::cin, input); // Read user input as string

  // If empty return default
  if (input.empty()) {
    return defaultValue;
  }

  std::istringstream iss(input);
  T value;
  iss >> value;

  if (iss.fail()) {
    std::cerr << "Invalid input. Using default value: " << defaultValue << "\n";
    return defaultValue;
  }
  return value;
}

ProgramOptions parseCommandLineArguments(int argc, char *argv[]) {
  ProgramOptions options;

  if (argc > 1) {
    options.choice = std::stoi(argv[1]);

    int start_index = 2; // Start after choice argument
    if (argc > 2) {
      if (std::string(argv[2]) == "-p") {
        options.use_custom_params = true;
        start_index = 3;
      } else if (std::string(argv[2]) == "-l") {
        options.load_request = true;
        options.path_model_filename = argv[3];
      }
    }
    for (int i = start_index; i < argc; i++) {
      options.params.push_back(argv[i]);
    }
  } else {
    std::cout << "Choose the method you want to use:\n"
              << "1: Single Neural Network\n"
              << "2: Bagging Neural Network\n"
              << "3: Boosting Neural Network\n";
    std::cin >> options.choice;
  }
  return options;
}

void createDirectory(const std::string &path) {
  if (!std::filesystem::exists(path)) {
    std::filesystem::create_directories(path);
    std::cout << "Directory created: " << path << std::endl;
  }
}
