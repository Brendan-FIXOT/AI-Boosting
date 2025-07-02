#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>
#include <string>

class NeuralNetwork {
public:
    NeuralNetwork(int input_size, int output_size, int hidden_size = 64, double learning_rate = 0.01);

    std::vector<double> forward(const std::vector<double>& x);
    void backward(const std::vector<double>& x, const std::vector<double>& y_true, const std::vector<double>& y_pred);

    void train(const std::vector<double>& X, int rowLength, const std::vector<double>& y);
    std::vector<double> evaluate(const std::vector<double>& X, int rowLength, const std::vector<double>& y);

    void save(const std::string& path) const;

private:
    int input_size, output_size, hidden_size;
    double learning_rate;
    std::vector<double> W1, W2, W3;
    std::vector<double> b1, b2, b3;
    std::vector<double> A1, A2, Z3;
    int epochs = 100;

    void init_weights(std::vector<double>& W, int in_size, int out_size);
    std::vector<double> matvec(const std::vector<double>& W, const std::vector<double>& x,
                               const std::vector<double>& b, int in_size, int out_size);
    std::vector<double> matvecT(const std::vector<double>& W, const std::vector<double>& v,
                                int in_size, int out_size);

    std::vector<double> relu(const std::vector<double>& x);
    std::vector<double> relu_derivative(const std::vector<double>& act, const std::vector<double>& grad);
    std::vector<double> softmax(const std::vector<double>& x);

    std::vector<double> outer_product(const std::vector<double>& a, const std::vector<double>& b);

    void update_weights(std::vector<double>& W, std::vector<double>& b,
                        const std::vector<double>& dW, const std::vector<double>& db);
};

#endif // NEURAL_NETWORK_H
