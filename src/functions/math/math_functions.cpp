#include "math_functions.h"


/**
 * Calculate the mean of the samples
 */
double Math::calculateMean(const std::vector<double> &labels) 
{
    if (labels.empty())
    {
        return 0.0; // Return 0 if labels are empty, to prevent undefined behavior
    }
    double sum = std::accumulate(labels.begin(), labels.end(),0);
    return sum / labels.size();
}

/**
 * Calculate the mean on a windows size of values
 */
double Math::movingAverage(const std::vector<double>& values, int window_size) {
    if (values.empty() || window_size <= 0) return 0.0;

    int start = std::max(0, static_cast<int>(values.size()) - window_size);
    double sum = 0.0;
    for (int i = start; i < values.size(); ++i)
        sum += values[i];
    return sum / (values.size() - start);
}

//Takes also mean as parameter for optimization in data_clean.cpp
double Math::calculateStdDev(const std::vector<double>& data, double mean) {
        double sum = 0.0;
    for (const auto& value : data) {
        sum += std::pow(value - mean, 2);
    }
    return std::sqrt(sum / data.size());
}

/**
 * Calculate the Mean Squared Error (MSE)
 */
double Math::calculateMSE(const std::vector<double> &labels)
{
    if (labels.empty())
    {
        return 0.0; // Return 0 to handle empty label case, preventing division by zero
    }
    double mean = calculateMean(labels);
    double mse = 0.0;
    for (double value : labels)
        mse += std::pow(value - mean, 2);
    return mse / labels.size();
}

std::vector<double> Math::calculateMSEderivative(const std::vector<double>& y_true, const std::vector<double>& y_pred) {
    std::vector<double> grad(y_true.size());
    for (size_t i = 0; i < y_true.size(); ++i) {
        grad[i] = 2.0 * (y_pred[i] - y_true[i]);
    }
    return grad;
}

std::vector<double> Math::calculateMAEderivative(const std::vector<double>& y_true, const std::vector<double>& y_pred) {
    std::vector<double> grad(y_true.size());
    for (size_t i = 0; i < y_true.size(); ++i) {
        grad[i] = (y_pred[i] > y_true[i]) ? 1.0 : (y_pred[i] < y_true[i]) ? -1.0 : 0.0;
    }
    return grad;
}


double Math::calculateMedian(const std::vector<double>& values) {
    if (values.empty()) {
        throw std::invalid_argument("Cannot compute median of an empty set.");
    }

    IncrementalMedian medianTracker;
    for (double val : values) {
        medianTracker.insert(val);
    }
    return medianTracker.getMedian();
}

double Math::incrementalMedian(std::vector<double>& sortedValues, size_t size) {
    if (size == 0) {
        throw std::invalid_argument("Cannot compute median of an empty subset.");
    }
    
    if (size % 2 == 1) { // Odd size
        return sortedValues[size / 2];
    } else { // Even size
        return (sortedValues[size / 2 - 1] + sortedValues[size / 2]) / 2.0;
    }
}



void Math::IncrementalMedian::insert(double value){
    if (leftMaxHeap.empty() || value <= leftMaxHeap.top()){
        leftMaxHeap.push(value);
    } else {
        rightMinHeap.push(value);
    }
    if (leftMaxHeap.size() > rightMinHeap.size()+ 1){
        rightMinHeap.push(leftMaxHeap.top());
        leftMaxHeap.pop();
        
    } else if(rightMinHeap.size() > leftMaxHeap.size()) {
        leftMaxHeap.push(rightMinHeap.top());
        rightMinHeap.pop();
    }
}

double Math::IncrementalMedian::getMedian() const {
    if (leftMaxHeap.size() > rightMinHeap.size()){
        return leftMaxHeap.top();
    } else {
        return (leftMaxHeap.top() + rightMinHeap.top())/2.0;
    }
}

double Math::calculateMedianSorted(const std::vector<double>& sortedValues) {
    size_t n = sortedValues.size();
    if (n % 2 == 0) {
        return (sortedValues[n / 2 - 1] + sortedValues[n / 2]) / 2.0;
    } else {
        return sortedValues[n / 2];
    }
}


double Math::calculateMAE(const std::vector<double>& values, double mean) {
    double error = 0.0;
    for (double value : values) {
        error += std::abs(value - mean);
    }
    return error / values.size();
}


// Loss functions
std::vector<double> Math::negativeGradient(const std::vector<double> &y_true,
                                           const std::vector<double> &y_pred) 
{
    std::vector<double> residuals(y_true.size());
    for (size_t i = 0; i < y_true.size(); ++i)
    {
        residuals[i] = y_true[i] - y_pred[i];
    }
    return residuals;
}

double Math::computeLossMSE(const std::vector<double> &y_true, const std::vector<double> &y_pred) 
{
    double loss = 0.0;
    for (size_t i = 0; i < y_true.size(); ++i)
    {
        loss += std::pow(y_true[i]-y_pred[i], 2);
    }
    return loss / y_true.size();
}

double Math::computeLossMAE(const std::vector<double> &y_true, const std::vector<double>& y_pred){
    double loss =0.0;
    for (size_t i = 0; i<y_true.size() ; ++i){
        loss+= std::abs(y_true[i]-y_pred[i]);
    }
    return loss /y_true.size();

}