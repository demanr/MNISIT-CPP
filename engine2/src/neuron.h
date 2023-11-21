
#include <vector>

using namespace std;

// void printVector(vector<double> v);

double ReLu(double num);

class Neuron
{
public:
    int inputAmount;
    double output;
    double bias;
    vector<double> weights;
    vector<double> inputs;
    // gradients
    vector<double> weightGrads;
    double biasGrad;

    // base constructor
    Neuron(int inputAmount);

    // calculate output forward pass
    double calcOutput(vector<double> inputs);

    // calculate gradients backward pass
    void calcGrads(double grad); // grad is gradient from neuron next

    void zeroGrad();
    void update(double learningRate);

    // move constructor
    Neuron(Neuron &&other);
};
