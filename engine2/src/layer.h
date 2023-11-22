#include <vector>
#include <string>
#include <iostream>
#include <cstdlib>

#include "neuron.h"

using namespace std;

class Layer
{
public:
    int numNeurons;
    vector<Neuron> neurons;
    vector<double> outputs;
    vector<double> inputs;
    // instantiate
    Layer(int numNeurons, int inputsToNeuron);
    // forward pass
    void forwardPass(vector<double> inputs);
    // backwards pass
    void backPass(vector<double> gradNext);
    void backPass(Layer &l2);

    // zero grad
    void zeroGrad();

    // update weights
    void update(double learningRate);

    // move constructor
    Layer(Layer &&other);
};
