#include <vector>
#include <string>
#include <iostream>
#include <cstdlib>
#include "neuron.cpp"

using namespace std;

class Layer
{
public:
    int numNeurons;
    vector<Neuron> neurons;
    vector<double> outputs;
    // instantiate
    Layer(int numNeurons, int inputsToNeuron)
    {
        this->numNeurons = numNeurons;
        this->neurons = vector<Neuron>(numNeurons);
        for (int i = 0; i < numNeurons; i++)
        {
            this->neurons.at(i) = Neuron(inputsToNeuron);
        }
        // set sizes of outputs
        this->outputs = vector<double>(numNeurons);
    }
    // forward pass
    void forwardPass(vector<double> prevOuts)
    {
        for (int i = 0; i < this->neurons.size(); i++)
        {
            this->outputs.at(i) = this->neurons.at(i).calcOutput(prevOuts);
        }
    }
    // zero grad
    void zeroGrad()
    {
        for (int i = 0; i < this->neurons.size(); i++)
        {
            this->neurons.at(i).zeroGrad();
        }
    }
};