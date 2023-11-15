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
    vector<double> inputs;
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
        this->inputs = vector<double>(inputsToNeuron);
    }
    // forward pass
    void forwardPass(vector<double> inputs)
    {
        for (int i = 0; i < this->neurons.size(); i++)
        {
            this->inputs.at(i) = inputs.at(i);
            this->outputs.at(i) = this->neurons.at(i).calcOutput(inputs);
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
    // backwards pass
    void backPass(vector<double> gradNext)
    {
        for (int i = 0; i < this->neurons.size(); i++)
        {
            for (int j = 0; j < gradNext.size(); j++)
            {
                this->neurons.at(i).calcGrads(gradNext.at(j));
            }
        }
    }
    // update weights
    void update(double learningRate)
    {
        for (int i = 0; i < this->neurons.size(); i++)
        {
            this->neurons.at(i).update(learningRate);
        }
    }
};

int main()
{
    /*
    layer1.forwardPass(inputs);
    layer2.forwardPass(layer1.outputs);

    // print outputs of layers
    cout << "layer1 outputs: ";
    for (int i = 0; i < layer1.outputs.size(); i++)
    {
        cout << layer1.outputs.at(i) << " ";
    }
    cout << endl;
    // print output of layer 2
    cout << "layer2 outputs: ";
    for (int i = 0; i < layer2.outputs.size(); i++)
    {
        cout << layer2.outputs.at(i) << " ";
    }
    // do backpropagation*/
}