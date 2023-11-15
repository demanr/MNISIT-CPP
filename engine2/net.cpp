#include <vector>
#include <string>
#include <iostream>
#include <cstdlib>
// #include "neuron.cpp"
#include "layer.cpp"

class Net
{
public:
    vector<Layer> layers;
    double learnRate;
    Net(vector<int> inputsPerLayer, vector<int> neuronsPerLayer, double learnRate)
    {
        // size of inputsPerLayer and neuronsPerLayer should be the same
        for (int i = 0; i < inputsPerLayer.size(); i++)
        {
            // add Layer to list of layers
            this->layers.push_back(Layer(neuronsPerLayer.at(i), inputsPerLayer.at(i)));
        }
        this->learnRate = learnRate;
    }
};