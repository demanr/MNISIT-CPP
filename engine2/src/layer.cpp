#include <vector>
#include <string>
#include <iostream>
#include <cstdlib>
#include "layer.h"

using namespace std;

Layer::Layer(int numNeurons, int inputsToNeuron)
{
    this->numNeurons = numNeurons;
    // this->neurons = vector<Neuron>(numNeurons);
    for (int i = 0; i < numNeurons; i++)
    {
        this->neurons.push_back(Neuron(inputsToNeuron));
    }
    // set sizes of outputs
    this->outputs = vector<double>(numNeurons);
    this->inputs = vector<double>(inputsToNeuron);
}

// instantiate

// forward pass
void Layer::forwardPass(vector<double> inputs)
{
    this->inputs = inputs;
    for (int i = 0; i < this->neurons.size(); i++)
    {
        this->outputs.at(i) = this->neurons.at(i).calcOutput(inputs);
    }
}

// zero grad
void Layer::zeroGrad()
{
    for (int i = 0; i < this->neurons.size(); i++)
    {
        this->neurons.at(i).zeroGrad();
    }
}

// backwards pass
void Layer::backPass(vector<double> gradNext)
{
    for (int i = 0; i < this->neurons.size(); i++)
    {

        for (int j = 0; j < gradNext.size(); j++)
        {
            this->neurons.at(i).calcGrads(gradNext.at(j));
        }
    }
}

// backwards pass with layer
void Layer::backPass(Layer &l2)
{
    for (int i = 0; i < l2.inputs.size(); i++)
    {
        for (int j = 0; j < l2.neurons.size(); j++)
        {
            double prevGrad = l2.neurons.at(j).weightGrads.at(i) / l2.inputs.at(i);
            if (l2.inputs.at(i) == 0)
            {
                prevGrad = 0;
            }
            this->neurons.at(i).calcGrads(prevGrad);
        }
    }
}

// update weights
void Layer::update(double learningRate)
{
    for (int i = 0; i < this->neurons.size(); i++)
    {
        this->neurons.at(i).update(learningRate);
    }
}

Layer::Layer(Layer &&other)
{
    this->neurons = std::move(other.neurons);
    this->outputs = std::move(other.outputs);
    this->inputs = std::move(other.inputs);
}

// class Layer
// {
// public:
//     int numNeurons;
//     vector<Neuron> neurons;
//     vector<double> outputs;
//     vector<double> inputs;
//     // instantiate
//     Layer(int numNeurons, int inputsToNeuron)
//     {
//         this->numNeurons = numNeurons;
//         // this->neurons = vector<Neuron>(numNeurons);
//         for (int i = 0; i < numNeurons; i++)
//         {
//             this->neurons.push_back(Neuron(inputsToNeuron));
//         }
//         // set sizes of outputs
//         this->outputs = vector<double>(numNeurons);
//         this->inputs = vector<double>(inputsToNeuron);
//     }
//     // forward pass
//     void forwardPass(vector<double> inputs)
//     {
//         for (int i = 0; i < this->neurons.size(); i++)
//         {
//             this->inputs.at(i) = inputs.at(i);
//             this->outputs.at(i) = this->neurons.at(i).calcOutput(inputs);
//         }
//     }

//     // zero grad
//     void zeroGrad()
//     {
//         for (int i = 0; i < this->neurons.size(); i++)
//         {
//             this->neurons.at(i).zeroGrad();
//         }
//     }

//     // backwards pass
//     void backPass(vector<double> gradNext)
//     {
//         for (int i = 0; i < this->neurons.size(); i++)
//         {
//             for (int j = 0; j < gradNext.size(); j++)
//             {
//                 this->neurons.at(i).calcGrads(gradNext.at(j));
//             }
//         }
//     }

//     // update weights
//     void update(double learningRate)
//     {
//         for (int i = 0; i < this->neurons.size(); i++)
//         {
//             this->neurons.at(i).update(learningRate);
//         }
//     }

//     // move constructor
//     Layer(Layer &&other)
//     {
//         this->neurons = std::move(other.neurons);
//         this->outputs = std::move(other.outputs);
//         this->inputs = std::move(other.inputs);
//     }
// };

// /*
// int main()
// {

//     layer1.forwardPass(inputs);
//     layer2.forwardPass(layer1.outputs);

//     // print outputs of layers
//     cout << "layer1 outputs: ";
//     for (int i = 0; i < layer1.outputs.size(); i++)
//     {
//         cout << layer1.outputs.at(i) << " ";
//     }
//     cout << endl;
//     // print output of layer 2
//     cout << "layer2 outputs: ";
//     for (int i = 0; i < layer2.outputs.size(); i++)
//     {
//         cout << layer2.outputs.at(i) << " ";
//     }
//     // do backpropagation
// }
// */
// /*
// #include <vector>
// #include <string>
// #include <iostream>
// #include <cstdlib>
// // #include "neuron.cpp"
// #include "neuron.h"

// using namespace std;

// class Layer
// {
// public:
//     int numNeurons;
//     vector<Neuron> neurons;
//     vector<double> outputs;
//     vector<double> inputs;
// };

/*
int main()
{

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
    // do backpropagation
}
*/