// #include <vector>
// #include <string>
// #include <iostream>
// #include <cstdlib>

// #include "neuron.h"

// using namespace std;

// #pragma once
// #ifndef LAYER_H
// #define LAYER_H

// class Layer
// {
// public:
//     int numNeurons;
//     vector<Neuron> neurons;
//     vector<double> outputs;
//     vector<double> inputs;
//     // instantiate
//     Layer(int numNeurons, int inputsToNeuron);
//     // forward pass
//     void forwardPass(vector<double> inputs);
//     // backwards pass
//     void backPass(vector<double> gradNext);

//     // zero grad
//     void zeroGrad();

//     // update weights
//     void update(double learningRate);

//     // move constructor
//     Layer(Layer &&other);
// };

// #endif