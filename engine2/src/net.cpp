// #include <vector>
// #include <string>
// #include <iostream>
// #include <cstdlib>

// class Net
// {
// public:
//     vector<Layer> layers;
//     vector<double> finalOutputs;
//     double learnRate;

//     Net(vector<int> inputsPerLayer, vector<int> neuronsPerLayer, double learnRate)
//     {
//         // size of inputsPerLayer and neuronsPerLayer should be the same
//         for (int i = 0; i < inputsPerLayer.size(); i++)
//         {
//             // add Layer to list of layers
//             this->layers.push_back(Layer(neuronsPerLayer.at(i), inputsPerLayer.at(i)));
//         }
//         this->learnRate = learnRate;
//     }
//     // forward pass
//     vector<double> forwardPass(vector<double> inputs)
//     {
//         // set inputs to first layer
//         this->layers.at(0).forwardPass(inputs);
//         // for each layer
//         for (int i = 1; i < this->layers.size(); i++)
//         {
//             // set inputs to next layer
//             this->layers.at(i).forwardPass(this->layers.at(i - 1).outputs);
//         }
//         // set final outputs
//         this->finalOutputs = this->layers.at(this->layers.size() - 1).outputs;
//         // return outputs of last layer
//         return this->finalOutputs;
//     }
//     // backwards pass, pass in final output layer gradients
//     void backPass(vector<double> gradNext)
//     {
//         // for each layer
//         for (int i = this->layers.size() - 1; i >= 0; i--)
//         {
//             // set grads to next layer
//             this->layers.at(i).backPass(gradNext);
//             // update gradNext
//             gradNext = this->layers.at(i - 1).outputs;
//         }
//     }
//     // update values
//     void update()
//     {
//         // for each layer
//         for (int i = 0; i < this->layers.size(); i++)
//         {
//             // update weights
//             this->layers.at(i).update(this->learnRate);
//         }
//     }
//     // zero grad
//     void zeroGrad()
//     {
//         // for each layer
//         for (int i = 0; i < this->layers.size(); i++)
//         {
//             // zero grad
//             this->layers.at(i).zeroGrad();
//         }
//     }
// };
