#include <iostream>
#include <cstdlib>
#include <vector>

#include "net.h"

using namespace std;

int main()
{
    srand(69);
    // test layer
    // set inputs

    Layer l1 = Layer(2, 2);
    Layer l2 = Layer(1, 2);

    vector<double> inputs = {-14, 27};
    l1.forwardPass(inputs);
    vector<double> outsL1 = l1.outputs;
    l2.forwardPass(outsL1);
    vector<double> outsL2 = l2.outputs;
    printf("outputs: %f\n", outsL2.at(0));
    cout << "paon" << endl;
    // backpass
    l2.backPass({1});

    cout << "DONE L2 BACKPASS" << endl;
    // go thru each neuron in previous layer
    vector<double> toBack = vector<double>(l2.neurons.at(0).weights.size());
    for (int i = 0; i < l2.neurons.size(); i++)
    {
        cout << "BACKING THRU L1" << endl;

        for (int j = 0; j < l2.neurons.at(i).weightGrads.size(); j++)
        {
            toBack.at(j) = l2.neurons.at(i).output;
            toBack.at(j) /= l2.neurons.at(i).weightGrads.at(j);

            if (l2.neurons.at(i).inputs.at(j) == 0)
            {
                toBack.at(j) = 0;
            }
        }

        // l1.backPass(l2.neurons.at(i).weightGrads);
    }
    l1.backPass(toBack);

    //  print neuron data
    //  cout << "neuron 1 data: " << endl;
    cout << "\nneuron 0 \ndata: " << endl;
    cout << "l1 weight: " << l1.neurons.at(0).weights.at(0) << " " << l1.neurons.at(0).weights.at(1) << endl;
    cout << "l2 weight: " << l2.neurons.at(0).weights.at(0) << " " << l2.neurons.at(0).weights.at(1) << endl;
    // print gradients
    cout << "l1 grad: " << l1.neurons.at(0).weightGrads.at(0) << " " << l1.neurons.at(0).weightGrads.at(1) << endl;
    cout << "l2 grad: " << l2.neurons.at(0).weightGrads.at(0) << " " << l2.neurons.at(0).weightGrads.at(1) << endl;

    cout << "bias L1: " << l1.neurons.at(0).bias << endl;
    cout << "bias L2: " << l2.neurons.at(0).bias << endl;

    cout << "\nneuron 1 \ndata: " << endl;
    cout << "l1 weight: " << l1.neurons.at(1).weights.at(0) << " " << l1.neurons.at(1).weights.at(1) << endl;
    cout << "bias L1: " << l1.neurons.at(1).bias << endl;
    // print gradients
    cout << "l1 grad: " << l1.neurons.at(1).weightGrads.at(0) << " " << l1.neurons.at(1).weightGrads.at(1) << endl;

    /*
    for (int i = 0; i < 2; i++)
    {
        cout << "\nneuron " << i << " \ndata: " << endl;
        // print weights
        for (int j = 0; j < 2; j++)
        {
            cout << "l1 weight:" << l1.neurons.at(i).weights.at(j) << " ";
            if (j == 0)
            {
                cout << "l2 weight:" << l2.neurons.at(i).weights.at(j) << " \n";
                cout << "bias L2: " << l2.neurons.at(i).bias << " ";
            }
            cout << "bias L1: " << l1.neurons.at(i).bias << " ";
        }
    }*/
    // test backward pass

    /*
    // set inputs
    vector<int> inputsPerLayer = {2, 2, 2};
    // set neurons per layer
    vector<int> neuronsPerLayer = {2, 2, 1};

    Net test = Net(inputsPerLayer, neuronsPerLayer, 0.05);
    // // testing net
    vector<double> outs = test.forwardPass({1, 9});
    cout << "outputs: ";
    for (int i = 0; i < outs.size(); i++)
    {
        cout << outs.at(i) << " ";
    }
    */
    cout << "done!" << endl;

    return 0;
}