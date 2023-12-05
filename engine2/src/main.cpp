#include <iostream>
#include <cstdlib>
#include <vector>
#include <cmath>

#include "net.h"
#include "mnistData.h"

using namespace std;

vector<double> reluback(Layer *nextlayer)
{
    vector<double> toBack = vector<double>(nextlayer->neurons.at(0).weights.size());

    // first we build a single vector of all the gradients combined
    for (int n = 0; n < nextlayer->inputs.size(); n++)
    {
        for (int i = 0; i < nextlayer->neurons.size(); i++)
        {
            // we have to be careful of division by 0
            double to_divide = (nextlayer->inputs.at(n));
            // if (to_divide == 0)
            // {
            //     to_divide = 0.00000000001;
            // }
            toBack.at(n) += nextlayer->neurons.at(i).weights.at(n) * nextlayer->neurons.at(i).weightGrads.at(n) / to_divide;
        }
    }

    // then we check if the input was negative
    // and change it to a much smaller because relu
    for (int n = 0; n < nextlayer->inputs.size(); n++)
    {
        if (nextlayer->inputs.at(n) <= 0)
        {
            // we make it a much smaller number
            // for non linear activation
            toBack.at(n) *= 0.1;
        }
    }

    return toBack;
}

vector<double> sigmoidback(Layer *nextlayer)
{
    vector<double> toBack = vector<double>(nextlayer->neurons.at(0).weights.size());

    // first we build a single vector of all the gradients combined
    for (int n = 0; n < nextlayer->inputs.size(); n++)
    {
        for (int i = 0; i < nextlayer->neurons.size(); i++)
        {
            // we have to be careful of division by 0
            double to_divide = (nextlayer->inputs.at(n));
            // if (to_divide == 0)
            // {
            //     to_divide = 0.00000000001;
            // }
            toBack.at(n) += nextlayer->neurons.at(i).weights.at(n) * nextlayer->neurons.at(i).weightGrads.at(n) / to_divide;
        }
    }

    // then we calculate the sigmoid of the function
    for (int n = 0; n < nextlayer->inputs.size(); n++)
    {
        if (nextlayer->inputs.at(n) <= 0)
        {
            // we make it a much smaller number
            // for non linear activation
            toBack.at(n) = toBack.at(n) * (1 - exp(-toBack.at(n)));
        }
    }

    return toBack;
}

int main()
{
    srand(time(NULL));

    // keep track of number of accurate guesses
    double numAccurate = 0;
    // vector of accuratcy
    vector<double> accuracyArr;

    // keep track of loss in vector
    vector<double> lossArr;
    // keep track of loss increments
    vector<double> lossIncr;

    // keep track of loss in test images
    vector<double> lossArrTest;
    // keep track of loss increments for test
    vector<double> lossIncrTest;

    // average of test
    vector<double> lossTestAvg;

    // test layer
    // set inputs
    // training images
    vector<vector<double>>
        mnist_train_images;
    vector<int> mnist_train_labels;

    // testing images
    vector<vector<double>> mnist_test_images;
    vector<int> mnist_test_labels;

    // load training data
    loadData(mnist_train_images, mnist_train_labels, "mnist_train.csv");
    // load testing data
    loadData(mnist_test_images, mnist_test_labels, "mnist_test.csv");

    // Layer l1 = Layer(100, 784);
    // Layer l2 = Layer(40, 100);
    // Layer l3 = Layer(10, 40);
    // Layer l4 = Layer(1, 10);
    Layer l1 = Layer(100, 784);
    Layer l2 = Layer(10, 100);
    Layer l3 = Layer(1, 10);

    vector<double> toBack;
    // learning rate
    double learningRate = 0.001;
    for (int epoch = 0; epoch < 40; epoch++)
    {
        // gradually decrease learning rate to prevent overshooting
        // if (epoch < 5)
        // {
        //    learningRate *= .75;
        //}
        double runningLoss = 0;
        for (int i = 0; i < mnist_train_images.size(); i++)
        {
            // load data
            vector<double> inputs = mnist_train_images.at(i);
            int expected = mnist_train_labels.at(i);

            // forward pass
            l1.forwardPass(inputs);
            vector<double> outsL1 = l1.outputs;

            l2.forwardPass(outsL1);
            vector<double> outsL2 = l2.outputs;

            l3.forwardPass(outsL2);
            vector<double> outsL3 = l3.outputs;

            // l4.forwardPass(outsL3);
            // vector<double> outsL4 = l4.outputs;

            // calculate loss
            // squared error loss
            // double loss = pow((outsL4.at(0) - expected), 2);
            double loss = pow((outsL3.at(0) - expected), 2);
            runningLoss += loss;

            if (i % 1001 == 1)
            {
                cout << "epoch: " << epoch << ", i: " << i << ", Current Loss: " << loss << ", Running Loss: " << (runningLoss / i) << "          \r" << flush;
                ;
            }

            // now we need to find the gradient of the loss and pass it back
            // to the previous layer
            // double lossGrad = 2 * (outsL4.at(0) - expected);
            double lossGrad = 2 * (outsL3.at(0) - expected);

            // zero before backpass
            l1.zeroGrad();
            l2.zeroGrad();
            l3.zeroGrad();
            // l4.zeroGrad();

            // l4.backPass({lossGrad});
            // toBack = reluback(&l4);
            // l3.backPass(toBack);
            l3.backPass({lossGrad});
            toBack = reluback(&l3);
            l2.backPass(toBack);
            toBack = reluback(&l2);
            l1.backPass(toBack);

            // now we just need to update the weights
            l1.update(learningRate);
            l2.update(learningRate);
            l3.update(learningRate);
            // l4.update(learningRate);
        }
        // add loss to vector lossArr
        lossArr.push_back((runningLoss / mnist_train_images.size()));
        lossIncr.push_back(epoch);
        cout << endl;

        // Check efficiency on test data every 5 epochs
        if (epoch % 2 == 0)
        {
            // test accuracy on test data
            numAccurate = 0;
            double runningLoss = 0;
            for (int i = 0; i < mnist_test_images.size(); i++)
            {
                // load data
                vector<double> inputs = mnist_test_images.at(i);
                int expected = mnist_test_labels.at(i);

                // forward pass
                l1.forwardPass(inputs);
                vector<double> outsL1 = l1.outputs;

                l2.forwardPass(outsL1);
                vector<double> outsL2 = l2.outputs;

                l3.forwardPass(outsL2);
                vector<double> outsL3 = l3.outputs;

                // l4.forwardPass(outsL3);
                //  vector<double> outsL4 = l4.outputs;
                //  see if guess is correct
                double guess = round(outsL3.at(0));
                if (guess == expected)
                {
                    numAccurate++;
                }
                // actual label and estimated label
                // cout << "expected: " << expected << ", estimated: " << outsL4.at(0) << endl;

                // calculate loss
                // squared error loss
                double lossTest = pow((outsL3.at(0) - expected), 2);
                runningLoss += lossTest;
                //  cout running loss
                // cout << "\nTest Data Current Loss : " << lossTest << ", Running Loss : " << (runningLoss / i) << "          \r" << flush;
                ;
                lossArrTest.push_back((lossTest));
            }
            accuracyArr.push_back((numAccurate / mnist_test_images.size()));
            // cout current accuracy
            cout << "\nCurrent Accuracy: " << (numAccurate / mnist_test_images.size()) << endl;
            // find average of test errors using lossArrTest
            double avg = 0;
            for (int i = 0; i < lossArrTest.size(); i++)
            {
                avg += lossArrTest.at(i);
            }
            avg /= lossArrTest.size();
            lossIncrTest.push_back(epoch);
            lossTestAvg.push_back(avg);
            lossArrTest.clear();
        }
    }
    // test accuracy on test data
    double runningLoss = 0;
    numAccurate = 0;
    for (int i = 0; i < mnist_test_images.size(); i++)
    {
        // load data
        vector<double> inputs = mnist_test_images.at(i);
        int expected = mnist_test_labels.at(i);

        // forward pass
        l1.forwardPass(inputs);
        vector<double> outsL1 = l1.outputs;

        l2.forwardPass(outsL1);
        vector<double> outsL2 = l2.outputs;

        l3.forwardPass(outsL2);
        vector<double> outsL3 = l3.outputs;

        // l4.forwardPass(outsL3);
        // vector<double> outsL4 = l4.outputs;

        // actual label and estimated label
        // cout << "expected: " << expected << ", estimated: " << outsL4.at(0) << endl;

        // calculate loss
        // squared error loss
        double loss = pow((outsL3.at(0) - expected), 2);
        // get guess
        double guess = round(outsL3.at(0));
        if (guess == expected)
        {
            numAccurate++;
        }
        //  cout running loss
        runningLoss += loss;
        cout << "\nFinal Test Data Current Loss : " << loss << ", Running Loss : " << (runningLoss / i) << "          \r" << flush;
        ;
    }
    // cout lossArr for graphing
    cout << "\nlossArr: \n";
    cout << "[";
    for (int i = 0; i < lossArr.size(); i++)
    {
        cout << lossArr.at(i) << ",";
    }
    // cout lossincr for graphing
    cout << "]";
    cout << "\nlossIncr: \n";
    cout << "[";
    for (int i = 0; i < lossIncr.size(); i++)
    {
        cout << lossIncr.at(i) << ",";
    }
    cout << "]";
    cout << "\nAverage of test loss mean squared eror: \n";
    cout << "[";
    for (int i = 0; i < lossTestAvg.size(); i++)
    {
        cout << lossTestAvg.at(i) << ",";
    }
    cout << "]";
    cout << "\nTest Loss increments: \n";
    cout << "[";
    for (int i = 0; i < lossIncrTest.size(); i++)
    {
        cout << lossIncrTest.at(i) << ",";
    }
    cout << "]\n";
    // cout accuracyArr
    cout << "\nAccuracyArr: \n";
    cout << "[";
    for (int i = 0; i < accuracyArr.size(); i++)
    {
        cout << accuracyArr.at(i) << ",";
    }
    cout << "]\n";
    // cout accuracy
    cout << "\nFinal Accuracy: " << (numAccurate / mnist_test_images.size()) << endl;

    // //  print neuron data
    // //  cout << "neuron 1 data: " << endl;
    // cout << "\nneuron 0 \ndata: " << endl;
    // cout << "l1 weight: " << l1.neurons.at(0).weights.at(0) << " " << l1.neurons.at(0).weights.at(1) << endl;
    // cout << "l2 weight: " << l2.neurons.at(0).weights.at(0) << " " << l2.neurons.at(0).weights.at(1) << endl;
    // // print gradients
    // cout << "l1 grad: " << l1.neurons.at(0).weightGrads.at(0) << " " << l1.neurons.at(0).weightGrads.at(1) << endl;
    // cout << "l2 grad: " << l2.neurons.at(0).weightGrads.at(0) << " " << l2.neurons.at(0).weightGrads.at(1) << endl;

    // cout << "bias L1: " << l1.neurons.at(0).bias << endl;
    // cout << "bias L2: " << l2.neurons.at(0).bias << endl;

    // cout << "\nneuron 1 \ndata: " << endl;
    // cout << "l1 weight: " << l1.neurons.at(1).weights.at(0) << " " << l1.neurons.at(1).weights.at(1) << endl;
    // cout << "bias L1: " << l1.neurons.at(1).bias << endl;
    // // print gradients
    // cout << "l1 grad: " << l1.neurons.at(1).weightGrads.at(0) << " " << l1.neurons.at(1).weightGrads.at(1) << endl;

    // /*
    // for (int i = 0; i < 2; i++)
    // {
    //     cout << "\nneuron " << i << " \ndata: " << endl;
    //     // print weights
    //     for (int j = 0; j < 2; j++)
    //     {
    //         cout << "l1 weight:" << l1.neurons.at(i).weights.at(j) << " ";
    //         if (j == 0)
    //         {
    //             cout << "l2 weight:" << l2.neurons.at(i).weights.at(j) << " \n";
    //             cout << "bias L2: " << l2.neurons.at(i).bias << " ";
    //         }
    //         cout << "bias L1: " << l1.neurons.at(i).bias << " ";
    //     }
    // }*/
    // // test backward pass

    // /*
    // // set inputs
    // vector<int> inputsPerLayer = {2, 2, 2};
    // // set neurons per layer
    // vector<int> neuronsPerLayer = {2, 2, 1};

    // Net test = Net(inputsPerLayer, neuronsPerLayer, 0.05);
    // // // testing net
    // vector<double> outs = test.forwardPass({1, 9});
    // cout << "outputs: ";
    // for (int i = 0; i < outs.size(); i++)
    // {
    //     cout << outs.at(i) << " ";
    // }
    // */
    cout << "done!\n"
         << endl;

    return 0;
}