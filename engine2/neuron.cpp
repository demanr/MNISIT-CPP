#include <vector>
#include <string>
#include <iostream>
#include <cstdlib>

using namespace std;

void printVector(vector<double> v)
{
    for (int i = 0; i < v.size(); i++)
    {
        cout << v[i] << ", ";
    }
    cout << endl;
}

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
    Neuron(int inputAmount)
    {
        this->weights = vector<double>(inputAmount);
        this->weightGrads = vector<double>(inputAmount);

        this->inputAmount = inputAmount;
        double startWeight;

        for (int i = 0; i < inputAmount; i++)
        {
            // gets random numbers btwn -1 and 1
            startWeight = (rand() / (double)RAND_MAX) * 2 - 1;
            this->weights.at(i) = startWeight;
            this->weightGrads.at(i) = 0.0;
        }

        bias = (rand() / (double)RAND_MAX) * 2 - 1;
    }

    // calculate output forward pass
    double calcOutput(vector<double> inputs)
    {

        double out = this->bias;
        this->inputs = inputs; // save inputs for backwards pass

        for (int i = 0; i < inputs.size(); i++)
        {
            out += inputs.at(i) * this->weights.at(i);
        }
        this->output = out;

        return out;
    }

    // calculate gradients backward pass
    void calcGrads(double grad) // grad is gradient from neuron next
    {
        this->biasGrad += grad;
        for (int i = 0; i < this->weights.size(); i++)
        {
            this->weightGrads[i] += grad * this->inputs[i];
        }
    }

    void zeroGrad()
    {
        this->biasGrad = 0.0;
        for (int i = 0; i < this->weights.size(); i++)
        {
            this->weightGrads[i] = 0.0;
        }
    }
};

int main()
{
    srand(time(NULL));
    // DATASET
    double x[100];
    double y[100];
    double m = -3.4;
    double b = 3.6;
    for (int i = 0; i < 100; i++)
    {

        // x goes from -2 to 2
        x[i] = (i / 25.0) - 2.0;
        y[i] = m * x[i] + b;
    }

    // NEURAL NETWORK SHIT
    Neuron n(1);

    cout << "Initial weights: ";
    printVector(n.weights);
    cout << "Initial bias: " << n.bias << endl;

    for (int epoch = 0; epoch < 100; epoch++)
    {
        for (int index = 0; index < 100; index++)
        {

            vector<double> input;
            input.push_back(x[index]);
            double yPred = n.calcOutput(input);
            double loss = pow(yPred - y[index], 2.0);

            double lossGrad = 2.0 * (yPred - y[index]);
            cout << yPred << endl;
            cout << "Loss: " << loss << endl;
            cout << lossGrad << endl;

            n.zeroGrad();
            n.calcGrads(lossGrad);

            cout << "Weight Gradients: ";
            printVector(n.weightGrads);
            cout << "Bias Gradient: " << n.biasGrad << endl;

            double learningRate = 0.001;

            // ADJUST THE WEIGHTS BASED ON THE GRADIENT
            for (int i = 0; i < n.weights.size(); i++)
            {
                n.weights.at(i) -= n.weightGrads.at(i) * learningRate;
            }

            n.bias -= learningRate * n.biasGrad;

            yPred = n.calcOutput(input);
            loss = pow(yPred - y[index], 2.0);

            cout << "New Loss: " << loss << endl;
            cout << "New Weights: ";
            printVector(n.weights);
            cout << "New Bias: " << n.bias << endl;
            cout << "--------------------------\n";
        }
    }

    return 0;
}