#include <fstream>
#include <iostream>
#include <vector>

using namespace std;

void loadData(vector<vector<double>> &mnist_train_images, vector<int> &mnist_train_labels, string filepath)
{
    // Open mnist_train.csv
    ifstream file(filepath);
    if (!file)
    {
        cout << "Cannot open mnist_train.csv\n";
        return;
    }

    // save mnist_train.csv to vector
    string line;

    int counter = 0;
    while (getline(file, line))
    {
        cout << "line: " << counter++ << "   \r";
        vector<double> row;
        string num = "";
        // first number is label
        mnist_train_labels.push_back(stoi(string(1, line[0])));

        // rest of numbers are image
        // start at 2 as first chars are label and comma
        for (int i = 2; i < line.length(); i++)
        {
            if (line[i] == ',')
            {
                // ensures network gets small numbers
                row.push_back(stod(num) / 255.0);
                num = "";
            }
            else
            {
                num += line[i];
            }
        }
        // ensures network gets small numbers and add final pixel
        row.push_back(stod(num) / 255.0);
        // add row to images
        mnist_train_images.push_back(row);
    }
    cout << endl
         << endl;
}