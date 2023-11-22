#include <fstream>
#include <iostream>
#include <vector>

using namespace std;

void printAsciiImage(const std::vector<double> &values)
{
    // Check if the size of the input vector is correct
    cout << "values.size(): " << values.size() << endl;

    if (values.size() != 784)
    {
        std::cerr << "Error: Input vector size is not 784." << std::endl;
        return;
    }

    // Define characters to represent different intensity levels
    const char intensityChars[] = {' ', '.', ',', ':', 'o', 'O', 'X', '#', '$', '@'};

    // Calculate the range for each intensity level
    const double range = 1.0 / (sizeof(intensityChars) / sizeof(intensityChars[0]) - 1);

    // Iterate over the vector and print ASCII characters based on the values
    for (int i = 0; i < values.size(); ++i)
    {
        // Adjust the intensity to a character in the range of ASCII characters
        int intensityLevel = static_cast<int>(values[i] / range);
        intensityLevel = std::min(std::max(intensityLevel, 0), static_cast<int>(sizeof(intensityChars) - 1));

        char pixel = intensityChars[intensityLevel];

        // Print the ASCII character
        std::cout << pixel;

        // Insert a newline character after every 28 characters to create a 28x28 image
        if ((i + 1) % 28 == 0)
        {
            std::cout << std::endl;
        }
    }
}

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
    /*
        // print first image
        for (int i = 0; i < 28; i++)
        {
            printAsciiImage(mnist_train_images[i]);
        }*/
}