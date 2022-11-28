#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

int main(int argc, char** argv)
{
    // The result of the read is placed in here
    // In C++, we use a vector like an array but vectors can dynamically grow
    // as required when we get more data.
    std::vector<std::vector<float>>     data;
    std::ifstream          file("DotData.txt");
    std::string   line;
    int n;
    file >> n;
    printf("%d\n", n);
    std::getline(file, line); // read the newline character
    int j = 0;
    // Read one line at a time into the variable line:
    while(std::getline(file, line))
    {
        std::vector<float>   lineData;
        std::stringstream  lineStream(line);

        float value;
        // Read an integer at a time from the line
        while(lineStream >> value)
        {
            // Add the integers from a line to a 1D array (vector)
            lineData.push_back(value);
        }
        // When all the integers have been read, add the 1D array
        // into a 2D array (as one line in the 2D array)
        data.push_back(lineData);
    }
    for (int i = 0; i < 2; i++) {
        std::cout << data[i][0] << " " << data[i][1] << std::endl;
    }
}