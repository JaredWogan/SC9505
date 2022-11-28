#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>

int main(int argc, char** argv) {
    int n = atoi(argv[1]);
    int chunk = atoi(argv[2]);
    std::srand((unsigned) time(NULL));

    if (n % chunk != 0 ) throw "arg2 must divide arg1";

    std::ofstream fout ("DotData.txt");
    std::stringstream ss;

    int c = n / chunk;
    if (n % chunk) c += 1;

    if (fout.is_open()) {
        fout << n << std::endl;
        for (int i = 0; i < c; i++) {
            for (int j = 0; j < chunk; j++) {
                float num1 = (float) std::rand() / (float) RAND_MAX;
                float num2 = (float) std::rand() / (float) RAND_MAX;
                ss << num1 << " " << num2 << std::endl;
            }
            fout << ss.rdbuf();
        }
        fout.close();
    } else {
        std::cout << "Couldn't open the file\n";
    }
    return 0;
}