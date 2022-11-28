#include <iostream>
#include <vector>
#include <cmath>

constexpr unsigned long int N = 1'000'000'000U;
constexpr int n_moment = 4;

int main(int argc, char **argv) {
    double moment = 0.0;
    std::vector<double> numbers;
    numbers.reserve(N);
    numbers.resize(N);
    for (int i = 1; i <= N; i++) {
        numbers[i-1] = (double) i;
    }
    for (int i = 0; i < N; i++) {
        moment += pow(numbers[i], n_moment);
    }
    moment /= N;

    printf("The 4th moment is %f\n", moment);
}