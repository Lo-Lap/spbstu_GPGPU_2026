#include <cuda_runtime.h>
#include <cmath>
#include <exception>
#include <iostream>
#include <iomanip>
#include <chrono>

#include "Attention.h"
#include "Matrix.h"
#include "CudaCheck.h"

float maxAbsError(Matrix const& a, Matrix const& b)
{
    float maxErr = 0.f;
    for (size_t i = 0; i < a.getHeight(); ++i)
    {
        for (size_t j = 0; j < a.getWidth(); ++j)
        {
            maxErr = std::max(maxErr, std::abs(a.at(i, j) - b.at(i, j)));
        }
    }
    return maxErr;
}

bool isCorrect(Matrix const& a, Matrix const& b, float eps = 1e-3f)
{
    return a.getHeight() == b.getHeight() && a.getWidth() == b.getWidth() && maxAbsError(a, b) <= eps;
}

int main()
{
    try
    {
        constexpr size_t N = 64;
        constexpr size_t D = 64;

        Matrix q = Matrix::random(N, D, 1, -1.f, 1.f);
        Matrix k = Matrix::random(N, D, 2, -1.f, 1.f);
        Matrix v = Matrix::random(N, D, 3, -1.f, 1.f);

        std::cout << "Simple Attention:\n";
        Matrix simple = Attention::forward(q, k, v, AttentionMode::SIMPLE);

        std::cout << "Flash Attention:\n";
        Matrix flash = Attention::forward(q, k, v, AttentionMode::FLASH);

        float err = maxAbsError(simple, flash);
        std::cout << "Max abs error: " << err << '\n';
        std::cout << (isCorrect(simple, flash) ? "CORRECT ANSWER" : "WRONG ANSWER") << '\n';

        CUDA_CHECK(cudaDeviceReset());
    }
    catch (std::exception const& e)
    {
        std::cerr << e.what() << std::endl;
        return -1;
    }

    return 0;
}
