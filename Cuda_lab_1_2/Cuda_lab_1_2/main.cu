#include <cuda_runtime.h>
#include <stdio.h>
#include <exception>
#include <cmath>
#include <chrono>
#include <iostream>

#include "Matrix.h"
#include "CudaCheck.h"

bool isCorrectAnswer(Matrix const& m, float val, float eps = 1e-6)
{
    for (size_t i = 0; i < m.getHeight(); i++) 
    {
        for (size_t j = 0; j < m.getWidth(); j++) 
        {
            if (std::abs(m.at(i, j) - val) > eps)
                return false;
        }
    }
    return true;
}

void printMatrix(Matrix const& m) 
{
    for (size_t i = 0; i < m.getHeight(); ++i)
    {
        for (size_t j = 0; j < m.getWidth(); ++j)
        {
            std::cout << m.at(i, j) << '\t';
        }
        std::cout << std::endl;
    }
}

int main()
{
    try 
    {
        size_t s =  1 << 10;
        Matrix m1 = Matrix::filled(1.f, s*2, s);
        Matrix m2 = Matrix::filled(1.f, s, s);
        
        auto start = std::chrono::high_resolution_clock::now();

        //lab 1
        Matrix m3 = m1.mul(m2, MatrixMulMode::SHARED);

        //lab 2
        //Matrix m3 = m1.mul(m2, MatrixMulMode::INTRINSICS);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        std::cout << "Matrix multiplication time (host): " << elapsed.count() << " ms\n";

        /*
        printMatrix(m1);
        std::cout << '*' << std::endl;
        printMatrix(m2);
        std::cout << '=' << std::endl;
        printMatrix(m3);
        */

        std::cout << (isCorrectAnswer(m3, (float)s) ? "CORRECT ANSWER" : "WRONG ANSWER") << '\n';

        CUDA_CHECK(cudaDeviceReset());
    }
    catch (std::exception& e) 
    {
        fprintf(stderr, e.what());
        return -1;
    }

    return 0;
}