#ifndef CUDA_CHECK_H
#define CUDA_CHECK_H

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

inline void cudaCheckImpl(cudaError_t err, int line) 
{
    if (err != cudaSuccess) 
    {
        throw std::runtime_error(
            std::string("CUDA error at line ") +
            std::to_string(line) + ": " + cudaGetErrorString(err));
    }
}

#define CUDA_CHECK(call) cudaCheckImpl((call), __LINE__)

#endif
