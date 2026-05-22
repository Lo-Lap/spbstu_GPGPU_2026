#include "MatrixCuda.cuh"
#include "CudaCheck.h"

#include <iostream>
#include <memory>
#include <device_launch_parameters.h>

const size_t WARP_SIZE = 32;

__global__ void simpleMatMul(float* a, float* b, float* c, size_t l, size_t m, size_t n)
{
    size_t cCol = blockIdx.x * blockDim.x + threadIdx.x;
    size_t cRow = blockIdx.y * blockDim.y + threadIdx.y;

    if (cRow >= l || cCol >= n)
        return;

    float sum = 0;
    for (size_t i = 0; i < m; i++)
        sum += a[cRow * m + i] * b[i * n + cCol];
    c[cRow * n + cCol] = sum;
}

__global__ void sharedMatMul(float* a, float* b, float* c, size_t l, size_t m, size_t n)
{
    size_t cCol = blockIdx.x * blockDim.x + threadIdx.x;
    size_t cRow = blockIdx.y * blockDim.y + threadIdx.y;

    size_t tileCol = threadIdx.x;
    size_t tileRow = threadIdx.y;

    __shared__ float aTile[WARP_SIZE][WARP_SIZE];
    __shared__ float bTile[WARP_SIZE][WARP_SIZE + 1]; // +1 to avoid bank conflicts

    float cVal = 0.f;
    bool isOutOfC = cRow >= l || cCol >= n;

    for (size_t tileId = 0; tileId < (m - 1) / WARP_SIZE + 1; tileId++)
    {
        size_t aCol = tileId * WARP_SIZE + tileCol;
        size_t bRow = tileId * WARP_SIZE + tileRow;

        aTile[tileRow][tileCol] = (cRow < l && aCol < m) ? a[cRow * m + aCol] : 0.f;
        bTile[tileRow][tileCol] = (bRow < m && cCol < n) ? b[bRow * n + cCol] : 0.f;
        __syncthreads();

        for (size_t i = 0; i < WARP_SIZE; i++)
            cVal += aTile[tileRow][i] * bTile[i][tileCol];
        __syncthreads();
    }
    if (!isOutOfC)
        c[cRow * n + cCol] = cVal;
}

__global__ void __launch_bounds__(1024, 1) warpIntrinsicsMatMul(float* a, float* b, float* c, size_t l, size_t m, size_t n)
{
    size_t cCol = blockIdx.x * blockDim.x + threadIdx.x;
    size_t cRow = blockIdx.y * blockDim.y + threadIdx.y;
    size_t tileCol = threadIdx.x;
    size_t tileRow = threadIdx.y;

    __shared__ float aTile[WARP_SIZE][WARP_SIZE];
    __shared__ float bTile[WARP_SIZE][WARP_SIZE + 1];

    float cVal = 0.f;
    bool isOutOfC = cRow >= l || cCol >= n;

    for (size_t tileId = 0; tileId < (m - 1) / WARP_SIZE + 1; tileId++)
    {
        size_t aCol = tileId * WARP_SIZE + tileCol;
        size_t bRow = tileId * WARP_SIZE + tileRow;
        aTile[tileRow][tileCol] = (cRow < l && aCol < m) ? a[cRow * m + aCol] : 0.f;
        bTile[tileRow][tileCol] = (bRow < m && cCol < n) ? b[bRow * n + cCol] : 0.f;
        __syncthreads();

        float aTileLocal = aTile[tileRow][tileCol];
        __syncwarp();
        for (size_t i = 0; i < WARP_SIZE; i++)
            cVal += __shfl_sync(0xffffffff, aTileLocal, i) * bTile[i][tileCol];
        __syncthreads();
    }
    if (!isOutOfC)
        c[cRow * n + cCol] = cVal;
}

void matrixMultiplication(const float* a, const float* b, float* c, size_t l, size_t m, size_t n, MatrixMulMode mode)
{
    CUDA_CHECK(cudaSetDevice(0));

    auto deleter = [](float* p) { cudaFree(p); };
    std::unique_ptr<float, decltype(deleter)> aDev(nullptr, deleter);
    std::unique_ptr<float, decltype(deleter)> bDev(nullptr, deleter);
    std::unique_ptr<float, decltype(deleter)> cDev(nullptr, deleter);


    float* rawA = nullptr;
    float* rawB = nullptr;
    float* rawC = nullptr;
    CUDA_CHECK(cudaMalloc(&rawA, l * m * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&rawB, m * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&rawC, l * n * sizeof(float)));
    aDev.reset(rawA);
    bDev.reset(rawB);
    cDev.reset(rawC);

    CUDA_CHECK(cudaMemcpy(aDev.get(), a, l * m * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(bDev.get(), b, m * n * sizeof(float), cudaMemcpyHostToDevice));

    dim3 blockInGrid((n - 1ULL) / WARP_SIZE + 1ULL, (l - 1ULL) / WARP_SIZE + 1ULL);
    dim3 threadInBlock(WARP_SIZE, WARP_SIZE);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    switch (mode) 
    {
    case MatrixMulMode::SIMPLE:
        simpleMatMul <<< blockInGrid, threadInBlock >>> (aDev.get(), bDev.get(), cDev.get(), l, m, n);
        break;
    case MatrixMulMode::SHARED:
        sharedMatMul <<< blockInGrid, threadInBlock >>> (aDev.get(), bDev.get(), cDev.get(), l, m, n);
        break;
    case MatrixMulMode::INTRINSICS:
        warpIntrinsicsMatMul <<< blockInGrid, threadInBlock >>> (aDev.get(), bDev.get(), cDev.get(), l, m, n);
        break;
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "GPU kernel time: " << milliseconds << " ms\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(c, cDev.get(), l * n * sizeof(float), cudaMemcpyDeviceToHost));
}