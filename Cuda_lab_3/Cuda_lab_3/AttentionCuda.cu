#include "AttentionCuda.cuh"
#include "CudaCheck.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <limits>
#include <memory>
#include <iostream>

static constexpr int BLOCK_THREADS = 128;
static constexpr int KV_TILE = 32;
static constexpr int MAX_D = 128; 

__device__ float dotRow(const float* q, const float* k, int qRow, int kRow, int d)
{
    float sum = 0.f;
    for (int t = 0; t < d; ++t)
        sum += q[qRow * d + t] * k[kRow * d + t];
    return sum;
}

__global__ void simpleAttentionKernel(const float* q, const float* k, const float* v, float* o, int n, int d, float scale)
{
    int row = blockIdx.x;
    if (row >= n || threadIdx.x != 0) 
        return;

    float maxScore = -FLT_MAX;
    for (int j = 0; j < n; ++j) 
    {
        float score = dotRow(q, k, row, j, d) * scale;
        maxScore = fmaxf(maxScore, score);
    }

    float denom = 0.f;
    for (int j = 0; j < n; ++j) 
    {
        float score = dotRow(q, k, row, j, d) * scale;
        denom += expf(score - maxScore);
    }

    for (int col = 0; col < d; ++col) 
    {
        float acc = 0.f;
        for (int j = 0; j < n; ++j) 
        {
            float score = dotRow(q, k, row, j, d) * scale;
            float p = expf(score - maxScore) / denom;
            acc += p * v[j * d + col];
        }
        o[row * d + col] = acc;
    }
}

__global__ void flashAttentionKernel(const float* q, const float* k, const float* v, float* o, int n, int d, float scale)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= n) 
        return;

    __shared__ float qTile[MAX_D];
    __shared__ float kTile[KV_TILE][MAX_D];
    __shared__ float vTile[KV_TILE][MAX_D];
    __shared__ float scores[KV_TILE];
    __shared__ float reduce[BLOCK_THREADS];
    __shared__ float sharedM, sharedL, sharedAlpha;

    if (tid < d) qTile[tid] = q[row * d + tid];
    __syncthreads();

    float acc = 0.f;
    float m = -FLT_MAX;
    float l = 0.f;

    for (int tileStart = 0; tileStart < n; tileStart += KV_TILE) 
    {
        int tileSize = min(KV_TILE, n - tileStart);

        for (int idx = tid; idx < tileSize * d; idx += BLOCK_THREADS) 
        {
            int tileRow = idx / d;
            int dim = idx % d;
            int globalRow = tileStart + tileRow;
            kTile[tileRow][dim] = k[globalRow * d + dim];
            vTile[tileRow][dim] = v[globalRow * d + dim];
        }
        __syncthreads();

        for (int tileRow = 0; tileRow < tileSize; ++tileRow) 
        {
            float partial = (tid < d) ? qTile[tid] * kTile[tileRow][tid] : 0.f;
            reduce[tid] = partial;
            __syncthreads();

            for (int stride = BLOCK_THREADS / 2; stride > 0; stride >>= 1) 
            {
                if (tid < stride) reduce[tid] += reduce[tid + stride];
                __syncthreads();
            }

            if (tid == 0) scores[tileRow] = reduce[0] * scale;
            __syncthreads();
        }

        if (tid == 0) 
        {
            float newM = m;
            for (int tileRow = 0; tileRow < tileSize; ++tileRow)
                newM = fmaxf(newM, scores[tileRow]);

            float alpha = (l == 0.f) ? 0.f : expf(m - newM);
            float newL = l * alpha;
            for (int tileRow = 0; tileRow < tileSize; ++tileRow)
                newL += expf(scores[tileRow] - newM);

            sharedM = newM;
            sharedL = newL;
            sharedAlpha = alpha;
            m = newM;
            l = newL;
        }
        __syncthreads();

        if (tid < d) 
        {
            acc *= sharedAlpha;
            for (int tileRow = 0; tileRow < tileSize; ++tileRow)
                acc += expf(scores[tileRow] - sharedM) * vTile[tileRow][tid];
        }
        __syncthreads();
    }

    if (tid < d) o[row * d + tid] = acc / sharedL;
}


void runAttention(const float* q, const float* k, const float* v,
    float* o, int n, int d, AttentionMode mode)
{
    CUDA_CHECK(cudaSetDevice(0));

    size_t bytes = static_cast<size_t>(n) * d * sizeof(float);

    auto deleter = [](float* p) { cudaFree(p); };
    std::unique_ptr<float, decltype(deleter)> qDev(nullptr, deleter);
    std::unique_ptr<float, decltype(deleter)> kDev(nullptr, deleter);
    std::unique_ptr<float, decltype(deleter)> vDev(nullptr, deleter);
    std::unique_ptr<float, decltype(deleter)> oDev(nullptr, deleter);

    float* rawQ = nullptr;
    float* rawK = nullptr; 
    float* rawV = nullptr; 
    float* rawO = nullptr;
    CUDA_CHECK(cudaMalloc(&rawQ, bytes));
    CUDA_CHECK(cudaMalloc(&rawK, bytes));
    CUDA_CHECK(cudaMalloc(&rawV, bytes));
    CUDA_CHECK(cudaMalloc(&rawO, bytes));
    qDev.reset(rawQ);
    kDev.reset(rawK);
    vDev.reset(rawV);
    oDev.reset(rawO);

    CUDA_CHECK(cudaMemcpy(qDev.get(), q, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(kDev.get(), k, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(vDev.get(), v, bytes, cudaMemcpyHostToDevice));

    float scale = 1.f / sqrtf(static_cast<float>(d));
    dim3 grid(n);
    dim3 block(BLOCK_THREADS);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    switch (mode) 
    {
    case AttentionMode::SIMPLE:
        simpleAttentionKernel <<< grid, block >>> (qDev.get(), kDev.get(), vDev.get(), oDev.get(), n, d, scale);
        break;
    case AttentionMode::FLASH:
        flashAttentionKernel <<< grid, block >>> (qDev.get(), kDev.get(), vDev.get(), oDev.get(), n, d, scale);
        break;
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    std::cout << "GPU attention kernel time: " << ms << " ms\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(o, oDev.get(), bytes, cudaMemcpyDeviceToHost));
}