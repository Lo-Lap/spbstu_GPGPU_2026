#include "ParticleEngine.cuh"
#include "ParticleCuda.cuh"
#include "CudaCheck.h"

#include <stdexcept>
#include <sstream>
#include <algorithm>

#include <curand_kernel.h>

ParticleEngine::ParticleEngine(int particleCount, GridInfo grid): particleCount(particleCount), grid(grid)
{
    blocks = (particleCount + threads - 1) / threads;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&particles.position), particleCount * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&particles.velocity), particleCount * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&particles.lifetime), particleCount * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&particles.type), particleCount * sizeof(int)));

    curandState* rngRaw = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&rngRaw), particleCount * sizeof(curandState)));
    particles.randomStates = rngRaw;

    gridDevice.cellTotal = grid.dims.x * grid.dims.y * grid.dims.z;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&gridDevice.cellCount), gridDevice.cellTotal * sizeof(int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&gridDevice.cellParticles),gridDevice.cellTotal * MAX_PARTICLES_PER_CELL * sizeof(int)));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceScore), sizeof(int)));
    resetScore();
}

ParticleEngine::~ParticleEngine()
{
    cudaFree(particles.position);
    cudaFree(particles.velocity);
    cudaFree(particles.lifetime);
    cudaFree(particles.type);
    cudaFree(particles.randomStates);
    cudaFree(gridDevice.cellCount);
    cudaFree(gridDevice.cellParticles);
    cudaFree(deviceScore);
}

void ParticleEngine::initialize(const ParticleEmitter& emitter, unsigned long seed)
{
    initializeRandomStatesKernel <<< blocks, threads >>> (particles, particleCount, seed);
    CUDA_CHECK(cudaGetLastError());

    SimulationParameters initSettings{};
    initSettings.particleCount = particleCount;
    initSettings.particleRadius = 0.08f;

    initializeParticlesKernel <<< blocks, threads >>> (particles, particleCount, emitter, initSettings);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void ParticleEngine::resetScore()
{
    CUDA_CHECK(cudaMemset(deviceScore, 0, sizeof(int)));
}

void ParticleEngine::step(float dt, const CollisionScene& scene, const SimulationParameters& settings, const ParticleEmitter& emitter)
{
    updateParticlesKernel <<< blocks, threads >>> (particles, particleCount, dt, scene, settings, emitter, deviceScore);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemset(gridDevice.cellCount, 0, gridDevice.cellTotal * sizeof(int)));

    buildGridKernel <<< blocks, threads >>> (particles, particleCount, grid, gridDevice.cellCount, gridDevice.cellParticles);
    CUDA_CHECK(cudaGetLastError());

    collideDifferentTypesKernel <<< blocks, threads >>> ( particles, particleCount, grid, gridDevice.cellCount, gridDevice.cellParticles, settings.particleRadius);
    CUDA_CHECK(cudaGetLastError());
}

int ParticleEngine::getScore() const
{
    int score = 0;
    CUDA_CHECK(cudaMemcpy(&score, deviceScore, sizeof(int), cudaMemcpyDeviceToHost));
    return score;
}

std::vector<float3> ParticleEngine::downloadSamplePositions(int maxCount) const
{
    int count = std::min(maxCount, particleCount);
    std::vector<float3> result(count);
    CUDA_CHECK(cudaMemcpy(result.data(), particles.position, count * sizeof(float3), cudaMemcpyDeviceToHost));
    return result;
}

std::vector<ParticleState> ParticleEngine::downloadParticleState(int maxCount) const
{
    int count = std::min(maxCount, particleCount);
    std::vector<float3> positions(count);
    std::vector<int> types(count);
    std::vector<ParticleState> result(count);

    CUDA_CHECK(cudaMemcpy(positions.data(), particles.position, count * sizeof(float3), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(types.data(), particles.type, count * sizeof(int), cudaMemcpyDeviceToHost));

    for (int i = 0; i < count; ++i)
    {
        result[i].position = positions[i];
        result[i].type = types[i];
    }
    return result;
}