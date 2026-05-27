#ifndef PARTICLE_CUDA
#define PARTICLE_CUDA

#include "ParticleEngine.cuh"
#include "SceneTypes.h"

__global__ void initializeRandomStatesKernel(DeviceParticleData particles, int n, unsigned long seed);

__global__ void initializeParticlesKernel(DeviceParticleData particles, int n, ParticleEmitter emitter, SimulationParameters settings);

__global__ void updateParticlesKernel(DeviceParticleData particles, int n, float dt, CollisionScene scene,
    SimulationParameters settings, ParticleEmitter emitter, int* score);

__global__ void buildGridKernel(DeviceParticleData particles, int n, GridInfo grid,
    int* cellCount, int* cellParticles);

__global__ void collideDifferentTypesKernel(DeviceParticleData particles, int n, GridInfo grid,
    int* cellCount, int* cellParticles, float radius);

#endif