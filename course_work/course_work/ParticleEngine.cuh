#ifndef PARTICLE_ENGINE
#define PARTICLE_ENGINE

#include <vector>
#include <cuda_runtime.h>
#include "SceneTypes.h"

struct ParticleState
{
    float3 position;
    int type;
};

struct DeviceParticleData
{
    float3* position = nullptr;
    float3* velocity = nullptr;
    float* lifetime = nullptr;
    int* type = nullptr;
    void* randomStates = nullptr;
};

struct CellGridDeviceData
{
    int* cellCount = nullptr;
    int* cellParticles = nullptr;
    int cellTotal = 0;
};

class ParticleEngine
{
public:
    ParticleEngine(int particleCount, GridInfo grid);
    ~ParticleEngine();

    ParticleEngine(const ParticleEngine&) = delete;
    ParticleEngine& operator=(const ParticleEngine&) = delete;

    void initialize(const ParticleEmitter& emitter, unsigned long seed);
    void resetScore();
    void step(float dt, const CollisionScene& scene, const SimulationParameters& settings, const ParticleEmitter& emitter);

    int getScore() const;
    std::vector<float3> downloadSamplePositions(int maxCount) const;
    std::vector<ParticleState> downloadParticleState(int maxCount) const;

private:
    int particleCount = 0;
    GridInfo grid{};
    DeviceParticleData particles{};
    CellGridDeviceData gridDevice{};
    int* deviceScore = nullptr;

    int threads = 256;
    int blocks = 0;
};

#endif