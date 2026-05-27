#ifndef SCENE_TYPES
#define SCENE_TYPES

#include <cuda_runtime.h>

constexpr int MAX_PLANES = 2;
constexpr int MAX_PARTICLES_PER_CELL = 64;

enum ParticleType
{
    BOUNCE = 0,
    SLIDE = 1
};

struct ParticleEmitter
{
    float3 center;
    float3 size;
    float3 baseVelocity;
};

struct Sphere
{
    float3 center;
    float radius;
};

struct Box
{
    float3 center;
    float3 halfSize;
};

struct Plane
{
    float3 point;
    float3 normal;
    float halfSize;
};

struct TargetBasket
{
    float3 minCorner;
    float3 maxCorner;
};

struct CollisionScene
{
    Sphere sphere;
    Box box;
    Plane planes[MAX_PLANES];
    int planeCount;
    TargetBasket basket;
};

struct GridInfo
{
    float3 origin;
    float cellSize;
    int3 dims;
};

struct SimulationParameters
{
    int particleCount;
    int targetScore;
    float gameDuration;
    float particleRadius;
    float worldMinY;
    float damping;
    float rollingFriction;
    float gravity;
};

#endif