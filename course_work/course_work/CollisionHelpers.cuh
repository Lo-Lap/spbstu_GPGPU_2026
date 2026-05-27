#ifndef COLLISION_HELPERS
#define COLLISION_HELPERS

#include <curand_kernel.h>
#include "SceneTypes.h"
#include "VectorMath.cuh"

inline __device__ curandState* getRandomStatePointer(DeviceParticleData particles)
{
    return reinterpret_cast<curandState*>(particles.randomStates);
}

inline __device__ float generateUniformRandom(curandState* state)
{
    return curand_uniform(state);
}

inline __device__ bool insideBasket(const float3& p, const TargetBasket& b)
{
    return p.x >= b.minCorner.x && p.x <= b.maxCorner.x &&
        p.y >= b.minCorner.y && p.y <= b.maxCorner.y &&
        p.z >= b.minCorner.z && p.z <= b.maxCorner.z;
}

inline __device__ float3 slideVelocity(const float3& velocity, const float3& normal, float rollingFriction)
{
    float3 normalPart = dot3(velocity, normal) * normal;
    float3 tangentPart = velocity - normalPart;
    return tangentPart * rollingFriction;
}

inline __device__ void respawnParticle(int i, DeviceParticleData particles, const ParticleEmitter& emitter, const SimulationParameters& settings)
{
    curandState* randomState = getRandomStatePointer(particles);
    curandState localState = randomState[i];

    float rx = generateUniformRandom(&localState) - 0.5f;
    float ry = generateUniformRandom(&localState) - 0.5f;
    float rz = generateUniformRandom(&localState) - 0.5f;

    particles.position[i] = make_float3(
        emitter.center.x + rx * emitter.size.x,
        emitter.center.y + ry * emitter.size.y,
        emitter.center.z + rz * emitter.size.z
    );

    float jitterX = (generateUniformRandom(&localState) - 0.5f) * 1.5f;
    float jitterZ = (generateUniformRandom(&localState) - 0.5f) * 1.5f;

    particles.velocity[i] = make_float3(
        emitter.baseVelocity.x + jitterX,
        emitter.baseVelocity.y,
        emitter.baseVelocity.z + jitterZ
    );

    particles.lifetime[i] = 8.0f + generateUniformRandom(&localState) * 8.0f;
    particles.type[i] = (i < settings.particleCount / 2) ? BOUNCE : SLIDE;
    randomState[i] = localState;
}

inline __device__ void collideSphere(float3& p, float3& v, int particleType, const Sphere& sphere, const SimulationParameters& settings)
{
    float3 delta = p - sphere.center;
    float distance = length3(delta);
    float minDistance = sphere.radius + settings.particleRadius;

    if (distance >= minDistance)
        return;

    float3 normal = distance < 1e-5f ? make_float3(0.0f, 1.0f, 0.0f) : delta / distance;
    p = sphere.center + normal * minDistance;

    if (particleType == BOUNCE)
        v = reflect3(v, normal) * settings.damping;
    else
        v = slideVelocity(v, normal, settings.rollingFriction);
}

inline __device__ void collidePlane(float3& p, float3& v, int particleType, const Plane& plane, const SimulationParameters& settings)
{
    float3 n = normalize3(plane.normal);
    float signedDistance = dot3(p - plane.point, n);

    if (signedDistance > settings.particleRadius)
        return;

    float3 u, w;
    if (fabsf(n.x) > 0.999f)
    {
        u = make_float3(0, 1, 0);
        w = make_float3(0, 0, 1);
    }
    else if (fabsf(n.y) > 0.999f)
    {
        u = make_float3(1, 0, 0);
        w = make_float3(0, 0, 1);
    }
    else if (fabsf(n.z) > 0.999f)
    {
        u = make_float3(1, 0, 0);
        w = make_float3(0, 1, 0);
    }
    else
    {
        float3 helper = fabsf(n.y) < 0.9f ? make_float3(0, 1, 0) : make_float3(1, 0, 0);
        u = normalize3(cross3(helper, n));
        w = cross3(n, u);
    }

    float3 d = p - plane.point;
    float projU = dot3(d, u);
    float projW = dot3(d, w);
    if (fabsf(projU) > plane.halfSize || fabsf(projW) > plane.halfSize)
        return;

    float vn = dot3(v, n);

    if (signedDistance < -settings.particleRadius && vn >= 0.0f)
        return;

    p += n * (settings.particleRadius - signedDistance);

    if (vn < 0.0f)
    {
        if (particleType == BOUNCE)
            v = reflect3(v, n) * settings.damping;
        else
            v = slideVelocity(v, n, settings.rollingFriction);
    }
}

inline __device__ void collideBox(float3& p,float3& v, int particleType, const Box& box, const SimulationParameters& settings)
{
    float3 minCorner = box.center - box.halfSize;
    float3 maxCorner = box.center + box.halfSize;

    float3 closest = clamp3(p, minCorner, maxCorner);
    float3 delta = p - closest;
    float distSq = lengthSquared3(delta);
    float radiusSq = settings.particleRadius * settings.particleRadius;

    if (distSq > 1e-8f && distSq < radiusSq)
    {
        float dist = sqrtf(distSq);
        float3 normal = delta / dist;
        p = closest + normal * settings.particleRadius;

        if (particleType == BOUNCE)
            v = reflect3(v, normal) * settings.damping;
        else
            v = slideVelocity(v, normal, settings.rollingFriction);
        return;
    }

    bool inside = p.x >= minCorner.x && p.x <= maxCorner.x &&
        p.y >= minCorner.y && p.y <= maxCorner.y &&
        p.z >= minCorner.z && p.z <= maxCorner.z;

    if (!inside)
        return;

    float left = fabsf(p.x - minCorner.x);
    float right = fabsf(maxCorner.x - p.x);
    float bottom = fabsf(p.y - minCorner.y);
    float top = fabsf(maxCorner.y - p.y);
    float back = fabsf(p.z - minCorner.z);
    float front = fabsf(maxCorner.z - p.z);

    float minPen = left;
    float3 normal = make_float3(-1.0f, 0.0f, 0.0f);
    if (right < minPen)
    {
        minPen = right;
        normal = make_float3(1.0f, 0.0f, 0.0f);
    }
    if (bottom < minPen)
    {
        minPen = bottom;
        normal = make_float3(0.0f, -1.0f, 0.0f);
    }
    if (top < minPen)
    {
        minPen = top;
        normal = make_float3(0.0f, 1.0f, 0.0f);
    }
    if (back < minPen)
    {
        minPen = back;
        normal = make_float3(0.0f, 0.0f, -1.0f);
    }
    if (front < minPen)
        normal = make_float3(0.0f, 0.0f, 1.0f);

    p += normal * (minPen + settings.particleRadius);

    if (particleType == BOUNCE)
        v = reflect3(v, normal) * settings.damping;
    else
        v = slideVelocity(v, normal, settings.rollingFriction);
}

inline __device__ int3 getCellCoordinate(const float3& p, const GridInfo& grid)
{
    return make_int3(
        floorToInt((p.x - grid.origin.x) / grid.cellSize),
        floorToInt((p.y - grid.origin.y) / grid.cellSize),
        floorToInt((p.z - grid.origin.z) / grid.cellSize)
    );
}

inline __device__ int computeLinearCellIndex(const int3& c, const GridInfo& grid)
{
    if (c.x < 0 || c.y < 0 || c.z < 0 ||
        c.x >= grid.dims.x || c.y >= grid.dims.y 
        || c.z >= grid.dims.z)
    {
        return -1;
    }

    return c.x + c.y * grid.dims.x + c.z * grid.dims.x * grid.dims.y;
}

#endif