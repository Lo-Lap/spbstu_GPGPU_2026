#include "ParticleCuda.cuh"
#include "CollisionHelpers.cuh"

__global__ void initializeRandomStatesKernel(DeviceParticleData particles, int n, unsigned long seed)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
    {
        return;
    }
    curandState* randomState = getRandomStatePointer(particles);
    curand_init(seed, i, 0, &randomState[i]);
}

__global__ void initializeParticlesKernel(DeviceParticleData particles, int n,
    ParticleEmitter emitter, SimulationParameters settings)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    respawnParticle(i, particles, emitter, settings);
}

__global__ void updateParticlesKernel(DeviceParticleData particles, int n, float dt,
    CollisionScene scene, SimulationParameters settings, ParticleEmitter emitter, int* score)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    float3 p = particles.position[i];
    float3 v = particles.velocity[i];
    float life = particles.lifetime[i] - dt;
    int t = particles.type[i];

    v.y -= settings.gravity * dt;
    p += v * dt;

    collideSphere(p, v, t, scene.sphere, settings);

    int planeCount = scene.planeCount < MAX_PLANES ? scene.planeCount : MAX_PLANES;
    for (int k = 0; k < planeCount; ++k)
        collidePlane(p, v, t, scene.planes[k], settings);

    particles.position[i] = p;
    particles.velocity[i] = v;
    particles.lifetime[i] = life;

    if (insideBasket(p, scene.basket))
    {
        atomicAdd(score, 1);
        respawnParticle(i, particles, emitter, settings);
        return;
    }
    if (life <= 0.0f || p.y < settings.worldMinY)
        respawnParticle(i, particles, emitter, settings);
}

__global__ void buildGridKernel(DeviceParticleData particles, int n, GridInfo grid,
    int* cellCount, int* cellParticles)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    int3 c = getCellCoordinate(particles.position[i], grid);
    int cellId = computeLinearCellIndex(c, grid);
    if (cellId < 0)
        return;

    int localIndex = atomicAdd(&cellCount[cellId], 1);
    if (localIndex < MAX_PARTICLES_PER_CELL)
        cellParticles[cellId * MAX_PARTICLES_PER_CELL + localIndex] = i;
}

__global__ void collideDifferentTypesKernel(DeviceParticleData particles, int n, GridInfo grid,
    int* cellCount, int* cellParticles, float radius)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
    {
        return;
    }
    float3 p = particles.position[i];
    float3 v = particles.velocity[i];
    int t = particles.type[i];

    int3 baseCell = getCellCoordinate(p, grid);
    float minDistance = radius * 2.0f;
    float minDistanceSq = minDistance * minDistance;

    for (int dz = -1; dz <= 1; ++dz)
    {
        for (int dy = -1; dy <= 1; ++dy)
        {
            for (int dx = -1; dx <= 1; ++dx)
            {
                int3 c = make_int3(baseCell.x + dx, baseCell.y + dy, baseCell.z + dz);
                int cellId = computeLinearCellIndex(c, grid);
                if (cellId < 0)
                    continue;
                int count = cellCount[cellId];
                count = count < MAX_PARTICLES_PER_CELL ? count : MAX_PARTICLES_PER_CELL;

                for (int k = 0; k < count; ++k)
                {
                    int j = cellParticles[cellId * MAX_PARTICLES_PER_CELL + k];
                    if (j == i)
                    {
                        continue;
                    }
                    if (particles.type[j] == t)
                        continue;

                    float3 otherP = particles.position[j];
                    float3 delta = p - otherP;
                    float distSq = lengthSquared3(delta);
                    if (distSq < 1e-8f || distSq > minDistanceSq)
                        continue;

                    float dist = sqrtf(distSq);
                    float3 normal = delta / dist;
                    float3 relativeVelocity = v - particles.velocity[j];
                    float relNormal = dot3(relativeVelocity, normal);
                    if (relNormal < 0.0f)
                    {
                        v -= normal * relNormal;
                    }
                }
            }
        }
    }
    particles.velocity[i] = v;
}