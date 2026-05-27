#ifndef VECTOR_MATH
#define VECTOR_MATH

#include <cuda_runtime.h>
#include <cmath>

__host__ __device__ inline float3 operator+(const float3& a, const float3& b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ inline float3 operator-(const float3& a, const float3& b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ inline float3 operator-(const float3& a)
{
    return make_float3(-a.x, -a.y, -a.z);
}

__host__ __device__ inline float3 operator*(const float3& a, float s)
{
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__host__ __device__ inline float3 operator*(float s, const float3& a)
{
    return a * s;
}

__host__ __device__ inline float3 operator/(const float3& a, float s)
{
    return make_float3(a.x / s, a.y / s, a.z / s);
}

__host__ __device__ inline float3& operator+=(float3& a, const float3& b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}

__host__ __device__ inline float3& operator-=(float3& a, const float3& b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    return a;
}

__host__ __device__ inline float3& operator*=(float3& a, float s)
{
    a.x *= s;
    a.y *= s;
    a.z *= s;
    return a;
}

__host__ __device__ inline float dot3(const float3& a, const float3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ inline float lengthSquared3(const float3& a)
{
    return dot3(a, a);
}

__host__ __device__ inline float length3(const float3& a)
{
    return std::sqrt(lengthSquared3(a));
}

__host__ __device__ inline float3 normalize3(const float3& a)
{
    float len = length3(a);
    if (len < 1e-6f)
        return make_float3(0.0f, 1.0f, 0.0f);

    return a / len;
}

__host__ __device__ inline float3 reflect3(const float3& v, const float3& normal)
{
    return v - 2.0f * dot3(v, normal) * normal;
}

__host__ __device__ inline float clampFloat(float v, float lo, float hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

__host__ __device__ inline float3 clamp3(const float3& p, const float3& lo, const float3& hi)
{
    return make_float3(
        clampFloat(p.x, lo.x, hi.x),
        clampFloat(p.y, lo.y, hi.y),
        clampFloat(p.z, lo.z, hi.z)
    );
}

__host__ __device__ inline int floorToInt(float v)
{
    return static_cast<int>(std::floor(v));
}

__host__ __device__ inline float3 cross3(const float3& a, const float3& b)
{
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
#endif