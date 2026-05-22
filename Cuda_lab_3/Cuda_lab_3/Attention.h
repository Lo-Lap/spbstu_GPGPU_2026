#ifndef ATTENTION 
#define ATTENTION

#include "Matrix.h"
#include "AttentionCuda.cuh"

class Attention 
{
public:
    static Matrix forward(const Matrix& q, const Matrix& k, const Matrix& v, AttentionMode mode);
};
#endif