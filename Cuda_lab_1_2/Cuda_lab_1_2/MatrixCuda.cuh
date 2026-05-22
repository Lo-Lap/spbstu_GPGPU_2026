#ifndef MATRIX_CUDA
#define MATRIX_CUDA

#include <cstddef>
#include "Matrix.h"

void matrixMultiplication(const float* a, const float* b, float* c, size_t l, size_t m, size_t n, MatrixMulMode mode);

#endif