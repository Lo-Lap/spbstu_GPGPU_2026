#include "Matrix.h"
#include "MatrixCuda.cuh"

#include <stdexcept>
#include <cstring>


Matrix::Matrix(size_t h, size_t w): m_data(nullptr), m_h(h), m_w(w)
{
    m_data = new float[h * w];
}

Matrix::Matrix(Matrix const& other): Matrix(other.m_h, other.m_w)
{
    std::memcpy(m_data, other.m_data, m_h * m_w * sizeof(float));
}

Matrix& Matrix::operator=(Matrix const& other)
{
    if (this == &other)
        return *this;
    delete[] m_data;
    m_h = other.m_h;
    m_w = other.m_w;
    m_data = new float[m_h * m_w];
    std::memcpy(m_data, other.m_data, m_h * m_w * sizeof(float));
    return *this;
}

Matrix::Matrix(Matrix&& other): m_data(other.m_data), m_h(other.m_h), m_w(other.m_w)
{
    other.m_data = nullptr;
    other.m_h = other.m_w = 0;
}

Matrix& Matrix::operator=(Matrix&& other)
{
    if (this != &other) 
    {
        delete[] m_data;
        m_data = other.m_data;
        m_h = other.m_h;
        m_w = other.m_w;
        other.m_data = nullptr;
        other.m_h = other.m_w = 0;
    }
    return *this;
}

Matrix Matrix::filled(float val, size_t h, size_t w)
{
    Matrix fullMatrix(h, w);
    for (size_t i = 0; i < h * w; i++)
        fullMatrix.m_data[i] = val;
    return fullMatrix;
}

Matrix Matrix::mul(Matrix const& other, MatrixMulMode mode) const
{
    if (this->m_w != other.m_h)
        throw std::invalid_argument("Matrix dimension mismatch");

    Matrix productMatrix(this->m_h, other.m_w);
    matrixMultiplication(this->m_data, other.m_data, productMatrix.m_data, this->m_h, this->m_w, other.m_w, mode);
    return productMatrix;
}

Matrix::~Matrix()
{
    delete[] m_data;
    m_data = nullptr;
}