#ifndef MATRIX
#define MATRIX

#include <cstddef>
#include <cstdint>

enum class MatrixMulMode 
{ 
    SIMPLE, 
    SHARED, 
    INTRINSICS 
};

class Matrix 
{
public:
    Matrix(size_t h, size_t w);
    ~Matrix();

    Matrix(Matrix const& other);
    Matrix& operator=(Matrix const& other);

    Matrix(Matrix&& other);
    Matrix& operator=(Matrix&& other);

    static Matrix filled(float val, size_t h, size_t w);
    static Matrix random(size_t h, size_t w, unsigned seed, float min, float max);

    size_t getHeight() const { return m_h; }
    size_t getWidth()  const { return m_w; }

    float  at(size_t i, size_t j) const { return m_data[i * m_w + j]; }
    float& at(size_t i, size_t j) { return m_data[i * m_w + j]; }

    const float* getData() const { return m_data; }
    float* getData() { return m_data; }

private:
    float* m_data;
    size_t m_h, m_w;
};

#endif