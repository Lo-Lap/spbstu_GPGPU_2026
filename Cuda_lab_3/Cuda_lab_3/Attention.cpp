#include "Attention.h"
#include <stdexcept>

Matrix Attention::forward(const Matrix& q, const Matrix& k, const Matrix& v, AttentionMode mode)
{
    if (q.getHeight() != k.getHeight() || q.getHeight() != v.getHeight())
        throw std::invalid_argument("Q, K, V must have the same number of rows N");

    if (q.getWidth() != k.getWidth() || q.getWidth() != v.getWidth())
        throw std::invalid_argument("Q, K, V must have the same hidden dimension D");

    if (q.getWidth() > 128)
        throw std::invalid_argument("This Flash Attention kernel supports D <= 128");

    if (q.getWidth() == 0 || q.getHeight() == 0)
        throw std::invalid_argument("Q, K, V must be non-empty");

    Matrix result(q.getHeight(), q.getWidth());
    runAttention(q.getData(), k.getData(), v.getData(), result.getData(), static_cast<int>(q.getHeight()), static_cast<int>(q.getWidth()), mode);
    return result;
}