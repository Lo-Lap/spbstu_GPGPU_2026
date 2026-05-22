#ifndef ATTENTION_CUDA 
#define ATTENTION_CUDA

enum class AttentionMode 
{ 
    SIMPLE,
    FLASH 
};

void runAttention(const float* q, const float* k, const float* v, float* o, int n, int d, AttentionMode mode);

#endif
