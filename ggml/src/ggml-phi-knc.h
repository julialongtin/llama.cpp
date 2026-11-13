// Formatted with: indent -npcs -nlp -i4 -l300
#pragma once

#include "ggml.h"

#ifdef  __cplusplus
extern "C"
{
#endif

    /* A forward declaration, to keep GCC happy. */
    void ggml_vec_dot_f32(int n, float *restrict s, size_t bs, const float *restrict x, size_t bx, const float *restrict y, size_t by, int nrc);
  float ggml_fp16_to_fp32(ggml_fp16_t src);

#ifdef  __cplusplus
}
#endif
