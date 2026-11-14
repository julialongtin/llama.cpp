/* Xeon PHI IMCI support. */
/* formatted by using emacs, with (M-x set-variable RET c-basic-offset RET 4 RET) executed. */
/* formatted by using emacs, with (M-x set-variable RET indent-tabs-mode RET nil RET) executed. */

// For uint32_t
#include <stdint.h>

// For size_t
#include <stdio.h>

// for memcpy
#include <string.h>

// Yes, we have to tell this header to actually export stuff.
#define GGML_COMMON_IMPL_C
#include "ggml-quants.h"
#include "ggml-impl.h"

// For block_q5_K and block_q8_K.
#include "ggml-common.h"

// Handle unused funcion arguments in a way that GCC does not warn about them.
#define UNUSED GGML_UNUSED

// For our vector types, and forward declarations.
#include "ggml-phi-knc-dot_q5_K_q8_K.h"

// We can fit 16 float32s in a single vector register.
#define GGML_F32_EPR 16

/* Clear a vector of 16 floats. */
void GGML_F32x16_VEC_ZERO(float32x16_t *target)
{
    uint8_t zero=0;

    __asm__ __volatile__ (
                          "vbroadcastss\t%[Z]%{uint8%},\t%%zmm0\n\t" // use an upscaling operator to clear our register.
                          "vmovaps\t\t%%zmm0,\t%[RES]\n\t"
                          : [RES]  "+m"  (*target)
                          : [Z]     "m"  (zero)
                          : "zmm0", "memory");
}

/* convert many FP16s to FP32s. */
void GGML_PHI_FP16_TO_FP32_ROW(const ggml_fp16_t * x, float * y, int n)
{
    int i = 0;

    // Handle the aligned part..
    if ((((uintptr_t) x & 31) == 0) && (((uintptr_t) y & 63) == 0)) {

      // 64 FP16s in (128 bytes = 2 cache lines), 64 FP32s out (256 bytes = 4 cache lines).
      for (; i + 64 <= n; i += 64) {
	__asm__ __volatile__ (
			      "vprefetch0\t(%[SRC])\n\t"
			      "vprefetch0\t64(%[SRC])\n\t"
			      "vmovaps\t(%[SRC])%{float16%},\t%%zmm1\n\t"    // load + up-convert, 4 x 16.
			      "vmovaps\t32(%[SRC])%{float16%},\t%%zmm2\n\t"
			      "vmovaps\t64(%[SRC])%{float16%},\t%%zmm3\n\t"
			      "vmovaps\t96(%[SRC])%{float16%},\t%%zmm4\n\t"
			      "vmovaps\t%%zmm1,\t(%[DST])\n\t"             // full-cache-line stores:
			      "vmovaps\t%%zmm2,\t64(%[DST])\n\t"           // no need to fetch the destination first.
			      "vmovaps\t%%zmm3,\t128(%[DST])\n\t"
			      "vmovaps\t%%zmm4,\t192(%[DST])\n\t"
			      :
			      : [SRC] "r" (x + i),
				[DST] "r" (y + i)
			      : "zmm1", "zmm2", "zmm3", "zmm4", "memory");
      }

      // what is left over, 16 at a time.
      for (; i + 16 <= n; i += 16) {
	__asm__ __volatile__ (
			      "vmovaps\t%[SRC]%{float16%},\t%%zmm1\n\t"  // load 16 FP16s, up-converting to 16 FP32s.
			      "vmovaps\t\t%%zmm1,\t%[DST]\n\t"          // store 16 FP32s.
			      : [DST] "=m" (*(float (*)[16]) (y + i))
			      : [SRC]  "m" (*(const ggml_fp16_t (*)[16]) (x + i))
			      : "zmm1");
      }
    }

    // and the remainder.
    for (; i < n; i++) {
        y[i] = ggml_fp16_to_fp32(x[i]);
    }
}

/* Convert a FP32 to a FP16. */
inline ggml_fp16_t GGML_PHI_FP32_TO_FP16_INLINE(float src)
{
    uint32_t mask=0x0001;

    // we declare this as an array, so it ends up in a different memory section.
    ggml_fp16_t f16[1] __attribute__((aligned(64)));

    __asm__ __volatile__ (
                          "kmov\t%[M],\t%%k1\n\t"
                          "vbroadcastss\t%[SRC],\t%%zmm1%{%%k1%}\n\t"
                          "vmovaps\t\t%%zmm1%{float16%},\t%[DST]%{%%k1%}\n\t"
                          : [DST]  "+m"  (f16)
                          : [SRC]   "m"  (src),
                            [M]     "r"  (mask)
                          : "zmm1", "memory", "k1");
    return f16[0];
}

/* convert many FP32s to FP16s. */
void GGML_PHI_FP32_TO_FP16_ROW(const float * x, ggml_fp16_t * y, int n)
{
    int i = 0;

    if ((((uintptr_t) x & 63) == 0) && (((uintptr_t) y & 31) == 0)) {
        // 64 FP32s in (256 bytes = 4 cache lines), 64 FP16s out (128 bytes = 2 cache lines).
        // The stores only fill half a cache line each, so the no-read hint can't be used;
        // instead, prefetch the destination with intent to write (vprefetche1).
        for (; i + 64 <= n; i += 64) {
	  __asm__ __volatile__ (
				"vprefetch0\t(%[SRC])\n\t"                     // near: source into L1.
				"vprefetch0\t64(%[SRC])\n\t"
				"vprefetch0\t128(%[SRC])\n\t"
				"vprefetch0\t192(%[SRC])\n\t"
				"vmovaps\t\t(%[SRC]),\t%%zmm1\n\t"             // load, 4 x 16.
				"vmovaps\t\t64(%[SRC]),\t%%zmm2\n\t"
				"vmovaps\t\t128(%[SRC]),\t%%zmm3\n\t"
				"vmovaps\t\t192(%[SRC]),\t%%zmm4\n\t"
				"vmovaps\t\t%%zmm1%{float16%},\t(%[DST])\n\t"  // store + down-convert, 4 x 16.
				"vmovaps\t\t%%zmm2%{float16%},\t32(%[DST])\n\t"
				"vmovaps\t\t%%zmm3%{float16%},\t64(%[DST])\n\t"
				"vmovaps\t\t%%zmm4%{float16%},\t96(%[DST])\n\t"
				:
				: [SRC] "r" (x + i),
				  [DST] "r" (y + i)
				: "zmm1", "zmm2", "zmm3", "zmm4", "memory");
	}

        // what is left over, 16 at a time.
        for (; i + 16 <= n; i += 16) {
	  __asm__ __volatile__ (
				"vmovaps\t\t(%[SRC]),\t%%zmm2\n\t"
				"vmovaps\t\t%%zmm2%{float16%},\t(%[DST])\n\t"
				:
				: [SRC] "r" (x + i),
				  [DST] "r" (y + i)
				: "zmm2", "memory");
	}
    }

    // and the remainder
    for (; i < n; i++) {
        y[i] = GGML_PHI_FP32_TO_FP16_INLINE(x[i]);
    }
}


// The assembly below addresses the block members by offset. Make sure they are where we think they are.
_Static_assert(offsetof(block_q5_K, d)  ==  0, "block_q5_K.d moved");
_Static_assert(offsetof(block_q5_K, qh) == 16, "block_q5_K.qh moved");
_Static_assert(offsetof(block_q5_K, qs) == 48, "block_q5_K.qs moved");
_Static_assert(offsetof(block_q8_K, d)  ==  0, "block_q8_K.d moved");
_Static_assert(offsetof(block_q8_K, qs) ==  4, "block_q8_K.qs moved");

// Unpack one super-block of 256 5 bit values, multiply them by the 256 8 bit values of y,
// scale each group of 32 products by its 6 bit scale, and add (sum * x->d * y->d) into res.
// scale: the 8 unpacked scales of this super-block.
__attribute__((noinline))
void GGML_Q5K_Q8K_BLOCK_FMA (const block_q5_K * x, const block_q8_K * y, const uint8_t * scale, float32x16_t * res)
{
  uint8_t zero = 0;
  uint8_t lowmask = 0x0F;
  uint8_t bit5 = 0x10;
  uint8_t m = 1;
  uint64_t q8offset = ((uint64_t) y->qs) & 0x3f;

  __asm__ __volatile__ (
			"vprefetch0\t(%[X])\n\t"                            // Issue our memory requests first thing.
			"vprefetch0\t64(%[X])\n\t"
			"vprefetch0\t(%[Y])\n\t"
			"vprefetch0\t64(%[Y])\n\t"
			"lea\t48(%[X]),\t%%r8\n\t"                          // r8: our packed 4 bit values (x->qs).
			"mov\t%[SCALE],\t%%r9\n\t"                          // r9: our scales.
			"lea\t4(%[Y]),\t%%r11\n\t"                          // r11: our 8 bit values (y->qs), for vloadunpackld.
			"mov\t$0,\t%%ecx\n\t"
			"mov\t%%r11,\t%%r15\n\t"                            // Use r12-r15 to store the addresses for vloadunpackhd.
			"mov\t%%r11,\t%%r14\n\t"
			"mov\t%%r11,\t%%r13\n\t"
			"mov\t%%r11,\t%%r12\n\t"
			"mov\t%[OFFSET],\t%%r10\n\t"
			"cmp\t$32,%%r10\n\t"                                // Examine OFFSET, and decide which (if any) of the vloadunpackhd invocations needs to be increased by 64.
			"jl\t10f\n\t"
			"cmp\t$48,%%r10\n\t"
			"jl\t11f\n\t"
			"add\t$64,%%r12\n\t"                                // Greater than 47.
			"jmp\t14f\n\t"
			"11:\n\t"
			"add\t$64,%%r13\n\t"                                // Between 48 and 31.
			"jmp\t14f\n\t"
			"10:\n\t"                                           // Less than 32...
			"cmp\t$16,%%r10\n\t"
			"jz\t14f\n\t"                                       // Sixteen.
			"jl\t13f\n\t"
			"add\t$64,%%r14\n\t"                                // Between 32 and 15.
			"jmp\t14f\n\t"
			"13:\n\t"
			"add\t$64,%%r15\n\t"                                // Between 16 and zero.
			"14:\n\t"
			"vpbroadcastd\t%[MASK]%{uint8%},\t%%zmm0\n\t"       // Load our mask.
			"vpbroadcastd\t%[BIT5]%{uint8%},\t%%zmm1\n\t"       // Load the bit we want to add (conditionally).
			"vpbroadcastd\t%[M]%{uint8%},\t%%zmm2\n\t"          // Select which bit we want to test for. Start with bit 1.
			"vmovdqa32\t16(%[X])%{uint8%},\t%%zmm3\n\t"         // Load 16 sets of 8 packed single bits (x->qh).
			"vmovdqa32\t32(%[X])%{uint8%},\t%%zmm4\n\t"         // Load the next 16 sets of 8 packed single bits.
			"vpbroadcastd\t%[Z]%{uint8%},\t%%zmm7\n\t"          // Empty our result.

			"1:\n\t"                                            // One pass: 32 bytes of 4 bit pairs -> 64 5 bit values, times 64 8 bit values.
			"inc\t%%ecx\n\t"                                    // We are in our loop, increment our counter.

			// Unpack. zmm8, zmm9: the low nibbles (values 0-31). zmm10, zmm11: the high nibbles (values 32-63).
			"vptestmd\t%%zmm3,\t%%zmm2,\t%%k1\n\t"              // Test to see if our selected bit is set.
			"vptestmd\t%%zmm4,\t%%zmm2,\t%%k2\n\t"
			"vmovdqa32\t(%%r8)%{uint8%},\t%%zmm5\n\t"           // Load 16 bytes. Each holds two 4 bit sequences.
			"vmovdqa32\t16(%%r8)%{uint8%},\t%%zmm6\n\t"         // Load the next 16 bytes.
			"vpslld\t$1,\t%%zmm2,\t%%zmm2\n\t"                  // Select the next bit to test for.
			"vpandd\t%%zmm0,\t%%zmm5,\t%%zmm8\n\t"              // Apply a mask, storing the low four bits into a vector.
			"vpandd\t%%zmm0,\t%%zmm6,\t%%zmm9\n\t"
			"vpord\t%%zmm1,\t%%zmm8,\t%%zmm8%{%%k1%}\n\t"       // Turn on bit 5 for all values that passed the prior test.
			"vpord\t%%zmm1,\t%%zmm9,\t%%zmm9%{%%k2%}\n\t"
			"vptestmd\t%%zmm3,\t%%zmm2,\t%%k1\n\t"              // Test to see if our (next) selected bit is set.
			"vptestmd\t%%zmm4,\t%%zmm2,\t%%k2\n\t"
			"vpsrld\t$4,\t%%zmm5,\t%%zmm10\n\t"                 // The high four bits.
			"vpsrld\t$4,\t%%zmm6,\t%%zmm11\n\t"
			"vpslld\t$1,\t%%zmm2,\t%%zmm2\n\t"                  // Select the next bit to test for.
			"vpord\t%%zmm1,\t%%zmm10,\t%%zmm10%{%%k1%}\n\t"     // Turn on bit 5 for all values that passed the prior test.
			"vpord\t%%zmm1,\t%%zmm11,\t%%zmm11%{%%k2%}\n\t"

			// Load 64 8 bit values. Upscale them from int8 to int32.
			"vloadunpackld\t\t(%%r11)%{sint8%},\t%%zmm12\n\t"
			"vloadunpackld\t\t16(%%r11)%{sint8%},\t%%zmm13\n\t"
			"vloadunpackld\t\t32(%%r11)%{sint8%},\t%%zmm14\n\t"
			"vloadunpackld\t\t48(%%r11)%{sint8%},\t%%zmm15\n\t"
			"vprefetch1\t128(%%r11)\n\t"                        // Prepare for a run-through.
			"add\t$64,\t%%r11\n\t"
			"vloadunpackhd\t\t(%%r12)%{sint8%},\t%%zmm12\n\t"
			"add\t$64,\t%%r12\n\t"
			"vloadunpackhd\t\t16(%%r13)%{sint8%},\t%%zmm13\n\t"
			"add\t$64,\t%%r13\n\t"
			"vloadunpackhd\t\t32(%%r14)%{sint8%},\t%%zmm14\n\t"
			"add\t$64,\t%%r14\n\t"
			"vloadunpackhd\t\t48(%%r15)%{sint8%},\t%%zmm15\n\t"
			"add\t$64,\t%%r15\n\t"

			// Multiply, straight out of the registers: no trip through memory.
			"vpmulld\t%%zmm12,\t%%zmm8,\t%%zmm8\n\t"
			"vpmulld\t%%zmm13,\t%%zmm9,\t%%zmm9\n\t"
			"vpmulld\t%%zmm14,\t%%zmm10,\t%%zmm10\n\t"
			"vpmulld\t%%zmm15,\t%%zmm11,\t%%zmm11\n\t"

			"vpbroadcastd\t(%%r9)%{uint8%},\t%%zmm5\n\t"        // Load the scale for values 0-31.
			"vpbroadcastd\t1(%%r9)%{uint8%},\t%%zmm6\n\t"       // Load the scale for values 32-63.
			"add\t$2,\t%%r9\n\t"
			"vprefetch1\t96(%%r8)\n\t"                          // Prepare for a run-through.
			"add\t$32,\t%%r8\n\t"
			"vprefetch0\t(%%r11)\n\t"
			"vprefetch0\t64(%%r11)\n\t"
			"vprefetch0\t32(%%r8)\n\t"
			"cmp\t$4,\t%%ecx\n\t"                               // See if this is our last run-through.
			"vpmadd231d\t%%zmm8,\t%%zmm5,\t%%zmm7\n\t"          // Perform our multiply-adds.
			"vpmadd231d\t%%zmm9,\t%%zmm5,\t%%zmm7\n\t"
			"vpmadd231d\t%%zmm10,\t%%zmm6,\t%%zmm7\n\t"
			"vpmadd231d\t%%zmm11,\t%%zmm6,\t%%zmm7\n\t"
			"jl\t1b\n\t"

			"vbroadcastss\t(%[Y]),\t%%zmm3\n\t"                 // Load the scale factors coresponding to the two input vectors (y->d, x->d).
			"vbroadcastss\t(%[X])%{float16%},\t%%zmm4\n\t"
			"vmulps\t%%zmm3,\t%%zmm4,\t%%zmm5\n\t"              // Prepare the factor we're going to multiply the result by..
			"vmovaps\t\t(%[RES]),\t%%zmm6\n\t"                  // Load our inital state from sum..
			"vcvtfxpntdq2ps\t$0,%%zmm7,\t%%zmm9\n\t"            // Convert our ints to floats.
			"vfmadd231ps\t%%zmm5,\t%%zmm9,\t%%zmm6\n\t"         // Perform a fused multiply add.
			"vmovaps\t\t%%zmm6,\t(%[RES])\n\t"                  // Save the result.
			:
			: [X]      "r" (x),
			  [Y]      "r" (y),
			  [SCALE]  "r" (scale),
			  [RES]    "r" (res),
			  [OFFSET] "m" (q8offset),
			  [MASK]   "m" (lowmask),
			  [BIT5]   "m" (bit5),
			  [M]      "m" (m),
			  [Z]      "m" (zero)
			: "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13", "zmm14", "zmm15", "cc", "ecx", "k1", "k2", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "memory");
}


// A function for getting the dot product of two vectors, one of 5 bit resolution, and one of 8.
// Used during inference, if your model prints "llama_model_loader: - type q5_K:  XXX tensors", and XXX is not zero. :)
void ggml_vec_dot_q5_K_q8_K(int n, float * restrict s, size_t bs, const void * restrict vx, size_t bx, const void * restrict vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    /* Interpret X and Y as vectors. */
    const block_q5_K * restrict x = vx;
    const block_q8_K * restrict y = vy;

    /* The number of blocks we will process this in. */
    const int nb = n / QK_K;

    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];

    const uint8_t * scales = (const uint8_t*)&utmp[0];
    const uint8_t * mins   = (const uint8_t*)&utmp[2];

    float32x16_t sums;

    // Clear sums.
    GGML_F32x16_VEC_ZERO(&sums);

    float sumf = 0;

    for (int i = 0; i < nb; ++i) {

        // Extract scales and mins..
        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        int sumi = 0;

	// mins[k] applies to bsums[2k] and bsums[2k+1]: add those first, and halve the number of multiplies.
	for (int k = 0; k < QK_K/32; ++k) sumi += (y[i].bsums[2*k] + y[i].bsums[2*k+1]) * mins[k];

	// Unpack, multiply, scale, and accumulate this block, in one go.
	GGML_Q5K_Q8K_BLOCK_FMA(&x[i], &y[i], scales, &sums);

        const float dmin = ggml_fp16_to_fp32(x[i].dmin) * y[i].d;
        sumf -= dmin * sumi;
    }

    for (int l = 0; l < GGML_F32_EPR; ++l) sumf += ((float *)&sums)[l];
    *s = sumf;
}
