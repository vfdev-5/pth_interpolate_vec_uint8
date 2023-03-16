Diff template

- Source 1
```
#include <immintrin.h>
#include <cassert>
#include <cstring>

#define TORCH_INTERNAL_ASSERT assert
#define C10_RESTRICT __restrict
#define C10_UNLIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 0))

using uint8_t = unsigned char;
using uint32_t = unsigned int;


static __m128i inline mm_cvtepu8_epi32(const uint8_t* C10_RESTRICT ptr) {
  return _mm_cvtepu8_epi32(_mm_cvtsi32_si128(*(int32_t*)ptr));
}

void ImagingResampleHorizontalConvolution8u(
    uint8_t* C10_RESTRICT lineOut,
    int64_t out_xsize,
    const uint8_t* C10_RESTRICT lineIn,
    int64_t in_xsize,
    const int64_t* idx_ptr_xmin,
    const int64_t* idx_ptr_size,
    const int16_t* kk,
    int kmax,
    unsigned int coefs_precision,
    int64_t num_channels,
    bool is_last_line) {

  // Interpolation horizontal pass processing only one vertical line.
  // - Input data format is RGBA with R,G,B,A being uint8, we can encode 4 values as a single uint32 value.
  // - We split the size of weight vector for a given output index as a sum: K = n * 8 + m * 4 + k * 2 + l.
  // - We load and process 8 weights values in a loop ("block 8") then 4 weights and 2 weights values in
  // in another loops ("block 4" and "block 2") and finally we process 1 weight value in the final loop ("block 1").

  // Define various shuffling masks
  const auto kmask_low = _mm256_set_epi8(
      11, 10, 9, 8, 11, 10, 9, 8, 11, 10, 9, 8, 11, 10, 9, 8,
      3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0);
  const auto kmask_high = _mm256_set_epi8(
      15, 14, 13, 12, 15, 14, 13, 12, 15, 14, 13, 12, 15, 14, 13, 12,
      7, 6, 5, 4, 7, 6, 5, 4, 7, 6, 5, 4, 7, 6, 5, 4);
  const auto kmask_hl = _mm256_set_epi8(
      7, 6, 5, 4, 7, 6, 5, 4, 7, 6, 5, 4, 7, 6, 5, 4,
      3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0);

  const __m256i masks_low_high_c4[2] = {
    _mm256_set_epi8(
      -1, 7, -1, 3, -1, 6, -1, 2, -1, 5, -1, 1, -1, 4, -1, 0,
      -1, 7, -1, 3, -1, 6, -1, 2, -1, 5, -1, 1, -1, 4, -1, 0
    ),
    _mm256_set_epi8(
      -1, 15, -1, 11, -1, 14, -1, 10, -1, 13, -1, 9, -1, 12, -1, 8,
      -1, 15, -1, 11, -1, 14, -1, 10, -1, 13, -1, 9, -1, 12, -1, 8
    )
  };
  const __m256i masks_low_high_c3[2] = {
    _mm256_set_epi8(
      -1, -1, -1, -1, -1, 5, -1, 2, -1, 4, -1, 1, -1, 3, -1, 0,
      -1, -1, -1, -1, -1, 5, -1, 2, -1, 4, -1, 1, -1, 3, -1, 0
    ),
    _mm256_set_epi8(
      -1, -1, -1, -1, -1, 11, -1, 8, -1, 10, -1, 7, -1, 9, -1, 6,
      -1, -1, -1, -1, -1, 11, -1, 8, -1, 10, -1, 7, -1, 9, -1, 6
    )
  };
  const __m256i masks_lh_c3_c4[2] = {
    _mm256_set_epi8(
      -1, -1, -1, -1, -1, 11, -1, 8, -1, 10, -1, 7, -1, 9, -1, 6,
      -1, -1, -1, -1, -1, 5, -1, 2, -1, 4, -1, 1, -1, 3, -1, 0
    ),
    _mm256_set_epi8(
      -1, 15, -1, 11, -1, 14, -1, 10, -1, 13, -1, 9, -1, 12, -1, 8,
      -1, 7, -1, 3, -1, 6, -1, 2, -1, 5, -1, 1, -1, 4, -1, 0
    )
  };

  const __m128i masks_low128_c3_c4[2] = {
    _mm_set_epi8(
      -1, -1, -1, -1, -1, 5, -1, 2, -1, 4, -1, 1, -1, 3, -1, 0
    ),
    _mm_set_epi8(
      -1, 7, -1, 3, -1, 6, -1, 2, -1, 5, -1, 1, -1, 4, -1, 0
    )
  };

  const auto mask_low = (num_channels == 3) ? masks_low_high_c3[0] : masks_low_high_c4[0];
  const auto mask_high = (num_channels == 3) ? masks_low_high_c3[1] : masks_low_high_c4[1];
  const auto mask_hl = (num_channels == 3) ? masks_lh_c3_c4[0] : masks_lh_c3_c4[1];
  const auto mask_low128 = (num_channels == 3) ? masks_low128_c3_c4[0] : masks_low128_c3_c4[1];

  // out_xsize = output width, out_x = output x index
  // ids_size = interpolation size
  // ids_min = input x start index corresponding to output x index (out_x)

  const auto stride = num_channels * 1;  // num channels * sizeof(uint8)
  const auto zero = _mm_setzero_si128();

  TORCH_INTERNAL_ASSERT(stride == 3 || stride == 4);

  // Precompute xmax limits for block 8, block 4 and block 2
  // lineIn + stride * (x + xmin) + 32 <= lineIn + stride * (xmax + xmin)
  // --> x <= xmax - 32.0 / stride
  // Strict boundary:
  // --> x < xmax + 1 - int(ceil(32.0 / stride)) = xmax - b8_delta
  // Soft boundary for reading inside the buffer except its boundaries:
  // --> x < xmax + 1 - int(32.0 / stride) = xmax - b8_delta_soft
  // RGBA: b8_delta = b8_delta_soft = 7
  // RGB : b8_delta = 10
  // RGB : b8_delta_soft = 9
  const auto b8_delta = (stride == 4) ? 7 : ((is_last_line) ? 10 : 9);

  // lineIn + stride * (x + xmin) + 16 <= lineIn0 + stride * (xmax + xmin)
  // --> x <= xmax - 16.0 / stride
  // Strict boundary:
  // --> x < xmax + 1 - int(ceil(16.0 / stride)) = xmax - b4_delta
  // Soft boundary for reading inside the buffer except its boundaries:
  // --> x < xmax + 1 - int(16.0 / stride) = xmax - b4_delta_soft
  // RGBA: b4_delta = b4_delta_soft = 3
  // RGB : b4_delta = 5
  // RGB : b4_delta_soft = 4
  const auto b4_delta = (stride == 4) ? 3 : ((is_last_line) ? 5 : 4);

  // lineIn0 + stride * (x + xmin) + 8 <= lineIn0 + stride * (xmax + xmin)
  // --> x <= xmax - 8.0 / stride
  // Strict boundary:
  // --> x < xmax + 1 - int(ceil(8.0 / stride)) = xmax - b2_delta
  // Soft boundary for reading inside the buffer except its boundaries:
  // --> x < xmax + 1 - int(8.0 / stride) = xmax - b2_delta_soft
  // RGBA: b2_delta = b2_delta_soft = 1
  // RGB : b2_delta = 2
  // RGB : b2_delta_soft = 1
  const auto b2_delta = (stride == 4) ? 1 : ((is_last_line) ? 2 : 1);

  const auto max_out_x_strided = out_xsize * stride;
  const auto max_in_x_strided = in_xsize * stride;

//   for (const auto out_x : c10::irange(out_xsize)) {
  for (auto out_x=0; out_x < out_xsize; out_x++) {
    __m128i sss;
    const auto ids_min = idx_ptr_xmin[out_x];
    const auto ids_size = idx_ptr_size[out_x];
    const auto * k = &kk[out_x * kmax];
    int64_t i = 0;

    const auto * lineIn_min = lineIn + ids_min;

    if (ids_size < 8) {
      sss = _mm_set1_epi32(1 << (coefs_precision - 1));
    } else {
      // Lower part will be added to higher, use only half of the error
      auto sss256 = _mm256_set1_epi32(1 << (coefs_precision - 2));

      // block 8
      for (; i < ids_size - b8_delta; i += 8) {
        // Load 8 values from weight vector
        auto tmp = _mm_loadu_si128((__m128i*)&k[i]);
        // ksource = [
        //    wl_0 wh_0 wl_1 wh_1  wl_2 wh_2 wl_3 wh_3  wl_4 wh_4 wl_5 wh_5  wl_6 wh_6 wl_7 wh_7
        //    wl_0 wh_0 wl_1 wh_1  wl_2 wh_2 wl_3 wh_3  wl_4 wh_4 wl_5 wh_5  wl_6 wh_6 wl_7 wh_7
        // ]
        auto ksource = _mm256_insertf128_si256(_mm256_castsi128_si256(tmp), tmp, 1);

        // RGBA: Load 8 pixels from input:
        // source = [
        //    r0 g0 b0 a0  r1 g1 b1 a1  r2 g2 b2 a2  r3 g3 b3 a3
        //    r4 g4 b4 a4  r5 g5 b5 a5  r6 g6 b6 a6  r7 g7 b7 a7
        // ]
        // RGB: Load 10 pixels from input (however we can process only 8 pixels):
        // source = [
        //    r0 g0 b0 r1  g1 b1 r2 g2  b2 r3 g3 b3  r4 g4 b4 r5
        //    r4 g4 b4 r5  g5 b5 r6 g6  b6 r7 g7 b7  r8 g8 b8 r9
        // ]
        auto source = _mm256_inserti128_si256(_mm256_castsi128_si256(
            _mm_loadu_si128((__m128i *) (lineIn_min + stride * i))),
            _mm_loadu_si128((__m128i *) (lineIn_min + stride * (i + 4))), 1);

        // Extract lower part of each lane, cast to epi16 and reoder RGBARGBA -> RRGGBBAA
        // RGBA: pix1 = [
        //   r0 0 r1 0  g0 0 g1 0  b0 0 b1 0  a0 0 a1 0
        //   r4 0 r5 0  g4 0 g5 0  b4 0 b5 0  a4 0 a5 0
        // ]
        // RGB: pix1 = [
        //   r0 0 r1 0  g0 0 g1 0  b0 0 b1 0  0 0 0 0
        //   r4 0 r5 0  g4 0 g5 0  b4 0 b5 0  0 0 0 0
        // ]
        auto pix1 = _mm256_shuffle_epi8(source, mask_low);
        // mmk1 = [
        //   wl_0 wh_0 wl_1 wh_1  wl_0 wh_0 wl_1 wh_1  ...  ...
        //   wl_4 wh_4 wl_5 wh_5  wl_4 wh_4 wl_5 wh_5  ...  ...
        // ]
        auto mmk1 = _mm256_shuffle_epi8(ksource, kmask_low);
        // Compute output value as
        //   C += w0 * C0 + w1 * C1
        //   C += w4 * C4 + w5 * C5 for each channel in 32-bit precision
        sss256 = _mm256_add_epi32(sss256, _mm256_madd_epi16(pix1, mmk1));

        // Same as above for higher part of each lane
        auto pix2 = _mm256_shuffle_epi8(source, mask_high);
        auto mmk2 = _mm256_shuffle_epi8(ksource, kmask_high);
        // Compute output value as
        //    C += w2 * C2 + w3 * C3
        //    C += w6 * C6 + w7 * C7 for each channel in 32-bit precision
        sss256 = _mm256_add_epi32(sss256, _mm256_madd_epi16(pix2, mmk2));
      }

      // block 4
      for (; i < ids_size - b4_delta; i += 4) {
        // Load 4 values from weight vector
        auto tmp = _mm_loadl_epi64((__m128i *) &k[i]);
        // ksource = [
        //    wl_0 wh_0 wl_1 wh_1  wl_2 wh_2 wl_3 wh_3  0 0 0 0  0 0 0 0
        //    wl_0 wh_0 wl_1 wh_1  wl_2 wh_2 wl_3 wh_3  0 0 0 0  0 0 0 0
        // ]
        auto ksource = _mm256_insertf128_si256(_mm256_castsi128_si256(tmp), tmp, 1);

        // Load pixels from input line
        tmp = _mm_loadu_si128((__m128i *) (lineIn_min + stride * i));
        // RGBA: source = [
        //   r0 g0 b0 a0  r1 g1 b1 a1  r2 g2 b2 a2  r3 g3 b3 a3
        //   r0 g0 b0 a0  r1 g1 b1 a1  r2 g2 b2 a2  r3 g3 b3 a3
        // ]
        // RGB: source = [
        //   r0 g0 b0 r1  g1 b1 r2 g2  b2 r3 g3 b3  r4 g4 b4 r5
        //   r0 g0 b0 r1  g1 b1 r2 g2  b2 r3 g3 b3  r4 g4 b4 r5
        // ]
        auto source = _mm256_insertf128_si256(_mm256_castsi128_si256(tmp), tmp, 1);

        // Cast source to epi16 and reorder RGBARGBA -> RRGGBBAA
        // RGBA: pix = [
        //   r0 0 r1 0  g0 0 g1 0  b0 0 b1 0  a0 0 a1 0
        //   r2 0 r3 0  g2 0 g3 0  b2 0 b3 0  a2 0 a3 0
        // ]
        // RGB: pix = [
        //   r0 0 r1 0  g0 0 g1 0  b0 0 b1 0  0 0 0 0
        //   r2 0 r3 0  g2 0 g3 0  b2 0 b3 0  0 0 0 0
        // ]
        auto pix = _mm256_shuffle_epi8(source, mask_hl);
        // mmk = [
        //   wl_0 wh_0 wl_1 wh_1  wl_0 wh_0 wl_1 wh_1  ... ...
        //   wl_2 wh_2 wl_3 wh_3  wl_2 wh_2 wl_3 wh_3  ... ...
        // ]
        auto mmk = _mm256_shuffle_epi8(ksource, kmask_hl);
        // Compute output value as
        //   C += w0 * C0 + w1 * C1
        //   C += w2 * C2 + w3 * C3 for each channel in 32-bit precision
        sss256 = _mm256_add_epi32(sss256, _mm256_madd_epi16(pix, mmk));
      }

      // Sum results between the lanes
      sss = _mm_add_epi32(
          _mm256_extracti128_si256(sss256, 0),
          _mm256_extracti128_si256(sss256, 1));
    }

    // block 2
    for (; i < ids_size - b2_delta; i += 2) {
      // Load 2 values from weight vector
      // mmk = [wl_0 wh_0 wl_1 wh_1  wl_0 wh_0 wl_1 wh_1  ...]
      auto mmk = _mm_set1_epi32(*(int32_t*)&k[i]);
      // Load pixels from input line
      // RGBA: source = [
      //   r0 g0 b0 a0  r1 g1 b1 a1  0 0 0 0  0 0 0 0
      // ]
      // RGB: source = [
      //   r0 g0 b0 r1  g1 b1 r2 g2  0 0 0 0  0 0 0 0
      // ]
      auto source = _mm_loadl_epi64((__m128i *) (lineIn_min + stride * i));
      // Cast source to epi16 and reorder RGBARGBA -> RRGGBBAA
      auto pix = _mm_shuffle_epi8(source, mask_low128);
      // Compute output value as C += w0 * C0 + w1 * C1 for each channel in 32-bit precision
      sss = _mm_add_epi32(sss, _mm_madd_epi16(pix, mmk));
    }

    // block 1
    for (; i < ids_size - 1; i++) {
      // Load 1 value from weight vector
      // mmk = [wl_0 wh_0 0 0  wl_0 wh_0 0 0  ...]
      auto mmk = _mm_set1_epi32(k[i]);
      // Load one pixel from input line
      // RGBA: pix = [
      //   r0 0 0 0  g0 0 0 0  b0 0 0 0  a0 0 0 0
      // ]
      // RGB: pix = [
      //   r0 0 0 0  g0 0 0 0  b0 0 0 0  r1 0 0 0
      // ]
      auto pix = mm_cvtepu8_epi32(lineIn_min + stride * i);
      // Compute output value as C += w0 * C0 for each channel in 32-bit precision
      sss = _mm_add_epi32(sss, _mm_madd_epi16(pix, mmk));
    }

    if (i == ids_size - 1) {
      // last element
      auto mmk = _mm_set1_epi32(k[i]);
      __m128i pix;
      auto p = lineIn_min + stride * i;
      if (num_channels == 3 && C10_UNLIKELY(is_last_line && ids_min + stride * i + 4 >= max_in_x_strided)) {
        uint8_t output[4];
        std::memcpy(output, p, 3);
        pix = mm_cvtepu8_epi32(output);
      } else {
        pix = mm_cvtepu8_epi32(p);
      }
      sss = _mm_add_epi32(sss, _mm_madd_epi16(pix, mmk));
    }

    // Shift back the output integer values: output = output >> weights_precision
    sss = _mm_srai_epi32(sss, coefs_precision);
    // Convert packed signed 32-bit integers to packed 16-bit integers using signed saturation
    // (a a a a b b b b | c c c c d d d d) -> (a a b b c c d d | 0 0 0 0 0 0 0 0)
    sss = _mm_packs_epi32(sss, zero);
    // Convert packed signed 16-bit integers to packed 8-bit integers using unsigned saturation
    // (a a b b c c d d) -> (a b c d 0 0 0 0)
    sss = _mm_packus_epi16(sss, zero);
    // Write the output into single uint32
    // (a b c d) -> x_uint32
    auto o = _mm_cvtsi128_si32(sss);
    const auto out_x_strided = stride * out_x;
    if (num_channels == 3 && C10_UNLIKELY(out_x_strided + 4 >= max_out_x_strided)) {
      if (C10_UNLIKELY(is_last_line)) {
        // When we handle the last line, we can not access the next 4 bytes
        // as they are out of memory bounds.
        std::memcpy(lineOut + out_x_strided, (uint8_t *) &o, 3);
      } else {
        // This is a boundary case when we want to write 4 bytes to the output buffer but
        // the 4th bytes is already computed. It means that we can not overwrite it.
        //               v----------|
        // Output = [... X1 X2 X3 | A B C D ...]
        // First, we store store next 4 bytes (A B C D)
        // Second, we write 4 bytes to (X1 X2 X3 | A) -> (U V W | Z)
        //          [... U V W | Z B C D ...]
        // Third, we overwrite next 4 bytes (Z B C D) with stored values (A B C D)

        char next[4];
        std::memcpy(next, lineOut + out_x_strided + stride, 4);
        std::memcpy(lineOut + out_x_strided, (uint8_t *) &o, 4);
        std::memcpy(lineOut + out_x_strided + stride, next, 4);
      }
    } else if (num_channels == 3) {
      // We simply write 4 bytes (... R G B X 0 0 0 0 0 ...) where X is a garbage value
      // that we will overwrite on the next iteration: (... R G B R G B X 0 0 ...)
      std::memcpy(lineOut + out_x_strided, (uint8_t *) &o, 4);
    } else {
      // num_channels = 4 -> lineOut + out_x_strided should be uint32 aligned
      *(uint32_t *)(lineOut + out_x_strided) = o;
    }
  }
}

void foo(
    uint8_t* C10_RESTRICT lineOut,
    int64_t out_xsize,
    const uint8_t* C10_RESTRICT lineIn,
    int64_t in_xsize,
    const int64_t* idx_ptr_xmin,
    const int64_t* idx_ptr_size,
    const int16_t* kk,
    int kmax,
    unsigned int coefs_precision,
    bool is_last_line
) {
    constexpr int num_channels = 3;

    ImagingResampleHorizontalConvolution8u(
        lineOut,
        out_xsize,
        lineIn,
        in_xsize,
        idx_ptr_xmin,
        idx_ptr_size,
        kk,
        kmax,
        coefs_precision,
        num_channels,
        is_last_line);
}
```

- Source 2
```
#include <immintrin.h>
#include <cassert>
#include <cstring>

#define TORCH_INTERNAL_ASSERT assert
#define C10_RESTRICT __restrict
#define C10_UNLIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 0))

using uint8_t = unsigned char;
using uint32_t = unsigned int;


static __m128i inline mm_cvtepu8_epi32(const uint8_t* C10_RESTRICT ptr) {
  return _mm_cvtepu8_epi32(_mm_cvtsi32_si128(*(int32_t*)ptr));
}

void ImagingResampleHorizontalConvolution8u(
    uint8_t* C10_RESTRICT lineOut,
    int64_t out_xsize,
    const uint8_t* C10_RESTRICT lineIn,
    int64_t in_xsize,
    const int64_t* idx_ptr_xmin,
    const int64_t* idx_ptr_size,
    const int16_t* kk,
    int kmax,
    unsigned int coefs_precision,
    int64_t num_channels,
    bool is_last_line) {

  // Interpolation horizontal pass processing only one vertical line.
  // - Input data format is RGBA with R,G,B,A being uint8, we can encode 4 values as a single uint32 value.
  // - We split the size of weight vector for a given output index as a sum: K = n * 8 + m * 4 + k * 2 + l.
  // - We load and process 8 weights values in a loop ("block 8") then 4 weights and 2 weights values in
  // in another loops ("block 4" and "block 2") and finally we process 1 weight value in the final loop ("block 1").

  // Define various shuffling masks
  const auto kmask_low = _mm256_set_epi8(
      11, 10, 9, 8, 11, 10, 9, 8, 11, 10, 9, 8, 11, 10, 9, 8,
      3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0);
  const auto kmask_high = _mm256_set_epi8(
      15, 14, 13, 12, 15, 14, 13, 12, 15, 14, 13, 12, 15, 14, 13, 12,
      7, 6, 5, 4, 7, 6, 5, 4, 7, 6, 5, 4, 7, 6, 5, 4);
  const auto kmask_hl = _mm256_set_epi8(
      7, 6, 5, 4, 7, 6, 5, 4, 7, 6, 5, 4, 7, 6, 5, 4,
      3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0);

  const __m256i masks_low_high_c4[2] = {
    _mm256_set_epi8(
      -1, 7, -1, 3, -1, 6, -1, 2, -1, 5, -1, 1, -1, 4, -1, 0,
      -1, 7, -1, 3, -1, 6, -1, 2, -1, 5, -1, 1, -1, 4, -1, 0
    ),
    _mm256_set_epi8(
      -1, 15, -1, 11, -1, 14, -1, 10, -1, 13, -1, 9, -1, 12, -1, 8,
      -1, 15, -1, 11, -1, 14, -1, 10, -1, 13, -1, 9, -1, 12, -1, 8
    )
  };
  const __m256i masks_low_high_c3[2] = {
    _mm256_set_epi8(
      -1, -1, -1, -1, -1, 5, -1, 2, -1, 4, -1, 1, -1, 3, -1, 0,
      -1, -1, -1, -1, -1, 5, -1, 2, -1, 4, -1, 1, -1, 3, -1, 0
    ),
    _mm256_set_epi8(
      -1, -1, -1, -1, -1, 11, -1, 8, -1, 10, -1, 7, -1, 9, -1, 6,
      -1, -1, -1, -1, -1, 11, -1, 8, -1, 10, -1, 7, -1, 9, -1, 6
    )
  };
  const __m256i masks_lh_c3_c4[2] = {
    _mm256_set_epi8(
      -1, -1, -1, -1, -1, 11, -1, 8, -1, 10, -1, 7, -1, 9, -1, 6,
      -1, -1, -1, -1, -1, 5, -1, 2, -1, 4, -1, 1, -1, 3, -1, 0
    ),
    _mm256_set_epi8(
      -1, 15, -1, 11, -1, 14, -1, 10, -1, 13, -1, 9, -1, 12, -1, 8,
      -1, 7, -1, 3, -1, 6, -1, 2, -1, 5, -1, 1, -1, 4, -1, 0
    )
  };

  const __m128i masks_low128_c3_c4[2] = {
    _mm_set_epi8(
      -1, -1, -1, -1, -1, 5, -1, 2, -1, 4, -1, 1, -1, 3, -1, 0
    ),
    _mm_set_epi8(
      -1, 7, -1, 3, -1, 6, -1, 2, -1, 5, -1, 1, -1, 4, -1, 0
    )
  };

  const auto mask_low = (num_channels == 3) ? masks_low_high_c3[0] : masks_low_high_c4[0];
  const auto mask_high = (num_channels == 3) ? masks_low_high_c3[1] : masks_low_high_c4[1];
  const auto mask_hl = (num_channels == 3) ? masks_lh_c3_c4[0] : masks_lh_c3_c4[1];
  const auto mask_low128 = (num_channels == 3) ? masks_low128_c3_c4[0] : masks_low128_c3_c4[1];

  // out_xsize = output width, out_x = output x index
  // ids_size = interpolation size
  // ids_min = input x start index corresponding to output x index (out_x)

  const auto stride = num_channels * 1;  // num channels * sizeof(uint8)
  const auto zero = _mm_setzero_si128();

  TORCH_INTERNAL_ASSERT(stride == 3 || stride == 4);

  // Precompute xmax limits for block 8, block 4 and block 2
  // lineIn + stride * (x + xmin) + 32 <= lineIn + stride * (xmax + xmin)
  // --> x <= xmax - 32.0 / stride
  // Strict boundary:
  // --> x < xmax + 1 - int(ceil(32.0 / stride)) = xmax - b8_delta
  // Soft boundary for reading inside the buffer except its boundaries:
  // --> x < xmax + 1 - int(32.0 / stride) = xmax - b8_delta_soft
  // RGBA: b8_delta = b8_delta_soft = 7
  // RGB : b8_delta = 10
  // RGB : b8_delta_soft = 9
  const auto b8_delta = (stride == 4) ? 7 : ((is_last_line) ? 10 : 9);

  // lineIn + stride * (x + xmin) + 16 <= lineIn0 + stride * (xmax + xmin)
  // --> x <= xmax - 16.0 / stride
  // Strict boundary:
  // --> x < xmax + 1 - int(ceil(16.0 / stride)) = xmax - b4_delta
  // Soft boundary for reading inside the buffer except its boundaries:
  // --> x < xmax + 1 - int(16.0 / stride) = xmax - b4_delta_soft
  // RGBA: b4_delta = b4_delta_soft = 3
  // RGB : b4_delta = 5
  // RGB : b4_delta_soft = 4
  const auto b4_delta = (stride == 4) ? 3 : ((is_last_line) ? 5 : 4);

  // lineIn0 + stride * (x + xmin) + 8 <= lineIn0 + stride * (xmax + xmin)
  // --> x <= xmax - 8.0 / stride
  // Strict boundary:
  // --> x < xmax + 1 - int(ceil(8.0 / stride)) = xmax - b2_delta
  // Soft boundary for reading inside the buffer except its boundaries:
  // --> x < xmax + 1 - int(8.0 / stride) = xmax - b2_delta_soft
  // RGBA: b2_delta = b2_delta_soft = 1
  // RGB : b2_delta = 2
  // RGB : b2_delta_soft = 1
  const auto b2_delta = (stride == 4) ? 1 : ((is_last_line) ? 2 : 1);

  const auto max_out_x_strided = out_xsize * stride;
  const auto max_in_x_strided = in_xsize * stride;

//   for (const auto out_x : c10::irange(out_xsize)) {
  for (auto out_x=0; out_x < out_xsize; out_x++) {
    __m128i sss;
    const auto ids_min = idx_ptr_xmin[out_x];
    const auto ids_size = idx_ptr_size[out_x];
    const auto * k = &kk[out_x * kmax];
    int64_t i = 0;

    const auto * lineIn_min = lineIn + ids_min;

    if (ids_size < 8) {
      sss = _mm_set1_epi32(1 << (coefs_precision - 1));
    } else {
      // Lower part will be added to higher, use only half of the error
      auto sss256 = _mm256_set1_epi32(1 << (coefs_precision - 2));

      // block 8
      for (; i < ids_size - b8_delta; i += 8) {
        // Load 8 values from weight vector
        auto tmp = _mm_loadu_si128((__m128i*)&k[i]);
        // ksource = [
        //    wl_0 wh_0 wl_1 wh_1  wl_2 wh_2 wl_3 wh_3  wl_4 wh_4 wl_5 wh_5  wl_6 wh_6 wl_7 wh_7
        //    wl_0 wh_0 wl_1 wh_1  wl_2 wh_2 wl_3 wh_3  wl_4 wh_4 wl_5 wh_5  wl_6 wh_6 wl_7 wh_7
        // ]
        auto ksource = _mm256_insertf128_si256(_mm256_castsi128_si256(tmp), tmp, 1);

        // RGBA: Load 8 pixels from input:
        // source = [
        //    r0 g0 b0 a0  r1 g1 b1 a1  r2 g2 b2 a2  r3 g3 b3 a3
        //    r4 g4 b4 a4  r5 g5 b5 a5  r6 g6 b6 a6  r7 g7 b7 a7
        // ]
        // RGB: Load 10 pixels from input (however we can process only 8 pixels):
        // source = [
        //    r0 g0 b0 r1  g1 b1 r2 g2  b2 r3 g3 b3  r4 g4 b4 r5
        //    r4 g4 b4 r5  g5 b5 r6 g6  b6 r7 g7 b7  r8 g8 b8 r9
        // ]
        auto source = _mm256_inserti128_si256(_mm256_castsi128_si256(
            _mm_loadu_si128((__m128i *) (lineIn_min + stride * i))),
            _mm_loadu_si128((__m128i *) (lineIn_min + stride * (i + 4))), 1);

        // Extract lower part of each lane, cast to epi16 and reoder RGBARGBA -> RRGGBBAA
        // RGBA: pix1 = [
        //   r0 0 r1 0  g0 0 g1 0  b0 0 b1 0  a0 0 a1 0
        //   r4 0 r5 0  g4 0 g5 0  b4 0 b5 0  a4 0 a5 0
        // ]
        // RGB: pix1 = [
        //   r0 0 r1 0  g0 0 g1 0  b0 0 b1 0  0 0 0 0
        //   r4 0 r5 0  g4 0 g5 0  b4 0 b5 0  0 0 0 0
        // ]
        auto pix1 = _mm256_shuffle_epi8(source, mask_low);
        // mmk1 = [
        //   wl_0 wh_0 wl_1 wh_1  wl_0 wh_0 wl_1 wh_1  ...  ...
        //   wl_4 wh_4 wl_5 wh_5  wl_4 wh_4 wl_5 wh_5  ...  ...
        // ]
        auto mmk1 = _mm256_shuffle_epi8(ksource, kmask_low);
        // Compute output value as
        //   C += w0 * C0 + w1 * C1
        //   C += w4 * C4 + w5 * C5 for each channel in 32-bit precision
        sss256 = _mm256_add_epi32(sss256, _mm256_madd_epi16(pix1, mmk1));

        // Same as above for higher part of each lane
        auto pix2 = _mm256_shuffle_epi8(source, mask_high);
        auto mmk2 = _mm256_shuffle_epi8(ksource, kmask_high);
        // Compute output value as
        //    C += w2 * C2 + w3 * C3
        //    C += w6 * C6 + w7 * C7 for each channel in 32-bit precision
        sss256 = _mm256_add_epi32(sss256, _mm256_madd_epi16(pix2, mmk2));
      }

      // block 4
      for (; i < ids_size - b4_delta; i += 4) {
        // Load 4 values from weight vector
        auto tmp = _mm_loadl_epi64((__m128i *) &k[i]);
        // ksource = [
        //    wl_0 wh_0 wl_1 wh_1  wl_2 wh_2 wl_3 wh_3  0 0 0 0  0 0 0 0
        //    wl_0 wh_0 wl_1 wh_1  wl_2 wh_2 wl_3 wh_3  0 0 0 0  0 0 0 0
        // ]
        auto ksource = _mm256_insertf128_si256(_mm256_castsi128_si256(tmp), tmp, 1);

        // Load pixels from input line
        tmp = _mm_loadu_si128((__m128i *) (lineIn_min + stride * i));
        // RGBA: source = [
        //   r0 g0 b0 a0  r1 g1 b1 a1  r2 g2 b2 a2  r3 g3 b3 a3
        //   r0 g0 b0 a0  r1 g1 b1 a1  r2 g2 b2 a2  r3 g3 b3 a3
        // ]
        // RGB: source = [
        //   r0 g0 b0 r1  g1 b1 r2 g2  b2 r3 g3 b3  r4 g4 b4 r5
        //   r0 g0 b0 r1  g1 b1 r2 g2  b2 r3 g3 b3  r4 g4 b4 r5
        // ]
        auto source = _mm256_insertf128_si256(_mm256_castsi128_si256(tmp), tmp, 1);

        // Cast source to epi16 and reorder RGBARGBA -> RRGGBBAA
        // RGBA: pix = [
        //   r0 0 r1 0  g0 0 g1 0  b0 0 b1 0  a0 0 a1 0
        //   r2 0 r3 0  g2 0 g3 0  b2 0 b3 0  a2 0 a3 0
        // ]
        // RGB: pix = [
        //   r0 0 r1 0  g0 0 g1 0  b0 0 b1 0  0 0 0 0
        //   r2 0 r3 0  g2 0 g3 0  b2 0 b3 0  0 0 0 0
        // ]
        auto pix = _mm256_shuffle_epi8(source, mask_hl);
        // mmk = [
        //   wl_0 wh_0 wl_1 wh_1  wl_0 wh_0 wl_1 wh_1  ... ...
        //   wl_2 wh_2 wl_3 wh_3  wl_2 wh_2 wl_3 wh_3  ... ...
        // ]
        auto mmk = _mm256_shuffle_epi8(ksource, kmask_hl);
        // Compute output value as
        //   C += w0 * C0 + w1 * C1
        //   C += w2 * C2 + w3 * C3 for each channel in 32-bit precision
        sss256 = _mm256_add_epi32(sss256, _mm256_madd_epi16(pix, mmk));
      }

      // Sum results between the lanes
      sss = _mm_add_epi32(
          _mm256_extracti128_si256(sss256, 0),
          _mm256_extracti128_si256(sss256, 1));
    }

    // block 2
    for (; i < ids_size - b2_delta; i += 2) {
      // Load 2 values from weight vector
      // mmk = [wl_0 wh_0 wl_1 wh_1  wl_0 wh_0 wl_1 wh_1  ...]
      auto mmk = _mm_set1_epi32(*(int32_t*)&k[i]);
      // Load pixels from input line
      // RGBA: source = [
      //   r0 g0 b0 a0  r1 g1 b1 a1  0 0 0 0  0 0 0 0
      // ]
      // RGB: source = [
      //   r0 g0 b0 r1  g1 b1 r2 g2  0 0 0 0  0 0 0 0
      // ]
      auto source = _mm_loadl_epi64((__m128i *) (lineIn_min + stride * i));
      // Cast source to epi16 and reorder RGBARGBA -> RRGGBBAA
      auto pix = _mm_shuffle_epi8(source, mask_low128);
      // Compute output value as C += w0 * C0 + w1 * C1 for each channel in 32-bit precision
      sss = _mm_add_epi32(sss, _mm_madd_epi16(pix, mmk));
    }

    // block 1
    for (; i < ids_size - 1; i++) {
      // Load 1 value from weight vector
      // mmk = [wl_0 wh_0 0 0  wl_0 wh_0 0 0  ...]
      auto mmk = _mm_set1_epi32(k[i]);
      // Load one pixel from input line
      // RGBA: pix = [
      //   r0 0 0 0  g0 0 0 0  b0 0 0 0  a0 0 0 0
      // ]
      // RGB: pix = [
      //   r0 0 0 0  g0 0 0 0  b0 0 0 0  r1 0 0 0
      // ]
      auto pix = mm_cvtepu8_epi32(lineIn_min + stride * i);
      // Compute output value as C += w0 * C0 for each channel in 32-bit precision
      sss = _mm_add_epi32(sss, _mm_madd_epi16(pix, mmk));
    }

    if (i == ids_size - 1) {
      // last element
      auto mmk = _mm_set1_epi32(k[i]);
      __m128i pix;
      auto p = lineIn_min + stride * i;
      if (num_channels == 3 && C10_UNLIKELY(is_last_line && ids_min + stride * i + 4 >= max_in_x_strided)) {
        uint8_t output[4];
        std::memcpy(output, p, 3);
        pix = mm_cvtepu8_epi32(output);
      } else {
        pix = mm_cvtepu8_epi32(p);
      }
      sss = _mm_add_epi32(sss, _mm_madd_epi16(pix, mmk));
    }

    // Shift back the output integer values: output = output >> weights_precision
    sss = _mm_srai_epi32(sss, coefs_precision);
    // Convert packed signed 32-bit integers to packed 16-bit integers using signed saturation
    // (a a a a b b b b | c c c c d d d d) -> (a a b b c c d d | 0 0 0 0 0 0 0 0)
    sss = _mm_packs_epi32(sss, zero);
    // Convert packed signed 16-bit integers to packed 8-bit integers using unsigned saturation
    // (a a b b c c d d) -> (a b c d 0 0 0 0)
    sss = _mm_packus_epi16(sss, zero);
    // Write the output into single uint32
    // (a b c d) -> x_uint32
    auto o = _mm_cvtsi128_si32(sss);
    const auto out_x_strided = stride * out_x;
    if (num_channels == 3 && C10_UNLIKELY(out_x_strided + 4 >= max_out_x_strided)) {
      if (C10_UNLIKELY(is_last_line)) {
        // When we handle the last line, we can not access the next 4 bytes
        // as they are out of memory bounds.
        std::memcpy(lineOut + out_x_strided, (uint8_t *) &o, 3);
      } else {
        // This is a boundary case when we want to write 4 bytes to the output buffer but
        // the 4th bytes is already computed. It means that we can not overwrite it.
        //               v----------|
        // Output = [... X1 X2 X3 | A B C D ...]
        // First, we store store next 4 bytes (A B C D)
        // Second, we write 4 bytes to (X1 X2 X3 | A) -> (U V W | Z)
        //          [... U V W | Z B C D ...]
        // Third, we overwrite next 4 bytes (Z B C D) with stored values (A B C D)

        char next[4];
        std::memcpy(next, lineOut + out_x_strided + stride, 4);
        std::memcpy(lineOut + out_x_strided, (uint8_t *) &o, 4);
        std::memcpy(lineOut + out_x_strided + stride, next, 4);
      }
    } else if (num_channels == 3) {
      // We simply write 4 bytes (... R G B X 0 0 0 0 0 ...) where X is a garbage value
      // that we will overwrite on the next iteration: (... R G B R G B X 0 0 ...)
      std::memcpy(lineOut + out_x_strided, (uint8_t *) &o, 4);
    } else {
      // num_channels = 4 -> lineOut + out_x_strided should be uint32 aligned
      *(uint32_t *)(lineOut + out_x_strided) = o;
    }
  }
}

void foo(
    uint8_t* C10_RESTRICT lineOut,
    int64_t out_xsize,
    const uint8_t* C10_RESTRICT lineIn,
    int64_t in_xsize,
    const int64_t* idx_ptr_xmin,
    const int64_t* idx_ptr_size,
    const int16_t* kk,
    int kmax,
    unsigned int coefs_precision,
    bool is_last_line
) {
    int num_channels = 3;

    ImagingResampleHorizontalConvolution8u(
        lineOut,
        out_xsize,
        lineIn,
        in_xsize,
        idx_ptr_xmin,
        idx_ptr_size,
        kk,
        kmax,
        coefs_precision,
        num_channels,
        is_last_line);
}
```

- Assembly for source 1
  - compiler: x86-64 GCC 9.4
  - Flags: -std=c++17 -O3 -mavx -mfma -mavx2 -mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store -Wno-deprecated -fvisibility-inlines-hidden -fopenmp -O3 -fPIC -Wno-narrowing -Wall -Wextra -Werror=return-type -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-sign-compare -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function -Wno-unused-result -Wno-unused-local-typedefs -Wno-strict-overflow -Wno-strict-aliasing -Wno-error=deprecated-declarations -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=redundant-decls -Wno-error=old-style-cast -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Werror=cast-function-type -Wno-stringop-overflow
```
.LC8:
        .string "void ImagingResampleHorizontalConvolution8u(uint8_t*, int64_t, const uint8_t*, int64_t, const int64_t*, const int64_t*, const int16_t*, int, unsigned int, int64_t, bool)"
.LC9:
        .string "/app/example.cpp"
.LC10:
        .string "stride == 3 || stride == 4"
ImagingResampleHorizontalConvolution8u(unsigned char*, long, unsigned char const*, long, long const*, long const*, short const*, int, unsigned int, long, bool):
        push    rbp
        mov     rbp, rsp
        push    r15
        push    r14
        push    r13
        push    r12
        push    rbx
        mov     rbx, rdi
        and     rsp, -32
        sub     rsp, 160
        mov     eax, DWORD PTR 32[rbp]
        mov     QWORD PTR 128[rsp], rsi
        mov     esi, DWORD PTR 48[rbp]
        mov     QWORD PTR 104[rsp], r8
        mov     r8, QWORD PTR 40[rbp]
        mov     QWORD PTR 112[rsp], rdx
        mov     QWORD PTR 96[rsp], r9
        mov     BYTE PTR 47[rsp], sil
        cmp     r8, 3
        je      .L2
        cmp     r8, 4
        jne     .L43
        vmovdqa xmm4, XMMWORD PTR .LC0[rip]
        vmovdqa ymm8, YMMWORD PTR .LC1[rip]
        mov     QWORD PTR 48[rsp], 3
        vmovdqa ymm7, YMMWORD PTR .LC2[rip]
        vmovdqa ymm6, YMMWORD PTR .LC3[rip]
        mov     QWORD PTR 120[rsp], 1
        mov     QWORD PTR 56[rsp], 7
.L3:
        mov     rdi, QWORD PTR 128[rsp]
        imul    rcx, r8
        mov     rdx, rdi
        imul    rdx, r8
        mov     QWORD PTR 32[rsp], rcx
        mov     QWORD PTR 80[rsp], rdx
        test    rdi, rdi
        jle     .L40
        mov     edx, 1
        lea     ecx, -2[rax]
        vmovd   xmm12, eax
        mov     r13, r8
        mov     edi, edx
        vmovdqa ymm11, YMMWORD PTR .LC13[rip]
        vmovdqa ymm10, YMMWORD PTR .LC11[rip]
        lea     r15, 0[0+r8*8]
        lea     r9, [r8+r8]
        vpxor   xmm15, xmm15, xmm15
        sal     edi, cl
        lea     ecx, -1[rax]
        vmovd   xmm5, edi
        sal     edx, cl
        cmp     r8, 3
        mov     eax, 4
        vpbroadcastd    ymm5, xmm5
        vmovdqa ymm9, YMMWORD PTR .LC12[rip]
        lea     r12, 0[0+r8*4]
        vpxor   xmm14, xmm14, xmm14
        vmovdqa YMMWORD PTR [rsp], ymm5
        vmovd   xmm5, edx
        sete    dl
        sub     rax, rbx
        and     edx, esi
        mov     QWORD PTR 64[rsp], rax
        neg     r13
        mov     rsi, QWORD PTR 16[rbp]
        mov     BYTE PTR 79[rsp], dl
        movsx   rdx, DWORD PTR 24[rbp]
        vpshufd xmm13, xmm5, 0
        xor     r10d, r10d
        sal     r13, 2
        lea     rdi, [rdx+rdx]
        mov     QWORD PTR 88[rsp], rdi
.L28:
        mov     rax, QWORD PTR 104[rsp]
        mov     r14, QWORD PTR [rax+r10*8]
        mov     rax, QWORD PTR 96[rsp]
        mov     rdi, QWORD PTR [rax+r10*8]
        mov     rax, QWORD PTR 112[rsp]
        lea     r11, [rax+r14]
        cmp     rdi, 7
        jg      .L5
        vmovdqa xmm0, xmm13
        xor     eax, eax
.L15:
        mov     rcx, rdi
        sub     rcx, QWORD PTR 120[rsp]
        cmp     rcx, rax
        jle     .L6
        mov     rdx, r8
        imul    rdx, rax
        add     rdx, r11
.L18:
        vmovq   xmm1, QWORD PTR [rdx]
        vbroadcastss    xmm2, DWORD PTR [rsi+rax*2]
        add     rax, 2
        add     rdx, r9
        vpshufb xmm1, xmm1, xmm4
        vpmaddwd        xmm1, xmm1, xmm2
        vpaddd  xmm0, xmm1, xmm0
        cmp     rcx, rax
        jg      .L18
.L6:
        sub     rdi, 1
        cmp     rdi, rax
        jle     .L44
        mov     rcx, r8
        mov     QWORD PTR 136[rsp], rdi
        imul    rcx, rax
        add     rcx, r11
.L19:
        movsx   edx, WORD PTR [rsi+rax*2]
        vpmovzxbd       xmm1, DWORD PTR [rcx]
        add     rax, 1
        add     rcx, r8
        vmovd   xmm5, edx
        vpshufd xmm2, xmm5, 0
        vpmaddwd        xmm1, xmm1, xmm2
        vpaddd  xmm0, xmm1, xmm0
        cmp     rax, rdi
        jne     .L19
        mov     rdx, QWORD PTR 136[rsp]
.L16:
        cmp     rdi, rdx
        je      .L45
.L20:
        vpsrad  xmm0, xmm0, xmm12
        vpackssdw       xmm0, xmm0, xmm15
        vpackuswb       xmm0, xmm0, xmm14
        vmovd   DWORD PTR 156[rsp], xmm0
        vmovd   eax, xmm0
        cmp     r8, 3
        je      .L46
.L23:
        mov     DWORD PTR [rbx], eax
.L26:
        add     r10, 1
        add     rbx, r8
        add     rsi, QWORD PTR 88[rsp]
        cmp     QWORD PTR 128[rsp], r10
        jne     .L28
.L40:
        vzeroupper
        lea     rsp, -40[rbp]
        pop     rbx
        pop     r12
        pop     r13
        pop     r14
        pop     r15
        pop     rbp
        ret
.L46:
        mov     rdi, QWORD PTR 64[rsp]
        lea     rdx, [rdi+rbx]
        cmp     QWORD PTR 80[rsp], rdx
        jg      .L23
        cmp     BYTE PTR 47[rsp], 0
        jne     .L47
        mov     edx, DWORD PTR 3[rbx]
        vmovd   DWORD PTR [rbx], xmm0
        mov     DWORD PTR 3[rbx], edx
        jmp     .L26
.L45:
        movsx   eax, WORD PTR [rsi+rdi*2]
        imul    rdi, r8
        vmovd   xmm5, eax
        add     r11, rdi
        cmp     BYTE PTR 79[rsp], 0
        vpshufd xmm2, xmm5, 0
        jne     .L48
.L21:
        vpmovzxbd       xmm1, DWORD PTR [r11]
.L22:
        vpmaddwd        xmm1, xmm1, xmm2
        vpaddd  xmm0, xmm1, xmm0
        jmp     .L20
.L5:
        mov     rcx, rdi
        sub     rcx, QWORD PTR 56[rsp]
        test    rcx, rcx
        jle     .L49
        vmovdqa ymm0, YMMWORD PTR [rsp]
        lea     rdx, [r11+r12]
        xor     eax, eax
.L13:
        vmovdqu xmm3, XMMWORD PTR [rsi+rax*2]
        vmovdqu xmm1, XMMWORD PTR [rdx+r13]
        vinserti128     ymm3, ymm3, XMMWORD PTR [rsi+rax*2], 0x1
        vinserti128     ymm1, ymm1, XMMWORD PTR [rdx], 0x1
        add     rax, 8
        add     rdx, r15
        vpshufb ymm5, ymm1, ymm7
        vpshufb ymm2, ymm3, ymm10
        vpshufb ymm1, ymm1, ymm6
        vpmaddwd        ymm2, ymm5, ymm2
        vpshufb ymm3, ymm3, ymm9
        vpmaddwd        ymm1, ymm1, ymm3
        vpaddd  ymm0, ymm2, ymm0
        vpaddd  ymm0, ymm0, ymm1
        cmp     rcx, rax
        jg      .L13
        mov     rcx, rdi
        sub     rcx, QWORD PTR 48[rsp]
        cmp     rax, rcx
        jge     .L10
.L11:
        mov     rdx, rax
        imul    rdx, r8
        add     rdx, r11
.L14:
        vmovq   xmm1, QWORD PTR [rsi+rax*2]
        add     rax, 4
        vmovdqa xmm2, xmm1
        vinserti128     ymm2, ymm2, xmm1, 0x1
        vmovdqu xmm1, XMMWORD PTR [rdx]
        vinserti128     ymm1, ymm1, XMMWORD PTR [rdx], 0x1
        add     rdx, r12
        vpshufb ymm2, ymm2, ymm11
        vpshufb ymm1, ymm1, ymm8
        vpmaddwd        ymm1, ymm1, ymm2
        vpaddd  ymm0, ymm1, ymm0
        cmp     rcx, rax
        jg      .L14
.L10:
        vextracti128    xmm1, ymm0, 0x1
        vpaddd  xmm0, xmm0, xmm1
        jmp     .L15
.L44:
        mov     rdx, rax
        jmp     .L16
.L2:
        vmovdqa xmm4, XMMWORD PTR .LC4[rip]
        vmovdqa ymm8, YMMWORD PTR .LC5[rip]
        vmovdqa ymm7, YMMWORD PTR .LC6[rip]
        vmovdqa ymm6, YMMWORD PTR .LC7[rip]
        test    sil, sil
        je      .L30
        mov     QWORD PTR 48[rsp], 5
        mov     QWORD PTR 120[rsp], 2
        mov     QWORD PTR 56[rsp], 10
        jmp     .L3
.L30:
        mov     QWORD PTR 48[rsp], 4
        mov     QWORD PTR 120[rsp], 1
        mov     QWORD PTR 56[rsp], 9
        jmp     .L3
.L49:
        mov     rcx, rdi
        vmovdqa ymm0, YMMWORD PTR [rsp]
        sub     rcx, QWORD PTR 48[rsp]
        xor     eax, eax
        jmp     .L11
.L48:
        lea     rax, 4[rdi+r14]
        cmp     rax, QWORD PTR 32[rsp]
        jl      .L21
        movzx   eax, WORD PTR [r11]
        mov     WORD PTR 156[rsp], ax
        movzx   eax, BYTE PTR 2[r11]
        mov     BYTE PTR 158[rsp], al
        vpmovzxbd       xmm1, DWORD PTR 156[rsp]
        jmp     .L22
.L47:
        movzx   eax, WORD PTR 156[rsp]
        mov     WORD PTR [rbx], ax
        movzx   eax, BYTE PTR 158[rsp]
        mov     BYTE PTR 2[rbx], al
        jmp     .L26
.L43:
        lea     rcx, .LC8[rip]
        mov     edx, 99
        lea     rsi, .LC9[rip]
        lea     rdi, .LC10[rip]
        call    __assert_fail@PLT
foo(unsigned char*, long, unsigned char const*, long, long const*, long const*, short const*, int, unsigned int, bool):
        sub     rsp, 16
        movzx   eax, BYTE PTR 48[rsp]
        push    rax
        push    3
        mov     eax, DWORD PTR 56[rsp]
        push    rax
        mov     eax, DWORD PTR 56[rsp]
        push    rax
        push    QWORD PTR 56[rsp]
        call    ImagingResampleHorizontalConvolution8u(unsigned char*, long, unsigned char const*, long, long const*, long const*, short const*, int, unsigned int, long, bool)@PLT
        add     rsp, 56
        ret
.LC0:
        .byte   0
        .byte   -1
        .byte   4
        .byte   -1
        .byte   1
        .byte   -1
        .byte   5
        .byte   -1
        .byte   2
        .byte   -1
        .byte   6
        .byte   -1
        .byte   3
        .byte   -1
        .byte   7
        .byte   -1
.LC1:
        .byte   0
        .byte   -1
        .byte   4
        .byte   -1
        .byte   1
        .byte   -1
        .byte   5
        .byte   -1
        .byte   2
        .byte   -1
        .byte   6
        .byte   -1
        .byte   3
        .byte   -1
        .byte   7
        .byte   -1
        .byte   8
        .byte   -1
        .byte   12
        .byte   -1
        .byte   9
        .byte   -1
        .byte   13
        .byte   -1
        .byte   10
        .byte   -1
        .byte   14
        .byte   -1
        .byte   11
        .byte   -1
        .byte   15
        .byte   -1
.LC2:
        .byte   0
        .byte   -1
        .byte   4
        .byte   -1
        .byte   1
        .byte   -1
        .byte   5
        .byte   -1
        .byte   2
        .byte   -1
        .byte   6
        .byte   -1
        .byte   3
        .byte   -1
        .byte   7
        .byte   -1
        .byte   0
        .byte   -1
        .byte   4
        .byte   -1
        .byte   1
        .byte   -1
        .byte   5
        .byte   -1
        .byte   2
        .byte   -1
        .byte   6
        .byte   -1
        .byte   3
        .byte   -1
        .byte   7
        .byte   -1
.LC3:
        .byte   8
        .byte   -1
        .byte   12
        .byte   -1
        .byte   9
        .byte   -1
        .byte   13
        .byte   -1
        .byte   10
        .byte   -1
        .byte   14
        .byte   -1
        .byte   11
        .byte   -1
        .byte   15
        .byte   -1
        .byte   8
        .byte   -1
        .byte   12
        .byte   -1
        .byte   9
        .byte   -1
        .byte   13
        .byte   -1
        .byte   10
        .byte   -1
        .byte   14
        .byte   -1
        .byte   11
        .byte   -1
        .byte   15
        .byte   -1
.LC4:
        .byte   0
        .byte   -1
        .byte   3
        .byte   -1
        .byte   1
        .byte   -1
        .byte   4
        .byte   -1
        .byte   2
        .byte   -1
        .byte   5
        .byte   -1
        .byte   -1
        .byte   -1
        .byte   -1
        .byte   -1
.LC5:
        .byte   0
        .byte   -1
        .byte   3
        .byte   -1
        .byte   1
        .byte   -1
        .byte   4
        .byte   -1
        .byte   2
        .byte   -1
        .byte   5
        .byte   -1
        .byte   -1
        .byte   -1
        .byte   -1
        .byte   -1
        .byte   6
        .byte   -1
        .byte   9
        .byte   -1
        .byte   7
        .byte   -1
        .byte   10
        .byte   -1
        .byte   8
        .byte   -1
        .byte   11
        .byte   -1
        .byte   -1
        .byte   -1
        .byte   -1
        .byte   -1
.LC6:
        .byte   0
        .byte   -1
        .byte   3
        .byte   -1
        .byte   1
        .byte   -1
        .byte   4
        .byte   -1
        .byte   2
        .byte   -1
        .byte   5
        .byte   -1
        .byte   -1
        .byte   -1
        .byte   -1
        .byte   -1
        .byte   0
        .byte   -1
        .byte   3
        .byte   -1
        .byte   1
        .byte   -1
        .byte   4
        .byte   -1
        .byte   2
        .byte   -1
        .byte   5
        .byte   -1
        .byte   -1
        .byte   -1
        .byte   -1
        .byte   -1
.LC7:
        .byte   6
        .byte   -1
        .byte   9
        .byte   -1
        .byte   7
        .byte   -1
        .byte   10
        .byte   -1
        .byte   8
        .byte   -1
        .byte   11
        .byte   -1
        .byte   -1
        .byte   -1
        .byte   -1
        .byte   -1
        .byte   6
        .byte   -1
        .byte   9
        .byte   -1
        .byte   7
        .byte   -1
        .byte   10
        .byte   -1
        .byte   8
        .byte   -1
        .byte   11
        .byte   -1
        .byte   -1
        .byte   -1
        .byte   -1
        .byte   -1
.LC11:
        .byte   0
        .byte   1
        .byte   2
        .byte   3
        .byte   0
        .byte   1
        .byte   2
        .byte   3
        .byte   0
        .byte   1
        .byte   2
        .byte   3
        .byte   0
        .byte   1
        .byte   2
        .byte   3
        .byte   8
        .byte   9
        .byte   10
        .byte   11
        .byte   8
        .byte   9
        .byte   10
        .byte   11
        .byte   8
        .byte   9
        .byte   10
        .byte   11
        .byte   8
        .byte   9
        .byte   10
        .byte   11
.LC12:
        .byte   4
        .byte   5
        .byte   6
        .byte   7
        .byte   4
        .byte   5
        .byte   6
        .byte   7
        .byte   4
        .byte   5
        .byte   6
        .byte   7
        .byte   4
        .byte   5
        .byte   6
        .byte   7
        .byte   12
        .byte   13
        .byte   14
        .byte   15
        .byte   12
        .byte   13
        .byte   14
        .byte   15
        .byte   12
        .byte   13
        .byte   14
        .byte   15
        .byte   12
        .byte   13
        .byte   14
        .byte   15
.LC13:
        .byte   0
        .byte   1
        .byte   2
        .byte   3
        .byte   0
        .byte   1
        .byte   2
        .byte   3
        .byte   0
        .byte   1
        .byte   2
        .byte   3
        .byte   0
        .byte   1
        .byte   2
        .byte   3
        .byte   4
        .byte   5
        .byte   6
        .byte   7
        .byte   4
        .byte   5
        .byte   6
        .byte   7
        .byte   4
        .byte   5
        .byte   6
        .byte   7
        .byte   4
        .byte   5
        .byte   6
        .byte   7
```


- Assembly for source 2
  - compiler: x86-64 GCC 9.4
  - Flags: -std=c++17 -O3 -mavx -mfma -mavx2 -mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store -Wno-deprecated -fvisibility-inlines-hidden -fopenmp -O3 -fPIC -Wno-narrowing -Wall -Wextra -Werror=return-type -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-sign-compare -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function -Wno-unused-result -Wno-unused-local-typedefs -Wno-strict-overflow -Wno-strict-aliasing -Wno-error=deprecated-declarations -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=redundant-decls -Wno-error=old-style-cast -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Werror=cast-function-type -Wno-stringop-overflow

```
.LC8:
        .string "void ImagingResampleHorizontalConvolution8u(uint8_t*, int64_t, const uint8_t*, int64_t, const int64_t*, const int64_t*, const int16_t*, int, unsigned int, int64_t, bool)"
.LC9:
        .string "/app/example.cpp"
.LC10:
        .string "stride == 3 || stride == 4"
ImagingResampleHorizontalConvolution8u(unsigned char*, long, unsigned char const*, long, long const*, long const*, short const*, int, unsigned int, long, bool):
        push    rbp
        mov     rbp, rsp
        push    r15
        push    r14
        push    r13
        push    r12
        push    rbx
        mov     rbx, rdi
        and     rsp, -32
        sub     rsp, 160
        mov     eax, DWORD PTR 32[rbp]
        mov     QWORD PTR 128[rsp], rsi
        mov     esi, DWORD PTR 48[rbp]
        mov     QWORD PTR 104[rsp], r8
        mov     r8, QWORD PTR 40[rbp]
        mov     QWORD PTR 112[rsp], rdx
        mov     QWORD PTR 96[rsp], r9
        mov     BYTE PTR 47[rsp], sil
        cmp     r8, 3
        je      .L2
        cmp     r8, 4
        jne     .L43
        vmovdqa xmm4, XMMWORD PTR .LC0[rip]
        vmovdqa ymm8, YMMWORD PTR .LC1[rip]
        mov     QWORD PTR 48[rsp], 3
        vmovdqa ymm7, YMMWORD PTR .LC2[rip]
        vmovdqa ymm6, YMMWORD PTR .LC3[rip]
        mov     QWORD PTR 120[rsp], 1
        mov     QWORD PTR 56[rsp], 7
.L3:
        mov     rdi, QWORD PTR 128[rsp]
        imul    rcx, r8
        mov     rdx, rdi
        imul    rdx, r8
        mov     QWORD PTR 32[rsp], rcx
        mov     QWORD PTR 80[rsp], rdx
        test    rdi, rdi
        jle     .L40
        mov     edx, 1
        lea     ecx, -2[rax]
        vmovd   xmm12, eax
        mov     r13, r8
        mov     edi, edx
        vmovdqa ymm11, YMMWORD PTR .LC13[rip]
        vmovdqa ymm10, YMMWORD PTR .LC11[rip]
        lea     r15, 0[0+r8*8]
        lea     r9, [r8+r8]
        vpxor   xmm15, xmm15, xmm15
        sal     edi, cl
        lea     ecx, -1[rax]
        vmovd   xmm5, edi
        sal     edx, cl
        cmp     r8, 3
        mov     eax, 4
        vpbroadcastd    ymm5, xmm5
        vmovdqa ymm9, YMMWORD PTR .LC12[rip]
        lea     r12, 0[0+r8*4]
        vpxor   xmm14, xmm14, xmm14
        vmovdqa YMMWORD PTR [rsp], ymm5
        vmovd   xmm5, edx
        sete    dl
        sub     rax, rbx
        and     edx, esi
        mov     QWORD PTR 64[rsp], rax
        neg     r13
        mov     rsi, QWORD PTR 16[rbp]
        mov     BYTE PTR 79[rsp], dl
        movsx   rdx, DWORD PTR 24[rbp]
        vpshufd xmm13, xmm5, 0
        xor     r10d, r10d
        sal     r13, 2
        lea     rdi, [rdx+rdx]
        mov     QWORD PTR 88[rsp], rdi
.L28:
        mov     rax, QWORD PTR 104[rsp]
        mov     r14, QWORD PTR [rax+r10*8]
        mov     rax, QWORD PTR 96[rsp]
        mov     rdi, QWORD PTR [rax+r10*8]
        mov     rax, QWORD PTR 112[rsp]
        lea     r11, [rax+r14]
        cmp     rdi, 7
        jg      .L5
        vmovdqa xmm0, xmm13
        xor     eax, eax
.L15:
        mov     rcx, rdi
        sub     rcx, QWORD PTR 120[rsp]
        cmp     rcx, rax
        jle     .L6
        mov     rdx, r8
        imul    rdx, rax
        add     rdx, r11
.L18:
        vmovq   xmm1, QWORD PTR [rdx]
        vbroadcastss    xmm2, DWORD PTR [rsi+rax*2]
        add     rax, 2
        add     rdx, r9
        vpshufb xmm1, xmm1, xmm4
        vpmaddwd        xmm1, xmm1, xmm2
        vpaddd  xmm0, xmm1, xmm0
        cmp     rcx, rax
        jg      .L18
.L6:
        sub     rdi, 1
        cmp     rdi, rax
        jle     .L44
        mov     rcx, r8
        mov     QWORD PTR 136[rsp], rdi
        imul    rcx, rax
        add     rcx, r11
.L19:
        movsx   edx, WORD PTR [rsi+rax*2]
        vpmovzxbd       xmm1, DWORD PTR [rcx]
        add     rax, 1
        add     rcx, r8
        vmovd   xmm5, edx
        vpshufd xmm2, xmm5, 0
        vpmaddwd        xmm1, xmm1, xmm2
        vpaddd  xmm0, xmm1, xmm0
        cmp     rax, rdi
        jne     .L19
        mov     rdx, QWORD PTR 136[rsp]
.L16:
        cmp     rdi, rdx
        je      .L45
.L20:
        vpsrad  xmm0, xmm0, xmm12
        vpackssdw       xmm0, xmm0, xmm15
        vpackuswb       xmm0, xmm0, xmm14
        vmovd   DWORD PTR 156[rsp], xmm0
        vmovd   eax, xmm0
        cmp     r8, 3
        je      .L46
.L23:
        mov     DWORD PTR [rbx], eax
.L26:
        add     r10, 1
        add     rbx, r8
        add     rsi, QWORD PTR 88[rsp]
        cmp     QWORD PTR 128[rsp], r10
        jne     .L28
.L40:
        vzeroupper
        lea     rsp, -40[rbp]
        pop     rbx
        pop     r12
        pop     r13
        pop     r14
        pop     r15
        pop     rbp
        ret
.L46:
        mov     rdi, QWORD PTR 64[rsp]
        lea     rdx, [rdi+rbx]
        cmp     QWORD PTR 80[rsp], rdx
        jg      .L23
        cmp     BYTE PTR 47[rsp], 0
        jne     .L47
        mov     edx, DWORD PTR 3[rbx]
        vmovd   DWORD PTR [rbx], xmm0
        mov     DWORD PTR 3[rbx], edx
        jmp     .L26
.L45:
        movsx   eax, WORD PTR [rsi+rdi*2]
        imul    rdi, r8
        vmovd   xmm5, eax
        add     r11, rdi
        cmp     BYTE PTR 79[rsp], 0
        vpshufd xmm2, xmm5, 0
        jne     .L48
.L21:
        vpmovzxbd       xmm1, DWORD PTR [r11]
.L22:
        vpmaddwd        xmm1, xmm1, xmm2
        vpaddd  xmm0, xmm1, xmm0
        jmp     .L20
.L5:
        mov     rcx, rdi
        sub     rcx, QWORD PTR 56[rsp]
        test    rcx, rcx
        jle     .L49
        vmovdqa ymm0, YMMWORD PTR [rsp]
        lea     rdx, [r11+r12]
        xor     eax, eax
.L13:
        vmovdqu xmm3, XMMWORD PTR [rsi+rax*2]
        vmovdqu xmm1, XMMWORD PTR [rdx+r13]
        vinserti128     ymm3, ymm3, XMMWORD PTR [rsi+rax*2], 0x1
        vinserti128     ymm1, ymm1, XMMWORD PTR [rdx], 0x1
        add     rax, 8
        add     rdx, r15
        vpshufb ymm5, ymm1, ymm7
        vpshufb ymm2, ymm3, ymm10
        vpshufb ymm1, ymm1, ymm6
        vpmaddwd        ymm2, ymm5, ymm2
        vpshufb ymm3, ymm3, ymm9
        vpmaddwd        ymm1, ymm1, ymm3
        vpaddd  ymm0, ymm2, ymm0
        vpaddd  ymm0, ymm0, ymm1
        cmp     rcx, rax
        jg      .L13
        mov     rcx, rdi
        sub     rcx, QWORD PTR 48[rsp]
        cmp     rax, rcx
        jge     .L10
.L11:
        mov     rdx, rax
        imul    rdx, r8
        add     rdx, r11
.L14:
        vmovq   xmm1, QWORD PTR [rsi+rax*2]
        add     rax, 4
        vmovdqa xmm2, xmm1
        vinserti128     ymm2, ymm2, xmm1, 0x1
        vmovdqu xmm1, XMMWORD PTR [rdx]
        vinserti128     ymm1, ymm1, XMMWORD PTR [rdx], 0x1
        add     rdx, r12
        vpshufb ymm2, ymm2, ymm11
        vpshufb ymm1, ymm1, ymm8
        vpmaddwd        ymm1, ymm1, ymm2
        vpaddd  ymm0, ymm1, ymm0
        cmp     rcx, rax
        jg      .L14
.L10:
        vextracti128    xmm1, ymm0, 0x1
        vpaddd  xmm0, xmm0, xmm1
        jmp     .L15
.L44:
        mov     rdx, rax
        jmp     .L16
.L2:
        vmovdqa xmm4, XMMWORD PTR .LC4[rip]
        vmovdqa ymm8, YMMWORD PTR .LC5[rip]
        vmovdqa ymm7, YMMWORD PTR .LC6[rip]
        vmovdqa ymm6, YMMWORD PTR .LC7[rip]
        test    sil, sil
        je      .L30
        mov     QWORD PTR 48[rsp], 5
        mov     QWORD PTR 120[rsp], 2
        mov     QWORD PTR 56[rsp], 10
        jmp     .L3
.L30:
        mov     QWORD PTR 48[rsp], 4
        mov     QWORD PTR 120[rsp], 1
        mov     QWORD PTR 56[rsp], 9
        jmp     .L3
.L49:
        mov     rcx, rdi
        vmovdqa ymm0, YMMWORD PTR [rsp]
        sub     rcx, QWORD PTR 48[rsp]
        xor     eax, eax
        jmp     .L11
.L48:
        lea     rax, 4[rdi+r14]
        cmp     rax, QWORD PTR 32[rsp]
        jl      .L21
        movzx   eax, WORD PTR [r11]
        mov     WORD PTR 156[rsp], ax
        movzx   eax, BYTE PTR 2[r11]
        mov     BYTE PTR 158[rsp], al
        vpmovzxbd       xmm1, DWORD PTR 156[rsp]
        jmp     .L22
.L47:
        movzx   eax, WORD PTR 156[rsp]
        mov     WORD PTR [rbx], ax
        movzx   eax, BYTE PTR 158[rsp]
        mov     BYTE PTR 2[rbx], al
        jmp     .L26
.L43:
        lea     rcx, .LC8[rip]
        mov     edx, 99
        lea     rsi, .LC9[rip]
        lea     rdi, .LC10[rip]
        call    __assert_fail@PLT
foo(unsigned char*, long, unsigned char const*, long, long const*, long const*, short const*, int, unsigned int, bool):
        sub     rsp, 16
        movzx   eax, BYTE PTR 48[rsp]
        push    rax
        push    3
        mov     eax, DWORD PTR 56[rsp]
        push    rax
        mov     eax, DWORD PTR 56[rsp]
        push    rax
        push    QWORD PTR 56[rsp]
        call    ImagingResampleHorizontalConvolution8u(unsigned char*, long, unsigned char const*, long, long const*, long const*, short const*, int, unsigned int, long, bool)@PLT
        add     rsp, 56
        ret
.LC0:
        .byte   0
        .byte   -1
        .byte   4
        .byte   -1
        .byte   1
        .byte   -1
        .byte   5
        .byte   -1
        .byte   2
        .byte   -1
        .byte   6
        .byte   -1
        .byte   3
        .byte   -1
        .byte   7
        .byte   -1
.LC1:
        .byte   0
        .byte   -1
        .byte   4
        .byte   -1
        .byte   1
        .byte   -1
        .byte   5
        .byte   -1
        .byte   2
        .byte   -1
        .byte   6
        .byte   -1
        .byte   3
        .byte   -1
        .byte   7
        .byte   -1
        .byte   8
        .byte   -1
        .byte   12
        .byte   -1
        .byte   9
        .byte   -1
        .byte   13
        .byte   -1
        .byte   10
        .byte   -1
        .byte   14
        .byte   -1
        .byte   11
        .byte   -1
        .byte   15
        .byte   -1
.LC2:
        .byte   0
        .byte   -1
        .byte   4
        .byte   -1
        .byte   1
        .byte   -1
        .byte   5
        .byte   -1
        .byte   2
        .byte   -1
        .byte   6
        .byte   -1
        .byte   3
        .byte   -1
        .byte   7
        .byte   -1
        .byte   0
        .byte   -1
        .byte   4
        .byte   -1
        .byte   1
        .byte   -1
        .byte   5
        .byte   -1
        .byte   2
        .byte   -1
        .byte   6
        .byte   -1
        .byte   3
        .byte   -1
        .byte   7
        .byte   -1
.LC3:
        .byte   8
        .byte   -1
        .byte   12
        .byte   -1
        .byte   9
        .byte   -1
        .byte   13
        .byte   -1
        .byte   10
        .byte   -1
        .byte   14
        .byte   -1
        .byte   11
        .byte   -1
        .byte   15
        .byte   -1
        .byte   8
        .byte   -1
        .byte   12
        .byte   -1
        .byte   9
        .byte   -1
        .byte   13
        .byte   -1
        .byte   10
        .byte   -1
        .byte   14
        .byte   -1
        .byte   11
        .byte   -1
        .byte   15
        .byte   -1
.LC4:
        .byte   0
        .byte   -1
        .byte   3
        .byte   -1
        .byte   1
        .byte   -1
        .byte   4
        .byte   -1
        .byte   2
        .byte   -1
        .byte   5
        .byte   -1
        .byte   -1
        .byte   -1
        .byte   -1
        .byte   -1
.LC5:
        .byte   0
        .byte   -1
        .byte   3
        .byte   -1
        .byte   1
        .byte   -1
        .byte   4
        .byte   -1
        .byte   2
        .byte   -1
        .byte   5
        .byte   -1
        .byte   -1
        .byte   -1
        .byte   -1
        .byte   -1
        .byte   6
        .byte   -1
        .byte   9
        .byte   -1
        .byte   7
        .byte   -1
        .byte   10
        .byte   -1
        .byte   8
        .byte   -1
        .byte   11
        .byte   -1
        .byte   -1
        .byte   -1
        .byte   -1
        .byte   -1
.LC6:
        .byte   0
        .byte   -1
        .byte   3
        .byte   -1
        .byte   1
        .byte   -1
        .byte   4
        .byte   -1
        .byte   2
        .byte   -1
        .byte   5
        .byte   -1
        .byte   -1
        .byte   -1
        .byte   -1
        .byte   -1
        .byte   0
        .byte   -1
        .byte   3
        .byte   -1
        .byte   1
        .byte   -1
        .byte   4
        .byte   -1
        .byte   2
        .byte   -1
        .byte   5
        .byte   -1
        .byte   -1
        .byte   -1
        .byte   -1
        .byte   -1
.LC7:
        .byte   6
        .byte   -1
        .byte   9
        .byte   -1
        .byte   7
        .byte   -1
        .byte   10
        .byte   -1
        .byte   8
        .byte   -1
        .byte   11
        .byte   -1
        .byte   -1
        .byte   -1
        .byte   -1
        .byte   -1
        .byte   6
        .byte   -1
        .byte   9
        .byte   -1
        .byte   7
        .byte   -1
        .byte   10
        .byte   -1
        .byte   8
        .byte   -1
        .byte   11
        .byte   -1
        .byte   -1
        .byte   -1
        .byte   -1
        .byte   -1
.LC11:
        .byte   0
        .byte   1
        .byte   2
        .byte   3
        .byte   0
        .byte   1
        .byte   2
        .byte   3
        .byte   0
        .byte   1
        .byte   2
        .byte   3
        .byte   0
        .byte   1
        .byte   2
        .byte   3
        .byte   8
        .byte   9
        .byte   10
        .byte   11
        .byte   8
        .byte   9
        .byte   10
        .byte   11
        .byte   8
        .byte   9
        .byte   10
        .byte   11
        .byte   8
        .byte   9
        .byte   10
        .byte   11
.LC12:
        .byte   4
        .byte   5
        .byte   6
        .byte   7
        .byte   4
        .byte   5
        .byte   6
        .byte   7
        .byte   4
        .byte   5
        .byte   6
        .byte   7
        .byte   4
        .byte   5
        .byte   6
        .byte   7
        .byte   12
        .byte   13
        .byte   14
        .byte   15
        .byte   12
        .byte   13
        .byte   14
        .byte   15
        .byte   12
        .byte   13
        .byte   14
        .byte   15
        .byte   12
        .byte   13
        .byte   14
        .byte   15
.LC13:
        .byte   0
        .byte   1
        .byte   2
        .byte   3
        .byte   0
        .byte   1
        .byte   2
        .byte   3
        .byte   0
        .byte   1
        .byte   2
        .byte   3
        .byte   0
        .byte   1
        .byte   2
        .byte   3
        .byte   4
        .byte   5
        .byte   6
        .byte   7
        .byte   4
        .byte   5
        .byte   6
        .byte   7
        .byte   4
        .byte   5
        .byte   6
        .byte   7
        .byte   4
        .byte   5
        .byte   6
        .byte   7
```