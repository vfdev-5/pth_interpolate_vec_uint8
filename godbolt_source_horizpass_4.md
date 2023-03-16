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

void ImagingResampleHorizontalConvolution8u4x(
    uint8_t* C10_RESTRICT lineOut0,
    uint8_t* C10_RESTRICT lineOut1,
    uint8_t* C10_RESTRICT lineOut2,
    uint8_t* C10_RESTRICT lineOut3,
    int64_t out_xsize,
    const uint8_t* C10_RESTRICT lineIn0,
    const uint8_t* C10_RESTRICT lineIn1,
    const uint8_t* C10_RESTRICT lineIn2,
    const uint8_t* C10_RESTRICT lineIn3,
    int64_t in_xsize,
    const int64_t* idx_ptr_xmin,
    const int64_t* idx_ptr_size,
    const int16_t* kk,
    int kmax,
    unsigned int coefs_precision,
    int64_t num_channels,
    bool is_last_line) {

  // Interpolation horizontal pass processing together 4 vertical lines.
  // - Input data format is RGBA with R,G,B,A being uint8, we can encode 4 values as a single uint32 value.
  // - We split the size of weight vector for a given output index as a sum: K = n * 4 + m * 2 + k.
  // - We load and process 4 weights values in a loop ("block 4") then we process 2 weights values
  // in another loop ("block 2") and finally we process 1 weights value in the final loop ("block 1").

  // Define shuffling masks (low/high) for num_channels 4 and 3
  // Mask low casts lower half of each lane to epi16 and reorder RGBARGBA -> RRGGBBAA:
  //   [r1 g1 b1 a1  r2 g2 b2 a2  ... | R1 G1 B1 A1  R2 G2 B2 A2 ... ] ->
  //   [r1 0 r2 0  g1 0 g2 0  b1 0 b2 0  a1 0 a2 0 | R1 0 R2 0  G1 0 G2 0  B1 0 B2 0  A1 0 A2 0]
  // Mask high casts upper half of each lane to epi16 and reorder RGBARGBA -> RRGGBBAA::
  //   [ ... r3 g3 b3 a3  r4 g4 b4 a4 | ... R3 G3 B3 A3  R4 G4 B4 A4 ] ->
  //   [r3 0 r4 0  g3 0 g4 0  b3 0 b4 0  a3 0 a4 0 | R3 0 R4 0  G3 0 G4 0  B3 0 B4 0  A3 0 A4 0]

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

  const auto mask_low = (num_channels == 3) ? masks_low_high_c3[0] : masks_low_high_c4[0];
  const auto mask_high = (num_channels == 3) ? masks_low_high_c3[1] : masks_low_high_c4[1];

  const auto stride = num_channels * 1;  // num channels * sizeof(uint8)

  TORCH_INTERNAL_ASSERT(stride == 3 || stride == 4);

  // Precompute xmax limits for block 4 and block 2
  // lineIn0 + stride * (x + xmin) + 16 <= lineIn0 + stride * (xmax + xmin)
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

  // out_xsize = output width, out_x = output x index
  // xmax = interpolation size, x = interpolation index (horizontal <-> x dimension)
  // xmin = input x start index corresponding to output x index (out_x)

  const auto max_out_x_strided = out_xsize * stride;
  const auto max_in_x_strided = in_xsize * stride;

  const auto zero = _mm256_setzero_si256();
  const auto initial = _mm256_set1_epi32(1 << (coefs_precision - 1));

//   for (const auto out_x : c10::irange(out_xsize)) {
  for (auto out_x=0; out_x < out_xsize; out_x++) {
    const auto ids_min = idx_ptr_xmin[out_x];
    const auto ids_size = idx_ptr_size[out_x];
    const auto * k = &kk[out_x * kmax];
    int64_t i = 0;

    auto sss0 = initial;
    auto sss1 = initial;

    const auto * lineIn0_min = lineIn0 + ids_min;
    const auto * lineIn1_min = lineIn1 + ids_min;
    const auto * lineIn2_min = lineIn2 + ids_min;
    const auto * lineIn3_min = lineIn3 + ids_min;

    // block 4
    for (; i < ids_size - b4_delta; i += 4) {
      // Load 4 values from weight vector
      // mmk0 = [wl_0 wh_0 wl_1 wh_1  wl_0 wh_0 wl_1 wh_1  ...]
      // mmk1 = [wl_2 wh_2 wl_3 wh_3  wl_2 wh_2 wl_3 wh_3  ...]
      const auto mmk0 = _mm256_set1_epi32(*(int32_t*)&k[i]);
      const auto mmk1 = _mm256_set1_epi32(*(int32_t*)&k[i + 2]);

      // RGBA: Load 8 pixels (4 per line) from input lines 0 and 1:
      // source = [
      //   r0 g0 b0 a0  r1 g1 b1 a1  r2 g2 b2 a2  r3 g3 b3 a3
      //   R0 G0 B0 A0  R1 G1 B1 A1  R2 G2 B2 A2  R3 G3 B3 A3
      // ]
      // RGB: Load 10 pixels (5 per line)
      // source = [
      //   r0 g0 b0 r1  g1 b1 r2 g2  b2 r3 g3 b3  r4 g4 b4 r5
      //   R0 G0 B0 R1  G1 B1 R2 G2  B2 R3 G3 B3  R4 G4 B4 R5
      // ]
      auto source = _mm256_inserti128_si256(_mm256_castsi128_si256(
          _mm_loadu_si128((__m128i *) (lineIn0_min + stride * i))),
          _mm_loadu_si128((__m128i *) (lineIn1_min + stride * i)), 1);

      // Apply mask_low:
      // RGBA:
      //   [r0 0 r1 0  g0 0 g1 0  b0 0 b1 0  a0 0 a1 0 | R0 0 R1 0  G0 0 G1 0  B0 0 B1 0  A0 0 A1 0]
      // RGB:
      //   [r0 0 r1 0  g0 0 g1 0  b0 0 b1 0  0 0 0 0 | R0 0 R1 0  G0 0 G1 0  B0 0 B1 0  0 0 0 0]
      auto pix1 = _mm256_shuffle_epi8(source, mask_low);
      // Compute output value as C += w0 * C0 + w1 * C1 for each channel in 32-bit precision
      sss0 = _mm256_add_epi32(sss0, _mm256_madd_epi16(pix1, mmk0));

      // Apply mask_high:
      // RGBA:
      //   [r2 0 r3 0  g2 0 g3 0  b2 0 b3 0  a2 0 a3 0 | R2 0 R3 0  G2 0 G3 0  B2 0 B3 0  A2 0 A3 0]
      // RGB:
      //   [r2 0 r3 0  g2 0 g3 0  b2 0 b3 0  0 0 0 0 | R2 0 R3 0  G2 0 G3 0  B2 0 B3 0  0 0 0 0]
      auto pix2 = _mm256_shuffle_epi8(source, mask_high);
      // Compute output value as C += w2 * C2 + w3 * C3 for each channel in 32-bit precision
      sss0 = _mm256_add_epi32(sss0, _mm256_madd_epi16(pix2, mmk1));

      // Same as above to next lines 2 and 3:
      auto source2 = _mm256_inserti128_si256(_mm256_castsi128_si256(
          _mm_loadu_si128((__m128i *) (lineIn2_min + stride * i))),
          _mm_loadu_si128((__m128i *) (lineIn3_min + stride * i)), 1);
      auto pix3 = _mm256_shuffle_epi8(source2, mask_low);
      sss1 = _mm256_add_epi32(sss1, _mm256_madd_epi16(pix3, mmk0));
      auto pix4 = _mm256_shuffle_epi8(source2, mask_high);
      sss1 = _mm256_add_epi32(sss1, _mm256_madd_epi16(pix4, mmk1));
    }

    // block 2
    for (; i < ids_size - b2_delta; i += 2) {
      // Load 2 values from weight vector
      // mmk = [wl_0 wh_0 wl_1 wh_1  wl_0 wh_0 wl_1 wh_1  ...]
      const auto mmk = _mm256_set1_epi32(*(int32_t*)&k[i]);

      // Load 4 pixels (2 per line) from input lines 0 and 1:
      // RGBA: source = [
      //   r0 g0 b0 a0  r1 g1 b1 a1  0 0 0 0  0 0 0 0
      //   R0 G0 B0 A0  R1 G1 B1 A1  0 0 0 0  0 0 0 0
      // ]
      // RGB: source = [
      //   r0 g0 b0 r1  g1 b1 r2  0 0 0 0  0 0 0 0
      //   R0 G0 B0 R1  G1 B1 R2  0 0 0 0  0 0 0 0
      // ]
      auto source1 = _mm256_inserti128_si256(_mm256_castsi128_si256(
          _mm_loadl_epi64((__m128i *) (lineIn0_min + stride * i))),
          _mm_loadl_epi64((__m128i *) (lineIn1_min + stride * i)), 1);
      // Apply mask_low:
      // RGBA:
      //   [r0 0 r1 0  g0 0 g1 0  b0 0 b1 0  a0 0 a1 0 | R0 0 R1 0  G0 0 G1 0  B0 0 B1 0  A0 0 A1 0]
      // RGB:
      //   [r0 0 r1 0  g0 0 g1 0  b0 0 b1 0  0 0 0 0 | R0 0 R1 0  G0 0 G1 0  B0 0 B1 0  0 0 0 0]
      auto pix1 = _mm256_shuffle_epi8(source1, mask_low);
      // Compute output value as C += w0 * C0 + w1 * C1 for each channel in 32-bit precision
      sss0 = _mm256_add_epi32(sss0, _mm256_madd_epi16(pix1, mmk));

      // Same as above for lines 2 and 3:
      auto source2 = _mm256_inserti128_si256(_mm256_castsi128_si256(
          _mm_loadl_epi64((__m128i *) (lineIn2_min + stride * i))),
          _mm_loadl_epi64((__m128i *) (lineIn3_min + stride * i)), 1);
      auto pix2 = _mm256_shuffle_epi8(source2, mask_low);
      sss1 = _mm256_add_epi32(sss1, _mm256_madd_epi16(pix2, mmk));
    }

    // block 1
    for (; i < ids_size - 1; i++) {
      // Load 1 value from weight vector
      // mmk = [wl_0 wh_0 0 0  wl_0 wh_0 0 0  ...]
      const auto mmk = _mm256_set1_epi32(k[i]);

      // Load 2 pixels (one per line) from input lines 0 and 1:
      // RGBA: source = [
      //   r0 g0 b0 a0  0 0 0 0  0 0 0 0  0 0 0 0
      //   R0 G0 B0 A0  0 0 0 0  0 0 0 0  0 0 0 0
      // ]
      // RGB: source = [
      //   r0 g0 b0 r1  0 0 0 0  0 0 0 0  0 0 0 0
      //   R0 G0 B0 R1  0 0 0 0  0 0 0 0  0 0 0 0
      // ]
      auto pix1 = _mm256_inserti128_si256(_mm256_castsi128_si256(
          mm_cvtepu8_epi32(lineIn0_min + stride * i)),
          mm_cvtepu8_epi32(lineIn1_min + stride * i), 1);
      // Compute output value as C += w0 * C0 for each channel in 32-bit precision
      sss0 = _mm256_add_epi32(sss0, _mm256_madd_epi16(pix1, mmk));

      // Same as above for lines 2 and 3
      auto pix2 = _mm256_inserti128_si256(_mm256_castsi128_si256(
          mm_cvtepu8_epi32(lineIn2_min + stride * i)),
          mm_cvtepu8_epi32(lineIn3_min + stride * i), 1);
      sss1 = _mm256_add_epi32(sss1, _mm256_madd_epi16(pix2, mmk));
    }

    if (i == ids_size - 1) {
      // last element
      auto mmk = _mm256_set1_epi32(k[i]);
      // For num_channels == 3 (3 bytes = one pixel) we tolerate to read 4 bytes
      // lines 0, 1 and 2 wont go out of allocated memory bounds
      auto pix = _mm256_inserti128_si256(_mm256_castsi128_si256(
          mm_cvtepu8_epi32(lineIn0_min + stride * i)),
          mm_cvtepu8_epi32(lineIn1_min + stride * i), 1);
      sss0 = _mm256_add_epi32(sss0, _mm256_madd_epi16(pix, mmk));

      auto p0 = mm_cvtepu8_epi32(lineIn2_min + stride * i);
      __m128i p1;
      if (num_channels == 3 && C10_UNLIKELY(is_last_line && ids_min + stride * i + 4 >= max_in_x_strided)) {
        uint8_t output1[4];
        std::memcpy(output1, lineIn3_min + stride * i, 3);
        p1 = mm_cvtepu8_epi32(output1);
      } else {
        p1 = mm_cvtepu8_epi32(lineIn3_min + stride * i);
      }
      auto pix2 = _mm256_inserti128_si256(_mm256_castsi128_si256(p0), p1, 1);
      sss1 = _mm256_add_epi32(sss1, _mm256_madd_epi16(pix2, mmk));
    }

    // Shift back the output integer values: output = output >> weights_precision
    sss0 = _mm256_srai_epi32(sss0, coefs_precision);
    sss1 = _mm256_srai_epi32(sss1, coefs_precision);
    // Convert packed signed 32-bit integers to packed 16-bit integers using signed saturation
    // (a a a a b b b b | c c c c d d d d) -> (a a b b c c d d | 0 0 0 0 0 0 0 0)
    sss0 = _mm256_packs_epi32(sss0, zero);
    sss1 = _mm256_packs_epi32(sss1, zero);
    // Convert packed signed 16-bit integers to packed 8-bit integers using unsigned saturation
    // (a a b b c c d d) -> (a b c d 0 0 0 0)
    sss0 = _mm256_packus_epi16(sss0, zero);
    sss1 = _mm256_packus_epi16(sss1, zero);

    // Write the output into single uint32
    // (a b c d) -> x_uint32
    auto o0 = _mm_cvtsi128_si32(_mm256_castsi256_si128(sss0));
    auto o1 = _mm_cvtsi128_si32(_mm256_extracti128_si256(sss0, 1));
    auto o2 = _mm_cvtsi128_si32(_mm256_castsi256_si128(sss1));
    auto o3 = _mm_cvtsi128_si32(_mm256_extracti128_si256(sss1, 1));

    const auto out_x_strided = stride * out_x;

    if (num_channels == 3 && C10_UNLIKELY(out_x_strided + 4 >= max_out_x_strided)) {
      // This is a boundary case when we want to write 4 bytes to the output buffer but
      // the 4th bytes is already computed. It means that we can not overwrite it.
      //               v----------|
      // Output = [... X1 X2 X3 | A B C D ...]
      // First, we store store next 4 bytes (A B C D)
      // Second, we write 4 bytes to (X1 X2 X3 | A) -> (U V W | Z)
      //          [... U V W | Z B C D ...]
      // Third, we overwrite next 4 bytes (Z B C D) with stored values (A B C D)
      char next0[4];
      std::memcpy(next0, lineOut0 + out_x_strided + stride, 4);
      std::memcpy(lineOut0 + out_x_strided, (uint8_t *) &o0, 4);
      std::memcpy(lineOut0 + out_x_strided + stride, next0, 4);

      char next1[4];
      std::memcpy(next1, lineOut1 + out_x_strided + stride, 4);
      std::memcpy(lineOut1 + out_x_strided, (uint8_t *) &o1, 4);
      std::memcpy(lineOut1 + out_x_strided + stride, next1, 4);

      char next2[4];
      std::memcpy(next2, lineOut2 + out_x_strided + stride, 4);
      std::memcpy(lineOut2 + out_x_strided, (uint8_t *) &o2, 4);
      std::memcpy(lineOut2 + out_x_strided + stride, next2, 4);

      if (C10_UNLIKELY(is_last_line)) {
        // When we handle the last line, we can not access the next 4 bytes
        // as they are out of memory bounds.
        std::memcpy(lineOut3 + out_x_strided, (uint8_t *) &o3, num_channels);
      } else {
        char next3[4];
        std::memcpy(next3, lineOut3 + out_x_strided + stride, 4);
        std::memcpy(lineOut3 + out_x_strided, (uint8_t *) &o3, 4);
        std::memcpy(lineOut3 + out_x_strided + stride, next3, 4);
      }
    } else if (num_channels == 3) {
      // We simply write 4 bytes (... R G B X 0 0 0 0 0 ...) where X is a garbage value
      // that we will overwrite on the next iteration: (... R G B R G B X 0 0 ...)
      std::memcpy(lineOut0 + out_x_strided, (uint8_t *) &o0, 4);
      std::memcpy(lineOut1 + out_x_strided, (uint8_t *) &o1, 4);
      std::memcpy(lineOut2 + out_x_strided, (uint8_t *) &o2, 4);
      std::memcpy(lineOut3 + out_x_strided, (uint8_t *) &o3, 4);
    } else {
      // num_channels = 4 -> lineOutX + out_x_strided should be uint32 aligned
      *(uint32_t *)(lineOut0 + out_x_strided) = o0;
      *(uint32_t *)(lineOut1 + out_x_strided) = o1;
      *(uint32_t *)(lineOut2 + out_x_strided) = o2;
      *(uint32_t *)(lineOut3 + out_x_strided) = o3;
    }
  }
}


void foo(
    uint8_t* C10_RESTRICT lineOut0,
    uint8_t* C10_RESTRICT lineOut1,
    uint8_t* C10_RESTRICT lineOut2,
    uint8_t* C10_RESTRICT lineOut3,
    int64_t out_xsize,
    const uint8_t* C10_RESTRICT lineIn0,
    const uint8_t* C10_RESTRICT lineIn1,
    const uint8_t* C10_RESTRICT lineIn2,
    const uint8_t* C10_RESTRICT lineIn3,
    int64_t in_xsize,
    const int64_t* idx_ptr_xmin,
    const int64_t* idx_ptr_size,
    const int16_t* kk,
    int kmax,
    unsigned int coefs_precision,
    bool is_last_line
) {
    constexpr int num_channels = 3;

    ImagingResampleHorizontalConvolution8u4x(
        lineOut0,
        lineOut1,
        lineOut2,
        lineOut3,
        out_xsize,
        lineIn0,
        lineIn1,
        lineIn2,
        lineIn3,
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

void ImagingResampleHorizontalConvolution8u4x(
    uint8_t* C10_RESTRICT lineOut0,
    uint8_t* C10_RESTRICT lineOut1,
    uint8_t* C10_RESTRICT lineOut2,
    uint8_t* C10_RESTRICT lineOut3,
    int64_t out_xsize,
    const uint8_t* C10_RESTRICT lineIn0,
    const uint8_t* C10_RESTRICT lineIn1,
    const uint8_t* C10_RESTRICT lineIn2,
    const uint8_t* C10_RESTRICT lineIn3,
    int64_t in_xsize,
    const int64_t* idx_ptr_xmin,
    const int64_t* idx_ptr_size,
    const int16_t* kk,
    int kmax,
    unsigned int coefs_precision,
    int64_t num_channels,
    bool is_last_line) {

  // Interpolation horizontal pass processing together 4 vertical lines.
  // - Input data format is RGBA with R,G,B,A being uint8, we can encode 4 values as a single uint32 value.
  // - We split the size of weight vector for a given output index as a sum: K = n * 4 + m * 2 + k.
  // - We load and process 4 weights values in a loop ("block 4") then we process 2 weights values
  // in another loop ("block 2") and finally we process 1 weights value in the final loop ("block 1").

  // Define shuffling masks (low/high) for num_channels 4 and 3
  // Mask low casts lower half of each lane to epi16 and reorder RGBARGBA -> RRGGBBAA:
  //   [r1 g1 b1 a1  r2 g2 b2 a2  ... | R1 G1 B1 A1  R2 G2 B2 A2 ... ] ->
  //   [r1 0 r2 0  g1 0 g2 0  b1 0 b2 0  a1 0 a2 0 | R1 0 R2 0  G1 0 G2 0  B1 0 B2 0  A1 0 A2 0]
  // Mask high casts upper half of each lane to epi16 and reorder RGBARGBA -> RRGGBBAA::
  //   [ ... r3 g3 b3 a3  r4 g4 b4 a4 | ... R3 G3 B3 A3  R4 G4 B4 A4 ] ->
  //   [r3 0 r4 0  g3 0 g4 0  b3 0 b4 0  a3 0 a4 0 | R3 0 R4 0  G3 0 G4 0  B3 0 B4 0  A3 0 A4 0]

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

  const auto mask_low = (num_channels == 3) ? masks_low_high_c3[0] : masks_low_high_c4[0];
  const auto mask_high = (num_channels == 3) ? masks_low_high_c3[1] : masks_low_high_c4[1];

  const auto stride = num_channels * 1;  // num channels * sizeof(uint8)

  TORCH_INTERNAL_ASSERT(stride == 3 || stride == 4);

  // Precompute xmax limits for block 4 and block 2
  // lineIn0 + stride * (x + xmin) + 16 <= lineIn0 + stride * (xmax + xmin)
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

  // out_xsize = output width, out_x = output x index
  // xmax = interpolation size, x = interpolation index (horizontal <-> x dimension)
  // xmin = input x start index corresponding to output x index (out_x)

  const auto max_out_x_strided = out_xsize * stride;
  const auto max_in_x_strided = in_xsize * stride;

  const auto zero = _mm256_setzero_si256();
  const auto initial = _mm256_set1_epi32(1 << (coefs_precision - 1));

//   for (const auto out_x : c10::irange(out_xsize)) {
  for (auto out_x=0; out_x < out_xsize; out_x++) {
    const auto ids_min = idx_ptr_xmin[out_x];
    const auto ids_size = idx_ptr_size[out_x];
    const auto * k = &kk[out_x * kmax];
    int64_t i = 0;

    auto sss0 = initial;
    auto sss1 = initial;

    const auto * lineIn0_min = lineIn0 + ids_min;
    const auto * lineIn1_min = lineIn1 + ids_min;
    const auto * lineIn2_min = lineIn2 + ids_min;
    const auto * lineIn3_min = lineIn3 + ids_min;

    // block 4
    for (; i < ids_size - b4_delta; i += 4) {
      // Load 4 values from weight vector
      // mmk0 = [wl_0 wh_0 wl_1 wh_1  wl_0 wh_0 wl_1 wh_1  ...]
      // mmk1 = [wl_2 wh_2 wl_3 wh_3  wl_2 wh_2 wl_3 wh_3  ...]
      const auto mmk0 = _mm256_set1_epi32(*(int32_t*)&k[i]);
      const auto mmk1 = _mm256_set1_epi32(*(int32_t*)&k[i + 2]);

      // RGBA: Load 8 pixels (4 per line) from input lines 0 and 1:
      // source = [
      //   r0 g0 b0 a0  r1 g1 b1 a1  r2 g2 b2 a2  r3 g3 b3 a3
      //   R0 G0 B0 A0  R1 G1 B1 A1  R2 G2 B2 A2  R3 G3 B3 A3
      // ]
      // RGB: Load 10 pixels (5 per line)
      // source = [
      //   r0 g0 b0 r1  g1 b1 r2 g2  b2 r3 g3 b3  r4 g4 b4 r5
      //   R0 G0 B0 R1  G1 B1 R2 G2  B2 R3 G3 B3  R4 G4 B4 R5
      // ]
      auto source = _mm256_inserti128_si256(_mm256_castsi128_si256(
          _mm_loadu_si128((__m128i *) (lineIn0_min + stride * i))),
          _mm_loadu_si128((__m128i *) (lineIn1_min + stride * i)), 1);

      // Apply mask_low:
      // RGBA:
      //   [r0 0 r1 0  g0 0 g1 0  b0 0 b1 0  a0 0 a1 0 | R0 0 R1 0  G0 0 G1 0  B0 0 B1 0  A0 0 A1 0]
      // RGB:
      //   [r0 0 r1 0  g0 0 g1 0  b0 0 b1 0  0 0 0 0 | R0 0 R1 0  G0 0 G1 0  B0 0 B1 0  0 0 0 0]
      auto pix1 = _mm256_shuffle_epi8(source, mask_low);
      // Compute output value as C += w0 * C0 + w1 * C1 for each channel in 32-bit precision
      sss0 = _mm256_add_epi32(sss0, _mm256_madd_epi16(pix1, mmk0));

      // Apply mask_high:
      // RGBA:
      //   [r2 0 r3 0  g2 0 g3 0  b2 0 b3 0  a2 0 a3 0 | R2 0 R3 0  G2 0 G3 0  B2 0 B3 0  A2 0 A3 0]
      // RGB:
      //   [r2 0 r3 0  g2 0 g3 0  b2 0 b3 0  0 0 0 0 | R2 0 R3 0  G2 0 G3 0  B2 0 B3 0  0 0 0 0]
      auto pix2 = _mm256_shuffle_epi8(source, mask_high);
      // Compute output value as C += w2 * C2 + w3 * C3 for each channel in 32-bit precision
      sss0 = _mm256_add_epi32(sss0, _mm256_madd_epi16(pix2, mmk1));

      // Same as above to next lines 2 and 3:
      auto source2 = _mm256_inserti128_si256(_mm256_castsi128_si256(
          _mm_loadu_si128((__m128i *) (lineIn2_min + stride * i))),
          _mm_loadu_si128((__m128i *) (lineIn3_min + stride * i)), 1);
      auto pix3 = _mm256_shuffle_epi8(source2, mask_low);
      sss1 = _mm256_add_epi32(sss1, _mm256_madd_epi16(pix3, mmk0));
      auto pix4 = _mm256_shuffle_epi8(source2, mask_high);
      sss1 = _mm256_add_epi32(sss1, _mm256_madd_epi16(pix4, mmk1));
    }

    // block 2
    for (; i < ids_size - b2_delta; i += 2) {
      // Load 2 values from weight vector
      // mmk = [wl_0 wh_0 wl_1 wh_1  wl_0 wh_0 wl_1 wh_1  ...]
      const auto mmk = _mm256_set1_epi32(*(int32_t*)&k[i]);

      // Load 4 pixels (2 per line) from input lines 0 and 1:
      // RGBA: source = [
      //   r0 g0 b0 a0  r1 g1 b1 a1  0 0 0 0  0 0 0 0
      //   R0 G0 B0 A0  R1 G1 B1 A1  0 0 0 0  0 0 0 0
      // ]
      // RGB: source = [
      //   r0 g0 b0 r1  g1 b1 r2  0 0 0 0  0 0 0 0
      //   R0 G0 B0 R1  G1 B1 R2  0 0 0 0  0 0 0 0
      // ]
      auto source1 = _mm256_inserti128_si256(_mm256_castsi128_si256(
          _mm_loadl_epi64((__m128i *) (lineIn0_min + stride * i))),
          _mm_loadl_epi64((__m128i *) (lineIn1_min + stride * i)), 1);
      // Apply mask_low:
      // RGBA:
      //   [r0 0 r1 0  g0 0 g1 0  b0 0 b1 0  a0 0 a1 0 | R0 0 R1 0  G0 0 G1 0  B0 0 B1 0  A0 0 A1 0]
      // RGB:
      //   [r0 0 r1 0  g0 0 g1 0  b0 0 b1 0  0 0 0 0 | R0 0 R1 0  G0 0 G1 0  B0 0 B1 0  0 0 0 0]
      auto pix1 = _mm256_shuffle_epi8(source1, mask_low);
      // Compute output value as C += w0 * C0 + w1 * C1 for each channel in 32-bit precision
      sss0 = _mm256_add_epi32(sss0, _mm256_madd_epi16(pix1, mmk));

      // Same as above for lines 2 and 3:
      auto source2 = _mm256_inserti128_si256(_mm256_castsi128_si256(
          _mm_loadl_epi64((__m128i *) (lineIn2_min + stride * i))),
          _mm_loadl_epi64((__m128i *) (lineIn3_min + stride * i)), 1);
      auto pix2 = _mm256_shuffle_epi8(source2, mask_low);
      sss1 = _mm256_add_epi32(sss1, _mm256_madd_epi16(pix2, mmk));
    }

    // block 1
    for (; i < ids_size - 1; i++) {
      // Load 1 value from weight vector
      // mmk = [wl_0 wh_0 0 0  wl_0 wh_0 0 0  ...]
      const auto mmk = _mm256_set1_epi32(k[i]);

      // Load 2 pixels (one per line) from input lines 0 and 1:
      // RGBA: source = [
      //   r0 g0 b0 a0  0 0 0 0  0 0 0 0  0 0 0 0
      //   R0 G0 B0 A0  0 0 0 0  0 0 0 0  0 0 0 0
      // ]
      // RGB: source = [
      //   r0 g0 b0 r1  0 0 0 0  0 0 0 0  0 0 0 0
      //   R0 G0 B0 R1  0 0 0 0  0 0 0 0  0 0 0 0
      // ]
      auto pix1 = _mm256_inserti128_si256(_mm256_castsi128_si256(
          mm_cvtepu8_epi32(lineIn0_min + stride * i)),
          mm_cvtepu8_epi32(lineIn1_min + stride * i), 1);
      // Compute output value as C += w0 * C0 for each channel in 32-bit precision
      sss0 = _mm256_add_epi32(sss0, _mm256_madd_epi16(pix1, mmk));

      // Same as above for lines 2 and 3
      auto pix2 = _mm256_inserti128_si256(_mm256_castsi128_si256(
          mm_cvtepu8_epi32(lineIn2_min + stride * i)),
          mm_cvtepu8_epi32(lineIn3_min + stride * i), 1);
      sss1 = _mm256_add_epi32(sss1, _mm256_madd_epi16(pix2, mmk));
    }

    if (i == ids_size - 1) {
      // last element
      auto mmk = _mm256_set1_epi32(k[i]);
      // For num_channels == 3 (3 bytes = one pixel) we tolerate to read 4 bytes
      // lines 0, 1 and 2 wont go out of allocated memory bounds
      auto pix = _mm256_inserti128_si256(_mm256_castsi128_si256(
          mm_cvtepu8_epi32(lineIn0_min + stride * i)),
          mm_cvtepu8_epi32(lineIn1_min + stride * i), 1);
      sss0 = _mm256_add_epi32(sss0, _mm256_madd_epi16(pix, mmk));

      auto p0 = mm_cvtepu8_epi32(lineIn2_min + stride * i);
      __m128i p1;
      if (num_channels == 3 && C10_UNLIKELY(is_last_line && ids_min + stride * i + 4 >= max_in_x_strided)) {
        uint8_t output1[4];
        std::memcpy(output1, lineIn3_min + stride * i, 3);
        p1 = mm_cvtepu8_epi32(output1);
      } else {
        p1 = mm_cvtepu8_epi32(lineIn3_min + stride * i);
      }
      auto pix2 = _mm256_inserti128_si256(_mm256_castsi128_si256(p0), p1, 1);
      sss1 = _mm256_add_epi32(sss1, _mm256_madd_epi16(pix2, mmk));
    }

    // Shift back the output integer values: output = output >> weights_precision
    sss0 = _mm256_srai_epi32(sss0, coefs_precision);
    sss1 = _mm256_srai_epi32(sss1, coefs_precision);
    // Convert packed signed 32-bit integers to packed 16-bit integers using signed saturation
    // (a a a a b b b b | c c c c d d d d) -> (a a b b c c d d | 0 0 0 0 0 0 0 0)
    sss0 = _mm256_packs_epi32(sss0, zero);
    sss1 = _mm256_packs_epi32(sss1, zero);
    // Convert packed signed 16-bit integers to packed 8-bit integers using unsigned saturation
    // (a a b b c c d d) -> (a b c d 0 0 0 0)
    sss0 = _mm256_packus_epi16(sss0, zero);
    sss1 = _mm256_packus_epi16(sss1, zero);

    // Write the output into single uint32
    // (a b c d) -> x_uint32
    auto o0 = _mm_cvtsi128_si32(_mm256_castsi256_si128(sss0));
    auto o1 = _mm_cvtsi128_si32(_mm256_extracti128_si256(sss0, 1));
    auto o2 = _mm_cvtsi128_si32(_mm256_castsi256_si128(sss1));
    auto o3 = _mm_cvtsi128_si32(_mm256_extracti128_si256(sss1, 1));

    const auto out_x_strided = stride * out_x;

    if (num_channels == 3 && C10_UNLIKELY(out_x_strided + 4 >= max_out_x_strided)) {
      // This is a boundary case when we want to write 4 bytes to the output buffer but
      // the 4th bytes is already computed. It means that we can not overwrite it.
      //               v----------|
      // Output = [... X1 X2 X3 | A B C D ...]
      // First, we store store next 4 bytes (A B C D)
      // Second, we write 4 bytes to (X1 X2 X3 | A) -> (U V W | Z)
      //          [... U V W | Z B C D ...]
      // Third, we overwrite next 4 bytes (Z B C D) with stored values (A B C D)
      char next0[4];
      std::memcpy(next0, lineOut0 + out_x_strided + stride, 4);
      std::memcpy(lineOut0 + out_x_strided, (uint8_t *) &o0, 4);
      std::memcpy(lineOut0 + out_x_strided + stride, next0, 4);

      char next1[4];
      std::memcpy(next1, lineOut1 + out_x_strided + stride, 4);
      std::memcpy(lineOut1 + out_x_strided, (uint8_t *) &o1, 4);
      std::memcpy(lineOut1 + out_x_strided + stride, next1, 4);

      char next2[4];
      std::memcpy(next2, lineOut2 + out_x_strided + stride, 4);
      std::memcpy(lineOut2 + out_x_strided, (uint8_t *) &o2, 4);
      std::memcpy(lineOut2 + out_x_strided + stride, next2, 4);

      if (C10_UNLIKELY(is_last_line)) {
        // When we handle the last line, we can not access the next 4 bytes
        // as they are out of memory bounds.
        std::memcpy(lineOut3 + out_x_strided, (uint8_t *) &o3, num_channels);
      } else {
        char next3[4];
        std::memcpy(next3, lineOut3 + out_x_strided + stride, 4);
        std::memcpy(lineOut3 + out_x_strided, (uint8_t *) &o3, 4);
        std::memcpy(lineOut3 + out_x_strided + stride, next3, 4);
      }
    } else if (num_channels == 3) {
      // We simply write 4 bytes (... R G B X 0 0 0 0 0 ...) where X is a garbage value
      // that we will overwrite on the next iteration: (... R G B R G B X 0 0 ...)
      std::memcpy(lineOut0 + out_x_strided, (uint8_t *) &o0, 4);
      std::memcpy(lineOut1 + out_x_strided, (uint8_t *) &o1, 4);
      std::memcpy(lineOut2 + out_x_strided, (uint8_t *) &o2, 4);
      std::memcpy(lineOut3 + out_x_strided, (uint8_t *) &o3, 4);
    } else {
      // num_channels = 4 -> lineOutX + out_x_strided should be uint32 aligned
      *(uint32_t *)(lineOut0 + out_x_strided) = o0;
      *(uint32_t *)(lineOut1 + out_x_strided) = o1;
      *(uint32_t *)(lineOut2 + out_x_strided) = o2;
      *(uint32_t *)(lineOut3 + out_x_strided) = o3;
    }
  }
}


void foo(
    uint8_t* C10_RESTRICT lineOut0,
    uint8_t* C10_RESTRICT lineOut1,
    uint8_t* C10_RESTRICT lineOut2,
    uint8_t* C10_RESTRICT lineOut3,
    int64_t out_xsize,
    const uint8_t* C10_RESTRICT lineIn0,
    const uint8_t* C10_RESTRICT lineIn1,
    const uint8_t* C10_RESTRICT lineIn2,
    const uint8_t* C10_RESTRICT lineIn3,
    int64_t in_xsize,
    const int64_t* idx_ptr_xmin,
    const int64_t* idx_ptr_size,
    const int16_t* kk,
    int kmax,
    unsigned int coefs_precision,
    bool is_last_line
) {
    int num_channels = 3;

    ImagingResampleHorizontalConvolution8u4x(
        lineOut0,
        lineOut1,
        lineOut2,
        lineOut3,
        out_xsize,
        lineIn0,
        lineIn1,
        lineIn2,
        lineIn3,
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
.LC4:
        .string "void ImagingResampleHorizontalConvolution8u4x(uint8_t*, uint8_t*, uint8_t*, uint8_t*, int64_t, const uint8_t*, const uint8_t*, const uint8_t*, const uint8_t*, int64_t, const int64_t*, const int64_t*, const int16_t*, int, unsigned int, int64_t, bool)"
.LC5:
        .string "/app/example.cpp"
.LC6:
        .string "stride == 3 || stride == 4"
ImagingResampleHorizontalConvolution8u4x(unsigned char*, unsigned char*, unsigned char*, unsigned char*, long, unsigned char const*, unsigned char const*, unsigned char const*, unsigned char const*, long, long const*, long const*, short const*, int, unsigned int, long, bool):
        push    rbp
        mov     rbp, rsp
        push    r15
        push    r14
        push    r13
        push    r12
        push    rbx
        and     rsp, -32
        add     rsp, -128
        mov     rbx, QWORD PTR 88[rbp]
        mov     QWORD PTR 24[rsp], rdx
        mov     edx, DWORD PTR 96[rbp]
        mov     QWORD PTR 40[rsp], rdi
        mov     edi, DWORD PTR 80[rbp]
        mov     QWORD PTR 32[rsp], rsi
        mov     QWORD PTR 96[rsp], rcx
        mov     QWORD PTR 80[rsp], r8
        mov     QWORD PTR 56[rsp], r9
        mov     BYTE PTR 14[rsp], dl
        cmp     rbx, 3
        je      .L2
        cmp     rbx, 4
        jne     .L36
        mov     QWORD PTR 64[rsp], 1
        vmovdqa ymm4, YMMWORD PTR .LC0[rip]
        mov     rsi, r8
        mov     QWORD PTR 72[rsp], 3
        vmovdqa ymm5, YMMWORD PTR .LC1[rip]
.L3:
        mov     rax, rsi
        lea     ecx, -1[rdi]
        imul    rax, rbx
        mov     QWORD PTR 16[rsp], rax
        mov     rax, QWORD PTR 40[rbp]
        imul    rax, rbx
        mov     QWORD PTR [rsp], rax
        mov     eax, 1
        sal     eax, cl
        vmovd   xmm6, eax
        vpbroadcastd    ymm6, xmm6
        test    rsi, rsi
        jle     .L32
        cmp     rbx, 3
        mov     rsi, QWORD PTR 64[rbp]
        lea     r15, [rbx+rbx]
        vmovd   xmm9, edi
        sete    al
        vpxor   xmm8, xmm8, xmm8
        vpxor   xmm7, xmm7, xmm7
        xor     r13d, r13d
        and     eax, edx
        lea     r12, 0[0+rbx*4]
        mov     QWORD PTR 104[rsp], 0
        mov     BYTE PTR 15[rsp], al
        movsx   rax, DWORD PTR 72[rbp]
        mov     r14, r12
        add     rax, rax
        mov     QWORD PTR 48[rsp], rax
.L21:
        mov     rax, QWORD PTR 104[rsp]
        mov     rdi, QWORD PTR 48[rbp]
        vmovdqa ymm0, ymm6
        xor     edx, edx
        mov     r9, QWORD PTR 16[rbp]
        mov     r8, QWORD PTR 24[rbp]
        vmovdqa ymm1, ymm6
        mov     r12, QWORD PTR [rdi+rax*8]
        mov     rdi, QWORD PTR 56[rbp]
        mov     r11, QWORD PTR [rdi+rax*8]
        mov     rax, QWORD PTR 56[rsp]
        add     r9, r12
        add     r8, r12
        mov     rdi, QWORD PTR 32[rbp]
        mov     rcx, r11
        sub     rcx, QWORD PTR 72[rsp]
        lea     r10, [rax+r12]
        xor     eax, eax
        add     rdi, r12
        test    rcx, rcx
        jle     .L37
.L5:
        vmovdqu xmm3, XMMWORD PTR [r10+rdx]
        vinserti128     ymm3, ymm3, XMMWORD PTR [r9+rdx], 0x1
        vpbroadcastd    ymm11, DWORD PTR [rsi+rax*2]
        vpbroadcastd    ymm10, DWORD PTR 4[rsi+rax*2]
        add     rax, 4
        vpshufb ymm2, ymm3, ymm4
        vpshufb ymm3, ymm3, ymm5
        vpmaddwd        ymm2, ymm2, ymm11
        vpmaddwd        ymm3, ymm3, ymm10
        vpaddd  ymm1, ymm2, ymm1
        vmovdqu xmm2, XMMWORD PTR [r8+rdx]
        vinserti128     ymm2, ymm2, XMMWORD PTR [rdi+rdx], 0x1
        add     rdx, r14
        vpaddd  ymm1, ymm1, ymm3
        vpshufb ymm3, ymm2, ymm4
        vpshufb ymm2, ymm2, ymm5
        vpmaddwd        ymm3, ymm3, ymm11
        vpmaddwd        ymm2, ymm2, ymm10
        vpaddd  ymm0, ymm3, ymm0
        vpaddd  ymm0, ymm0, ymm2
        cmp     rcx, rax
        jg      .L5
.L8:
        mov     rcx, r11
        sub     rcx, QWORD PTR 64[rsp]
        cmp     rcx, rax
        jle     .L6
        mov     rdx, rbx
        imul    rdx, rax
.L11:
        vmovq   xmm2, QWORD PTR [r9+rdx]
        vmovq   xmm3, QWORD PTR [r10+rdx]
        vpbroadcastd    ymm10, DWORD PTR [rsi+rax*2]
        add     rax, 2
        vinserti128     ymm3, ymm3, xmm2, 0x1
        vmovq   xmm2, QWORD PTR [r8+rdx]
        vpshufb ymm3, ymm3, ymm4
        vpmaddwd        ymm3, ymm3, ymm10
        vpaddd  ymm1, ymm3, ymm1
        vmovq   xmm3, QWORD PTR [rdi+rdx]
        add     rdx, r15
        vinserti128     ymm2, ymm2, xmm3, 0x1
        vpshufb ymm2, ymm2, ymm4
        vpmaddwd        ymm2, ymm2, ymm10
        vpaddd  ymm0, ymm2, ymm0
        cmp     rcx, rax
        jg      .L11
.L6:
        sub     r11, 1
        cmp     r11, rax
        jle     .L38
        mov     rcx, rbx
        mov     QWORD PTR 88[rsp], r11
        imul    rcx, rax
.L12:
        movsx   edx, WORD PTR [rsi+rax*2]
        vpmovzxbd       xmm2, DWORD PTR [r9+rcx]
        add     rax, 1
        vpmovzxbd       xmm3, DWORD PTR [r10+rcx]
        vmovd   xmm10, edx
        vinserti128     ymm3, ymm3, xmm2, 0x1
        vpbroadcastd    ymm10, xmm10
        vpmovzxbd       xmm2, DWORD PTR [rdi+rcx]
        vpmaddwd        ymm3, ymm3, ymm10
        vpaddd  ymm1, ymm3, ymm1
        vpmovzxbd       xmm3, DWORD PTR [r8+rcx]
        add     rcx, rbx
        vinserti128     ymm2, ymm3, xmm2, 0x1
        vpmaddwd        ymm2, ymm2, ymm10
        vpaddd  ymm0, ymm2, ymm0
        cmp     rax, r11
        jne     .L12
.L9:
        cmp     r11, QWORD PTR 88[rsp]
        je      .L39
.L13:
        vpsrad  ymm1, ymm1, xmm9
        vpsrad  ymm0, ymm0, xmm9
        mov     r11, QWORD PTR 96[rsp]
        vpackssdw       ymm1, ymm1, ymm8
        vpackssdw       ymm0, ymm0, ymm8
        lea     r8, [r11+r13]
        mov     r11, QWORD PTR 40[rsp]
        vpackuswb       ymm1, ymm1, ymm7
        vpackuswb       ymm0, ymm0, ymm7
        vmovdqa xmm2, xmm1
        vmovd   edi, xmm1
        vextracti128    xmm1, ymm1, 0x1
        vmovd   ecx, xmm1
        vmovd   edx, xmm0
        vmovdqa xmm1, xmm0
        vextracti128    xmm0, ymm0, 0x1
        vmovd   eax, xmm0
        vmovd   DWORD PTR 124[rsp], xmm0
        cmp     rbx, 3
        je      .L40
.L34:
        mov     DWORD PTR [r11+r13], edi
        mov     rdi, QWORD PTR 32[rsp]
        mov     DWORD PTR [rdi+r13], ecx
        mov     rdi, QWORD PTR 24[rsp]
        mov     DWORD PTR [rdi+r13], edx
        mov     rdi, QWORD PTR 96[rsp]
        mov     DWORD PTR [rdi+r13], eax
.L19:
        add     QWORD PTR 104[rsp], 1
        add     r13, rbx
        mov     rax, QWORD PTR 104[rsp]
        add     rsi, QWORD PTR 48[rsp]
        cmp     QWORD PTR 80[rsp], rax
        jne     .L21
.L32:
        vzeroupper
        lea     rsp, -40[rbp]
        pop     rbx
        pop     r12
        pop     r13
        pop     r14
        pop     r15
        pop     rbp
        ret
.L40:
        mov     r11, QWORD PTR 40[rsp]
        lea     r9, 4[r13]
        cmp     QWORD PTR 16[rsp], r9
        jg      .L34
        mov     r9d, DWORD PTR 3[r11+r13]
        vmovd   DWORD PTR [r11+r13], xmm2
        cmp     BYTE PTR 14[rsp], 0
        mov     DWORD PTR 3[r11+r13], r9d
        mov     r11, QWORD PTR 32[rsp]
        mov     edi, DWORD PTR 3[r11+r13]
        mov     DWORD PTR [r11+r13], ecx
        mov     DWORD PTR 3[r11+r13], edi
        mov     rdi, QWORD PTR 24[rsp]
        mov     ecx, DWORD PTR 3[rdi+r13]
        vmovd   DWORD PTR [rdi+r13], xmm1
        mov     DWORD PTR 3[rdi+r13], ecx
        jne     .L41
        mov     rdi, QWORD PTR 96[rsp]
        mov     edx, DWORD PTR 3[rdi+r13]
        vmovd   DWORD PTR [rdi+r13], xmm0
        mov     DWORD PTR 3[rdi+r13], edx
        jmp     .L19
.L39:
        movsx   eax, WORD PTR [rsi+r11*2]
        imul    r11, rbx
        vmovd   xmm10, eax
        vpmovzxbd       xmm2, DWORD PTR [r9+r11]
        vpmovzxbd       xmm3, DWORD PTR [r10+r11]
        vpbroadcastd    ymm10, xmm10
        add     rdi, r11
        cmp     BYTE PTR 15[rsp], 0
        vinserti128     ymm3, ymm3, xmm2, 0x1
        vpmovzxbd       xmm2, DWORD PTR [r8+r11]
        vpmaddwd        ymm3, ymm3, ymm10
        vpaddd  ymm1, ymm3, ymm1
        jne     .L42
.L14:
        vpmovzxbd       xmm3, DWORD PTR [rdi]
.L15:
        vinserti128     ymm2, ymm2, xmm3, 0x1
        vpmaddwd        ymm2, ymm2, ymm10
        vpaddd  ymm0, ymm2, ymm0
        jmp     .L13
.L38:
        mov     QWORD PTR 88[rsp], rax
        jmp     .L9
.L37:
        xor     eax, eax
        jmp     .L8
.L2:
        vmovdqa ymm4, YMMWORD PTR .LC2[rip]
        vmovdqa ymm5, YMMWORD PTR .LC3[rip]
        test    dl, dl
        je      .L43
        mov     QWORD PTR 64[rsp], 2
        mov     rsi, r8
        mov     QWORD PTR 72[rsp], 5
        jmp     .L3
.L43:
        mov     QWORD PTR 64[rsp], 1
        mov     rsi, r8
        mov     QWORD PTR 72[rsp], 4
        jmp     .L3
.L42:
        lea     rax, 4[r11+r12]
        cmp     rax, QWORD PTR [rsp]
        jl      .L14
        movzx   eax, WORD PTR [rdi]
        mov     WORD PTR 124[rsp], ax
        movzx   eax, BYTE PTR 2[rdi]
        mov     BYTE PTR 126[rsp], al
        vpmovzxbd       xmm3, DWORD PTR 124[rsp]
        jmp     .L15
.L41:
        movzx   eax, WORD PTR 124[rsp]
        mov     WORD PTR [r8], ax
        movzx   eax, BYTE PTR 126[rsp]
        mov     BYTE PTR 2[r8], al
        jmp     .L19
.L36:
        lea     rcx, .LC4[rip]
        mov     edx, 76
        lea     rsi, .LC5[rip]
        lea     rdi, .LC6[rip]
        call    __assert_fail@PLT
foo(unsigned char*, unsigned char*, unsigned char*, unsigned char*, long, unsigned char const*, unsigned char const*, unsigned char const*, unsigned char const*, long, long const*, long const*, short const*, int, unsigned int, bool):
        sub     rsp, 16
        movzx   eax, BYTE PTR 96[rsp]
        push    rax
        push    3
        mov     eax, DWORD PTR 104[rsp]
        push    rax
        mov     eax, DWORD PTR 104[rsp]
        push    rax
        push    QWORD PTR 104[rsp]
        push    QWORD PTR 104[rsp]
        push    QWORD PTR 104[rsp]
        push    QWORD PTR 104[rsp]
        push    QWORD PTR 104[rsp]
        push    QWORD PTR 104[rsp]
        push    QWORD PTR 104[rsp]
        call    ImagingResampleHorizontalConvolution8u4x(unsigned char*, unsigned char*, unsigned char*, unsigned char*, long, unsigned char const*, unsigned char const*, unsigned char const*, unsigned char const*, long, long const*, long const*, short const*, int, unsigned int, long, bool)@PLT
        add     rsp, 104
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
.LC2:
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
.LC3:
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
```


- Assembly for source 2
  - compiler: x86-64 GCC 9.4
  - Flags: -std=c++17 -O3 -mavx -mfma -mavx2 -mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store -Wno-deprecated -fvisibility-inlines-hidden -fopenmp -O3 -fPIC -Wno-narrowing -Wall -Wextra -Werror=return-type -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-sign-compare -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function -Wno-unused-result -Wno-unused-local-typedefs -Wno-strict-overflow -Wno-strict-aliasing -Wno-error=deprecated-declarations -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=redundant-decls -Wno-error=old-style-cast -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Werror=cast-function-type -Wno-stringop-overflow
```
.LC4:
        .string "void ImagingResampleHorizontalConvolution8u4x(uint8_t*, uint8_t*, uint8_t*, uint8_t*, int64_t, const uint8_t*, const uint8_t*, const uint8_t*, const uint8_t*, int64_t, const int64_t*, const int64_t*, const int16_t*, int, unsigned int, int64_t, bool)"
.LC5:
        .string "/app/example.cpp"
.LC6:
        .string "stride == 3 || stride == 4"
ImagingResampleHorizontalConvolution8u4x(unsigned char*, unsigned char*, unsigned char*, unsigned char*, long, unsigned char const*, unsigned char const*, unsigned char const*, unsigned char const*, long, long const*, long const*, short const*, int, unsigned int, long, bool):
        push    rbp
        mov     rbp, rsp
        push    r15
        push    r14
        push    r13
        push    r12
        push    rbx
        and     rsp, -32
        add     rsp, -128
        mov     rbx, QWORD PTR 88[rbp]
        mov     QWORD PTR 24[rsp], rdx
        mov     edx, DWORD PTR 96[rbp]
        mov     QWORD PTR 40[rsp], rdi
        mov     edi, DWORD PTR 80[rbp]
        mov     QWORD PTR 32[rsp], rsi
        mov     QWORD PTR 96[rsp], rcx
        mov     QWORD PTR 80[rsp], r8
        mov     QWORD PTR 56[rsp], r9
        mov     BYTE PTR 14[rsp], dl
        cmp     rbx, 3
        je      .L2
        cmp     rbx, 4
        jne     .L36
        mov     QWORD PTR 64[rsp], 1
        vmovdqa ymm4, YMMWORD PTR .LC0[rip]
        mov     rsi, r8
        mov     QWORD PTR 72[rsp], 3
        vmovdqa ymm5, YMMWORD PTR .LC1[rip]
.L3:
        mov     rax, rsi
        lea     ecx, -1[rdi]
        imul    rax, rbx
        mov     QWORD PTR 16[rsp], rax
        mov     rax, QWORD PTR 40[rbp]
        imul    rax, rbx
        mov     QWORD PTR [rsp], rax
        mov     eax, 1
        sal     eax, cl
        vmovd   xmm6, eax
        vpbroadcastd    ymm6, xmm6
        test    rsi, rsi
        jle     .L32
        cmp     rbx, 3
        mov     rsi, QWORD PTR 64[rbp]
        lea     r15, [rbx+rbx]
        vmovd   xmm9, edi
        sete    al
        vpxor   xmm8, xmm8, xmm8
        vpxor   xmm7, xmm7, xmm7
        xor     r13d, r13d
        and     eax, edx
        lea     r12, 0[0+rbx*4]
        mov     QWORD PTR 104[rsp], 0
        mov     BYTE PTR 15[rsp], al
        movsx   rax, DWORD PTR 72[rbp]
        mov     r14, r12
        add     rax, rax
        mov     QWORD PTR 48[rsp], rax
.L21:
        mov     rax, QWORD PTR 104[rsp]
        mov     rdi, QWORD PTR 48[rbp]
        vmovdqa ymm0, ymm6
        xor     edx, edx
        mov     r9, QWORD PTR 16[rbp]
        mov     r8, QWORD PTR 24[rbp]
        vmovdqa ymm1, ymm6
        mov     r12, QWORD PTR [rdi+rax*8]
        mov     rdi, QWORD PTR 56[rbp]
        mov     r11, QWORD PTR [rdi+rax*8]
        mov     rax, QWORD PTR 56[rsp]
        add     r9, r12
        add     r8, r12
        mov     rdi, QWORD PTR 32[rbp]
        mov     rcx, r11
        sub     rcx, QWORD PTR 72[rsp]
        lea     r10, [rax+r12]
        xor     eax, eax
        add     rdi, r12
        test    rcx, rcx
        jle     .L37
.L5:
        vmovdqu xmm3, XMMWORD PTR [r10+rdx]
        vinserti128     ymm3, ymm3, XMMWORD PTR [r9+rdx], 0x1
        vpbroadcastd    ymm11, DWORD PTR [rsi+rax*2]
        vpbroadcastd    ymm10, DWORD PTR 4[rsi+rax*2]
        add     rax, 4
        vpshufb ymm2, ymm3, ymm4
        vpshufb ymm3, ymm3, ymm5
        vpmaddwd        ymm2, ymm2, ymm11
        vpmaddwd        ymm3, ymm3, ymm10
        vpaddd  ymm1, ymm2, ymm1
        vmovdqu xmm2, XMMWORD PTR [r8+rdx]
        vinserti128     ymm2, ymm2, XMMWORD PTR [rdi+rdx], 0x1
        add     rdx, r14
        vpaddd  ymm1, ymm1, ymm3
        vpshufb ymm3, ymm2, ymm4
        vpshufb ymm2, ymm2, ymm5
        vpmaddwd        ymm3, ymm3, ymm11
        vpmaddwd        ymm2, ymm2, ymm10
        vpaddd  ymm0, ymm3, ymm0
        vpaddd  ymm0, ymm0, ymm2
        cmp     rcx, rax
        jg      .L5
.L8:
        mov     rcx, r11
        sub     rcx, QWORD PTR 64[rsp]
        cmp     rcx, rax
        jle     .L6
        mov     rdx, rbx
        imul    rdx, rax
.L11:
        vmovq   xmm2, QWORD PTR [r9+rdx]
        vmovq   xmm3, QWORD PTR [r10+rdx]
        vpbroadcastd    ymm10, DWORD PTR [rsi+rax*2]
        add     rax, 2
        vinserti128     ymm3, ymm3, xmm2, 0x1
        vmovq   xmm2, QWORD PTR [r8+rdx]
        vpshufb ymm3, ymm3, ymm4
        vpmaddwd        ymm3, ymm3, ymm10
        vpaddd  ymm1, ymm3, ymm1
        vmovq   xmm3, QWORD PTR [rdi+rdx]
        add     rdx, r15
        vinserti128     ymm2, ymm2, xmm3, 0x1
        vpshufb ymm2, ymm2, ymm4
        vpmaddwd        ymm2, ymm2, ymm10
        vpaddd  ymm0, ymm2, ymm0
        cmp     rcx, rax
        jg      .L11
.L6:
        sub     r11, 1
        cmp     r11, rax
        jle     .L38
        mov     rcx, rbx
        mov     QWORD PTR 88[rsp], r11
        imul    rcx, rax
.L12:
        movsx   edx, WORD PTR [rsi+rax*2]
        vpmovzxbd       xmm2, DWORD PTR [r9+rcx]
        add     rax, 1
        vpmovzxbd       xmm3, DWORD PTR [r10+rcx]
        vmovd   xmm10, edx
        vinserti128     ymm3, ymm3, xmm2, 0x1
        vpbroadcastd    ymm10, xmm10
        vpmovzxbd       xmm2, DWORD PTR [rdi+rcx]
        vpmaddwd        ymm3, ymm3, ymm10
        vpaddd  ymm1, ymm3, ymm1
        vpmovzxbd       xmm3, DWORD PTR [r8+rcx]
        add     rcx, rbx
        vinserti128     ymm2, ymm3, xmm2, 0x1
        vpmaddwd        ymm2, ymm2, ymm10
        vpaddd  ymm0, ymm2, ymm0
        cmp     rax, r11
        jne     .L12
.L9:
        cmp     r11, QWORD PTR 88[rsp]
        je      .L39
.L13:
        vpsrad  ymm1, ymm1, xmm9
        vpsrad  ymm0, ymm0, xmm9
        mov     r11, QWORD PTR 96[rsp]
        vpackssdw       ymm1, ymm1, ymm8
        vpackssdw       ymm0, ymm0, ymm8
        lea     r8, [r11+r13]
        mov     r11, QWORD PTR 40[rsp]
        vpackuswb       ymm1, ymm1, ymm7
        vpackuswb       ymm0, ymm0, ymm7
        vmovdqa xmm2, xmm1
        vmovd   edi, xmm1
        vextracti128    xmm1, ymm1, 0x1
        vmovd   ecx, xmm1
        vmovd   edx, xmm0
        vmovdqa xmm1, xmm0
        vextracti128    xmm0, ymm0, 0x1
        vmovd   eax, xmm0
        vmovd   DWORD PTR 124[rsp], xmm0
        cmp     rbx, 3
        je      .L40
.L34:
        mov     DWORD PTR [r11+r13], edi
        mov     rdi, QWORD PTR 32[rsp]
        mov     DWORD PTR [rdi+r13], ecx
        mov     rdi, QWORD PTR 24[rsp]
        mov     DWORD PTR [rdi+r13], edx
        mov     rdi, QWORD PTR 96[rsp]
        mov     DWORD PTR [rdi+r13], eax
.L19:
        add     QWORD PTR 104[rsp], 1
        add     r13, rbx
        mov     rax, QWORD PTR 104[rsp]
        add     rsi, QWORD PTR 48[rsp]
        cmp     QWORD PTR 80[rsp], rax
        jne     .L21
.L32:
        vzeroupper
        lea     rsp, -40[rbp]
        pop     rbx
        pop     r12
        pop     r13
        pop     r14
        pop     r15
        pop     rbp
        ret
.L40:
        mov     r11, QWORD PTR 40[rsp]
        lea     r9, 4[r13]
        cmp     QWORD PTR 16[rsp], r9
        jg      .L34
        mov     r9d, DWORD PTR 3[r11+r13]
        vmovd   DWORD PTR [r11+r13], xmm2
        cmp     BYTE PTR 14[rsp], 0
        mov     DWORD PTR 3[r11+r13], r9d
        mov     r11, QWORD PTR 32[rsp]
        mov     edi, DWORD PTR 3[r11+r13]
        mov     DWORD PTR [r11+r13], ecx
        mov     DWORD PTR 3[r11+r13], edi
        mov     rdi, QWORD PTR 24[rsp]
        mov     ecx, DWORD PTR 3[rdi+r13]
        vmovd   DWORD PTR [rdi+r13], xmm1
        mov     DWORD PTR 3[rdi+r13], ecx
        jne     .L41
        mov     rdi, QWORD PTR 96[rsp]
        mov     edx, DWORD PTR 3[rdi+r13]
        vmovd   DWORD PTR [rdi+r13], xmm0
        mov     DWORD PTR 3[rdi+r13], edx
        jmp     .L19
.L39:
        movsx   eax, WORD PTR [rsi+r11*2]
        imul    r11, rbx
        vmovd   xmm10, eax
        vpmovzxbd       xmm2, DWORD PTR [r9+r11]
        vpmovzxbd       xmm3, DWORD PTR [r10+r11]
        vpbroadcastd    ymm10, xmm10
        add     rdi, r11
        cmp     BYTE PTR 15[rsp], 0
        vinserti128     ymm3, ymm3, xmm2, 0x1
        vpmovzxbd       xmm2, DWORD PTR [r8+r11]
        vpmaddwd        ymm3, ymm3, ymm10
        vpaddd  ymm1, ymm3, ymm1
        jne     .L42
.L14:
        vpmovzxbd       xmm3, DWORD PTR [rdi]
.L15:
        vinserti128     ymm2, ymm2, xmm3, 0x1
        vpmaddwd        ymm2, ymm2, ymm10
        vpaddd  ymm0, ymm2, ymm0
        jmp     .L13
.L38:
        mov     QWORD PTR 88[rsp], rax
        jmp     .L9
.L37:
        xor     eax, eax
        jmp     .L8
.L2:
        vmovdqa ymm4, YMMWORD PTR .LC2[rip]
        vmovdqa ymm5, YMMWORD PTR .LC3[rip]
        test    dl, dl
        je      .L43
        mov     QWORD PTR 64[rsp], 2
        mov     rsi, r8
        mov     QWORD PTR 72[rsp], 5
        jmp     .L3
.L43:
        mov     QWORD PTR 64[rsp], 1
        mov     rsi, r8
        mov     QWORD PTR 72[rsp], 4
        jmp     .L3
.L42:
        lea     rax, 4[r11+r12]
        cmp     rax, QWORD PTR [rsp]
        jl      .L14
        movzx   eax, WORD PTR [rdi]
        mov     WORD PTR 124[rsp], ax
        movzx   eax, BYTE PTR 2[rdi]
        mov     BYTE PTR 126[rsp], al
        vpmovzxbd       xmm3, DWORD PTR 124[rsp]
        jmp     .L15
.L41:
        movzx   eax, WORD PTR 124[rsp]
        mov     WORD PTR [r8], ax
        movzx   eax, BYTE PTR 126[rsp]
        mov     BYTE PTR 2[r8], al
        jmp     .L19
.L36:
        lea     rcx, .LC4[rip]
        mov     edx, 76
        lea     rsi, .LC5[rip]
        lea     rdi, .LC6[rip]
        call    __assert_fail@PLT
foo(unsigned char*, unsigned char*, unsigned char*, unsigned char*, long, unsigned char const*, unsigned char const*, unsigned char const*, unsigned char const*, long, long const*, long const*, short const*, int, unsigned int, bool):
        sub     rsp, 16
        movzx   eax, BYTE PTR 96[rsp]
        push    rax
        push    3
        mov     eax, DWORD PTR 104[rsp]
        push    rax
        mov     eax, DWORD PTR 104[rsp]
        push    rax
        push    QWORD PTR 104[rsp]
        push    QWORD PTR 104[rsp]
        push    QWORD PTR 104[rsp]
        push    QWORD PTR 104[rsp]
        push    QWORD PTR 104[rsp]
        push    QWORD PTR 104[rsp]
        push    QWORD PTR 104[rsp]
        call    ImagingResampleHorizontalConvolution8u4x(unsigned char*, unsigned char*, unsigned char*, unsigned char*, long, unsigned char const*, unsigned char const*, unsigned char const*, unsigned char const*, long, long const*, long const*, short const*, int, unsigned int, long, bool)@PLT
        add     rsp, 104
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
.LC2:
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
.LC3:
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
```
