Diff template

- Source 1
```
#include <immintrin.h>
#include <cassert>
#include <cstring>


#define C10_RESTRICT __restrict
#define C10_UNLIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 0))

using uint8_t = unsigned char;
using uint32_t = unsigned int;


static __m128i inline mm_cvtepu8_epi32(const uint8_t* C10_RESTRICT ptr) {
  return _mm_cvtepu8_epi32(_mm_cvtsi32_si128(*(int32_t*)ptr));
}


void ImagingResampleVerticalConvolution8u(
    uint8_t* C10_RESTRICT lineOut,
    const uint8_t* C10_RESTRICT lineIn,
    int64_t xsize,
    int64_t ids_min,
    int64_t ids_size,
    const int16_t* k,
    unsigned int coefs_precision,
    int64_t num_channels) {

  // Interpolation vertical pass processing one line.
  // - We process x-axis data with blocks of 8, 2 and 1
  // - We split the size of weight vector for a given output index as a sum: K = n * 2 + m.

  // xsize = output width, also equals to input width
  // ids_size = interpolation size
  // ids_min = input y start index
  const auto stride = num_channels * 1;  // num channels * sizeof(uint8)

//   TORCH_INTERNAL_ASSERT(stride == 3 || stride == 4);
  assert(stride == 3 || stride == 4);

  const int64_t data_size = xsize * stride;
  const int64_t data_stride = stride;
  constexpr auto vec_size = 256 / 8;

  const auto initial = _mm_set1_epi32(1 << (coefs_precision - 1));
  const auto initial_256 = _mm256_set1_epi32(1 << (coefs_precision - 1));
  const auto zero = _mm_setzero_si128();
  const auto zero_256 = _mm256_setzero_si256();

  int64_t j = 0;
  // block 8
  const auto b8_usable_vec_stride = (vec_size / data_stride) * data_stride;
  for (; j < data_size - vec_size; j += b8_usable_vec_stride) {
    auto sss0 = initial_256;
    auto sss1 = initial_256;
    auto sss2 = initial_256;
    auto sss3 = initial_256;
    int64_t i = 0;
    const auto * lineIn_min = lineIn + j + ids_min;

    for (; i < ids_size - 1; i += 2) {
      // Load 2 values from weight vector
      auto mmk = _mm256_set1_epi32(*(int32_t*)&k[i]);

      // RGBA: Load 8 pixels per line
      // source1 = [
      //    r0 g0 b0 a0  r1 g1 b1 a1  r2 g2 b2 a2  r3 g3 b3 a3
      //    r4 g4 b4 a4  r5 g5 b5 a5  r6 g6 b6 a6  r7 g7 b7 a7
      // ]
      // RGB: Load 8 pixels per line (however we can process only 8 pixels):
      // source1 = [
      //    r0 g0 b0 r1  g1 b1 r2 g2  b2 r3 g3 b3  r4 g4 b4 r5
      //    r4 g4 b4 r5  g5 b5 r6 g6  b6 r7 g7 b7  r8 g8 b8 r9
      // ]
      auto source1 =
          _mm256_loadu_si256((__m256i*)(lineIn_min + data_size * i));
      auto source2 =
          _mm256_loadu_si256((__m256i*)(lineIn_min + data_size * (i + 1)));

      // Interleave source1 and source2 from the low half of each 128-bit lane
      // and cast the result to epi16
      // RGBA: pix1 = [
      //    r0 0 R0 0  g0 0 G0 0  b0 0 B0 0  a0 0 A0 0
      //    r1 0 R1 0  g1 0 G1 0  b1 0 B1 0  a1 0 A1 0
      // ]
      // RGB: pix1 = [
      //    r0 0 R0 0  g0 0 G0 0  b0 0 B0 0  0 0 0 0
      //    r1 0 R1 0  g1 0 G1 0  b1 0 B1 0  0 0 0 0
      // ]
      auto source_lo = _mm256_unpacklo_epi8(source1, source2);
      auto pix1 = _mm256_unpacklo_epi8(source_lo, zero_256);
      // Compute output value as
      //   C += w0 * c0 + w1 * C0
      //   C += w0 * c1 + w1 * C1 for each channel in 32-bit precision
      sss0 = _mm256_add_epi32(sss0, _mm256_madd_epi16(pix1, mmk));

      // RGBA: pix2 = [
      //    r2 0 R2 0  g2 0 G2 0  b2 0 B2 0  a2 0 A2 0
      //    r3 0 R3 0  g3 0 G3 0  b3 0 B3 0  a3 0 A3 0
      // ]
      // RGB: pix2 = [
      //    r2 0 R2 0  g2 0 G2 0  b2 0 B2 0  0 0 0 0
      //    r3 0 R3 0  g3 0 G3 0  b3 0 B3 0  0 0 0 0
      // ]
      auto pix2 = _mm256_unpackhi_epi8(source_lo, zero_256);
      // Compute output value as
      //   C += w0 * c2 + w1 * C2
      //   C += w0 * c3 + w1 * C3 for each channel in 32-bit precision
      sss1 = _mm256_add_epi32(sss1, _mm256_madd_epi16(pix2, mmk));

      // Same as above for the high half of each 128-bit lane
      auto source_hi = _mm256_unpackhi_epi8(source1, source2);
      auto pix3 = _mm256_unpacklo_epi8(source_hi, zero_256);
      sss2 = _mm256_add_epi32(sss2, _mm256_madd_epi16(pix3, mmk));
      auto pix4 = _mm256_unpackhi_epi8(source_hi, zero_256);
      sss3 = _mm256_add_epi32(sss3, _mm256_madd_epi16(pix4, mmk));
    }
    // Same processing as above but with a single weight value
    for (; i < ids_size; i += 1) {
      auto mmk = _mm256_set1_epi32(k[i]);

      auto source1 = _mm256_loadu_si256((__m256i*)(lineIn_min + i * data_size));

      auto source_lo = _mm256_unpacklo_epi8(source1, zero_256);
      auto pix1 = _mm256_unpacklo_epi8(source_lo, zero_256);
      sss0 = _mm256_add_epi32(sss0, _mm256_madd_epi16(pix1, mmk));
      auto pix2 = _mm256_unpackhi_epi8(source_lo, zero_256);
      sss1 = _mm256_add_epi32(sss1, _mm256_madd_epi16(pix2, mmk));

      auto source_hi = _mm256_unpackhi_epi8(source1, zero_256);
      auto pix3 = _mm256_unpacklo_epi8(source_hi, _mm256_setzero_si256());
      sss2 = _mm256_add_epi32(sss2, _mm256_madd_epi16(pix3, mmk));
      auto pix4 = _mm256_unpackhi_epi8(source_hi, _mm256_setzero_si256());
      sss3 = _mm256_add_epi32(sss3, _mm256_madd_epi16(pix4, mmk));
    }
    // Shift back the output integer values: output = output >> weights_precision
    sss0 = _mm256_srai_epi32(sss0, coefs_precision);
    sss1 = _mm256_srai_epi32(sss1, coefs_precision);
    sss2 = _mm256_srai_epi32(sss2, coefs_precision);
    sss3 = _mm256_srai_epi32(sss3, coefs_precision);
    // Convert packed signed 32-bit integers to packed 16-bit integers using signed saturation
    // (a a a a b b b b | c c c c d d d d) -> (a a b b c c d d)
    sss0 = _mm256_packs_epi32(sss0, sss1);
    sss2 = _mm256_packs_epi32(sss2, sss3);
    // Convert packed signed 16-bit integers to packed 8-bit integers using unsigned saturation
    // (a a b b c c d d) -> (a b c d)
    sss0 = _mm256_packus_epi16(sss0, sss2);

    // Stores 32 bytes
    _mm256_storeu_si256((__m256i*)(lineOut + j), sss0);
  }

  // TODO: Do we also need block 4 ???
  // block 2
  const auto b2_usable_vec_stride = (8 / data_stride) * data_stride;
  for (; j < data_size - vec_size / 4; j += b2_usable_vec_stride) {
    auto sss0 = initial;
    auto sss1 = initial;
    int64_t i = 0;
    const auto * lineIn_min = lineIn + j + ids_min;

    for (; i < ids_size - 1; i += 2) {
      // Load 2 values from weight vector
      // mmk = [wl_0 wh_0 wl_1 wh_1  wl_0 wh_0 wl_1 wh_1  ... ]
      auto mmk = _mm_set1_epi32(*(int32_t*)&k[i]);

      // Load 2 pixels per line
      // RGBA: source1 = [
      //    r0 g0 b0 a0  r1 g1 b1 a1  0 0 0 0  0 0 0 0
      // ]
      // RGB: source1 = [
      //    r0 g0 b0 r1  g1 b1 r2 g2  0 0 0 0  0 0 0 0
      // ]
      auto source1 = _mm_loadl_epi64((__m128i *) (lineIn_min + i * data_size));
      auto source2 = _mm_loadl_epi64((__m128i *) (lineIn_min + (i + 1) * data_size));
      // Interleave source1 and source2 and cast the result to epi16
      // RGBA: pix = [
      //    r0 0 R0 0  g0 0 G0 0  b0 0 B0 0  a0 0 A0 0
      // ]
      // RGB: pix = [
      //    r0 0 R0 0  g0 0 G0 0  b0 0 B0 0  0 0 0 0
      // ]
      auto source = _mm_unpacklo_epi8(source1, source2);
      auto pix = _mm_unpacklo_epi8(source, zero);
      // Compute output value as C += w0 * c0 + w1 * C0 for each channel in 32-bit precision
      sss0 = _mm_add_epi32(sss0, _mm_madd_epi16(pix, mmk));
      // RGBA: pix = [
      //    r1 0 R1 0  g1 0 G1 0  b1 0 B1 0  a1 0 A1 0
      // ]
      // RGB: pix = [
      //    r1 0 R1 0  g1 0 G1 0  b1 0 B1 0  0 0 0 0
      // ]
      pix = _mm_unpackhi_epi8(source, zero);
      // Compute output value as C += w0 * c1 + w1 * C1 for each channel in 32-bit precision
      sss1 = _mm_add_epi32(sss1, _mm_madd_epi16(pix, mmk));
    }
    // Same processing as above but with a single weight value
    for (; i < ids_size; i += 1) {
      auto mmk = _mm_set1_epi32(k[i]);

      auto source1 = _mm_loadl_epi64((__m128i*) (lineIn_min + i * data_size));

      auto source = _mm_unpacklo_epi8(source1, zero);
      auto pix1 = _mm_unpacklo_epi8(source, zero);
      sss0 = _mm_add_epi32(sss0, _mm_madd_epi16(pix1, mmk));
      auto pix2 = _mm_unpackhi_epi8(source, zero);
      sss1 = _mm_add_epi32(sss1, _mm_madd_epi16(pix2, mmk));
    }
    // Shift back the output integer values: output = output >> weights_precision
    sss0 = _mm_srai_epi32(sss0, coefs_precision);
    sss1 = _mm_srai_epi32(sss1, coefs_precision);
    // Convert packed signed 32-bit integers to packed 16-bit integers using signed saturation
    // (a a a a b b b b | c c c c d d d d) -> (a a b b c c d d)
    sss0 = _mm_packs_epi32(sss0, sss1);
    // Convert packed signed 16-bit integers to packed 8-bit integers using unsigned saturation
    // (a a b b c c d d) -> (a b c d)
    sss0 = _mm_packus_epi16(sss0, sss0);
    // Store 2 pixels to the output
    _mm_storel_epi64((__m128i*)(lineOut + j), sss0);
  }

  // block 1
  const auto b1_usable_vec_stride = (4 / data_stride) * data_stride;
  for (; j < data_size - 4; j += b1_usable_vec_stride) {
    auto sss = initial;
    int64_t i = 0;
    const auto * lineIn_min = lineIn + j + ids_min;

    for (; i < ids_size - 1; i += 2) {
      // Load 2 values from weight vector
      // mmk = [wl_0 wh_0 wl_1 wh_1  wl_0 wh_0 wl_1 wh_1  ... ]
      auto mmk = _mm_set1_epi32(*(int32_t*)&k[i]);

      // Load one pixel per line
      // RGBA: source1 = [
      //    r0 g0 b0 a0  0 0 0 0  0 0 0 0  0 0 0 0
      // ]
      // RGB: source1 = [
      //    r0 g0 b0 r1  0 0 0 0  0 0 0 0  0 0 0 0
      // ]
      auto source1 = _mm_cvtsi32_si128(*(int32_t*)(lineIn_min + i * data_size));
      auto source2 = _mm_cvtsi32_si128(*(int32_t*)(lineIn_min + (i + 1) * data_size));

      // Interleave source1 and source2 and cast the result to epi16
      // RGBA: pix = [
      //    r0 0 R0 0  g0 0 G0 0  b0 0 B0 0  a0 0 A0 0
      // ]
      // RGB: pix = [
      //    r0 0 R0 0  g0 0 G0 0  b0 0 B0 0  0 0 0 0
      // ]
      auto source = _mm_unpacklo_epi8(source1, source2);
      auto pix = _mm_unpacklo_epi8(source, zero);
      // Compute output value as C += w0 * c0 + w1 * C0 for each channel in 32-bit precision
      sss = _mm_add_epi32(sss, _mm_madd_epi16(pix, mmk));
    }

    for (; i < ids_size; i++) {
      auto mmk = _mm_set1_epi32(k[i]);
      auto pix = mm_cvtepu8_epi32(lineIn_min + i * data_size);
      sss = _mm_add_epi32(sss, _mm_madd_epi16(pix, mmk));
    }
    sss = _mm_srai_epi32(sss, coefs_precision);
    sss = _mm_packs_epi32(sss, zero);
    sss = _mm_packus_epi16(sss, zero);

    auto o = _mm_cvtsi128_si32(sss);

    // Here we write 4 bytes to the output even if num_channels < 4, e.g o = {r,g,b,X} for num_channels=3
    // It is OK to write 4th byte (e.g. X) as on the next step we will overwrite it with new data.
    // We also wont go out of bounds of lineOut memory allocation
    std::memcpy(lineOut + j, (uint8_t *) &o, 4);
  }

  for (; j < data_size; j += data_stride) {
    auto sss = initial;
    int64_t i = 0;
    const auto * lineIn_min = lineIn + j + ids_min;
    // For RGBA we can use (ids_size - 1) as tighter limit but for RGB we can read outside memory boundary
    // for the last remaining line
    for (; i < ids_size - 2; i += 2) {
      // Load two coefficients at once
      auto mmk = _mm_set1_epi32(*(int32_t*)&k[i]);

      // Load 2 lines
      auto source1 = _mm_cvtsi32_si128(*(int32_t*)(lineIn_min + i * data_size));
      auto source2 = _mm_cvtsi32_si128(*(int32_t*)(lineIn_min + (i + 1) * data_size));

      auto source = _mm_unpacklo_epi8(source1, source2);
      auto pix = _mm_unpacklo_epi8(source, zero);
      sss = _mm_add_epi32(sss, _mm_madd_epi16(pix, mmk));
    }

    // Same processing as above but with a single weight value
    for (; i < ids_size; i++) {
      auto mmk = _mm_set1_epi32(k[i]);

      const uint8_t * p = lineIn_min + i * data_size;
      __m128i pix;
      // There is no much perf gain using more detailed condition like
      // num_channels == 3 && ids_min + j + data_size * i + 4 >= in_max_size
      // const int64_t in_max_size = data_size * in_ysize;
      if (num_channels == 3) {
        uint8_t input[4];
        std::memcpy(input, p, 3);
        pix = mm_cvtepu8_epi32(input);
      } else {
        pix = mm_cvtepu8_epi32(p);
      }
      sss = _mm_add_epi32(sss, _mm_madd_epi16(pix, mmk));
    }

    // Shift back the output integer values: output = output >> weights_precision
    sss = _mm_srai_epi32(sss, coefs_precision);
    // Convert packed signed 32-bit integers to packed 16-bit integers using signed saturation
    // (a a a a b b b b | c c c c d d d d) -> (a a b b c c d d)
    sss = _mm_packs_epi32(sss, zero);
    // Convert packed signed 16-bit integers to packed 8-bit integers using unsigned saturation
    // (a a b b c c d d) -> (a b c d)
    sss = _mm_packus_epi16(sss, zero);
    // Store one pixel to the output
    auto o = _mm_cvtsi128_si32(sss);
    if (num_channels == 3 && C10_UNLIKELY(j + 4 >= data_size)) {
      std::memcpy(lineOut + j, (uint8_t *) &o, 3);
    } else {
      std::memcpy(lineOut + j, (uint8_t *) &o, 4);
    }
  }
}


void foo(
    uint8_t* C10_RESTRICT lineOut,
    const uint8_t* C10_RESTRICT lineIn,
    int64_t xsize,
    int64_t ids_min,
    int64_t ids_size,
    const int16_t* k,
    unsigned int coefs_precision)
{
    constexpr int num_channels = 3;

    ImagingResampleVerticalConvolution8u(
        lineOut,
        lineIn,
        xsize,
        ids_min,
        ids_size,
        k,
        coefs_precision,
        num_channels);
}
```


- Source 2
```
#include <immintrin.h>
#include <cassert>
#include <cstring>


#define C10_RESTRICT __restrict
#define C10_UNLIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 0))

using uint8_t = unsigned char;
using uint32_t = unsigned int;


static __m128i inline mm_cvtepu8_epi32(const uint8_t* C10_RESTRICT ptr) {
  return _mm_cvtepu8_epi32(_mm_cvtsi32_si128(*(int32_t*)ptr));
}


void ImagingResampleVerticalConvolution8u(
    uint8_t* C10_RESTRICT lineOut,
    const uint8_t* C10_RESTRICT lineIn,
    int64_t xsize,
    int64_t ids_min,
    int64_t ids_size,
    const int16_t* k,
    unsigned int coefs_precision,
    int64_t num_channels) {

  // Interpolation vertical pass processing one line.
  // - We process x-axis data with blocks of 8, 2 and 1
  // - We split the size of weight vector for a given output index as a sum: K = n * 2 + m.

  // xsize = output width, also equals to input width
  // ids_size = interpolation size
  // ids_min = input y start index
  const auto stride = num_channels * 1;  // num channels * sizeof(uint8)

//   TORCH_INTERNAL_ASSERT(stride == 3 || stride == 4);
  assert(stride == 3 || stride == 4);

  const int64_t data_size = xsize * stride;
  const int64_t data_stride = stride;
  constexpr auto vec_size = 256 / 8;

  const auto initial = _mm_set1_epi32(1 << (coefs_precision - 1));
  const auto initial_256 = _mm256_set1_epi32(1 << (coefs_precision - 1));
  const auto zero = _mm_setzero_si128();
  const auto zero_256 = _mm256_setzero_si256();

  int64_t j = 0;
  // block 8
  const auto b8_usable_vec_stride = (vec_size / data_stride) * data_stride;
  for (; j < data_size - vec_size; j += b8_usable_vec_stride) {
    auto sss0 = initial_256;
    auto sss1 = initial_256;
    auto sss2 = initial_256;
    auto sss3 = initial_256;
    int64_t i = 0;
    const auto * lineIn_min = lineIn + j + ids_min;

    for (; i < ids_size - 1; i += 2) {
      // Load 2 values from weight vector
      auto mmk = _mm256_set1_epi32(*(int32_t*)&k[i]);

      // RGBA: Load 8 pixels per line
      // source1 = [
      //    r0 g0 b0 a0  r1 g1 b1 a1  r2 g2 b2 a2  r3 g3 b3 a3
      //    r4 g4 b4 a4  r5 g5 b5 a5  r6 g6 b6 a6  r7 g7 b7 a7
      // ]
      // RGB: Load 8 pixels per line (however we can process only 8 pixels):
      // source1 = [
      //    r0 g0 b0 r1  g1 b1 r2 g2  b2 r3 g3 b3  r4 g4 b4 r5
      //    r4 g4 b4 r5  g5 b5 r6 g6  b6 r7 g7 b7  r8 g8 b8 r9
      // ]
      auto source1 =
          _mm256_loadu_si256((__m256i*)(lineIn_min + data_size * i));
      auto source2 =
          _mm256_loadu_si256((__m256i*)(lineIn_min + data_size * (i + 1)));

      // Interleave source1 and source2 from the low half of each 128-bit lane
      // and cast the result to epi16
      // RGBA: pix1 = [
      //    r0 0 R0 0  g0 0 G0 0  b0 0 B0 0  a0 0 A0 0
      //    r1 0 R1 0  g1 0 G1 0  b1 0 B1 0  a1 0 A1 0
      // ]
      // RGB: pix1 = [
      //    r0 0 R0 0  g0 0 G0 0  b0 0 B0 0  0 0 0 0
      //    r1 0 R1 0  g1 0 G1 0  b1 0 B1 0  0 0 0 0
      // ]
      auto source_lo = _mm256_unpacklo_epi8(source1, source2);
      auto pix1 = _mm256_unpacklo_epi8(source_lo, zero_256);
      // Compute output value as
      //   C += w0 * c0 + w1 * C0
      //   C += w0 * c1 + w1 * C1 for each channel in 32-bit precision
      sss0 = _mm256_add_epi32(sss0, _mm256_madd_epi16(pix1, mmk));

      // RGBA: pix2 = [
      //    r2 0 R2 0  g2 0 G2 0  b2 0 B2 0  a2 0 A2 0
      //    r3 0 R3 0  g3 0 G3 0  b3 0 B3 0  a3 0 A3 0
      // ]
      // RGB: pix2 = [
      //    r2 0 R2 0  g2 0 G2 0  b2 0 B2 0  0 0 0 0
      //    r3 0 R3 0  g3 0 G3 0  b3 0 B3 0  0 0 0 0
      // ]
      auto pix2 = _mm256_unpackhi_epi8(source_lo, zero_256);
      // Compute output value as
      //   C += w0 * c2 + w1 * C2
      //   C += w0 * c3 + w1 * C3 for each channel in 32-bit precision
      sss1 = _mm256_add_epi32(sss1, _mm256_madd_epi16(pix2, mmk));

      // Same as above for the high half of each 128-bit lane
      auto source_hi = _mm256_unpackhi_epi8(source1, source2);
      auto pix3 = _mm256_unpacklo_epi8(source_hi, zero_256);
      sss2 = _mm256_add_epi32(sss2, _mm256_madd_epi16(pix3, mmk));
      auto pix4 = _mm256_unpackhi_epi8(source_hi, zero_256);
      sss3 = _mm256_add_epi32(sss3, _mm256_madd_epi16(pix4, mmk));
    }
    // Same processing as above but with a single weight value
    for (; i < ids_size; i += 1) {
      auto mmk = _mm256_set1_epi32(k[i]);

      auto source1 = _mm256_loadu_si256((__m256i*)(lineIn_min + i * data_size));

      auto source_lo = _mm256_unpacklo_epi8(source1, zero_256);
      auto pix1 = _mm256_unpacklo_epi8(source_lo, zero_256);
      sss0 = _mm256_add_epi32(sss0, _mm256_madd_epi16(pix1, mmk));
      auto pix2 = _mm256_unpackhi_epi8(source_lo, zero_256);
      sss1 = _mm256_add_epi32(sss1, _mm256_madd_epi16(pix2, mmk));

      auto source_hi = _mm256_unpackhi_epi8(source1, zero_256);
      auto pix3 = _mm256_unpacklo_epi8(source_hi, _mm256_setzero_si256());
      sss2 = _mm256_add_epi32(sss2, _mm256_madd_epi16(pix3, mmk));
      auto pix4 = _mm256_unpackhi_epi8(source_hi, _mm256_setzero_si256());
      sss3 = _mm256_add_epi32(sss3, _mm256_madd_epi16(pix4, mmk));
    }
    // Shift back the output integer values: output = output >> weights_precision
    sss0 = _mm256_srai_epi32(sss0, coefs_precision);
    sss1 = _mm256_srai_epi32(sss1, coefs_precision);
    sss2 = _mm256_srai_epi32(sss2, coefs_precision);
    sss3 = _mm256_srai_epi32(sss3, coefs_precision);
    // Convert packed signed 32-bit integers to packed 16-bit integers using signed saturation
    // (a a a a b b b b | c c c c d d d d) -> (a a b b c c d d)
    sss0 = _mm256_packs_epi32(sss0, sss1);
    sss2 = _mm256_packs_epi32(sss2, sss3);
    // Convert packed signed 16-bit integers to packed 8-bit integers using unsigned saturation
    // (a a b b c c d d) -> (a b c d)
    sss0 = _mm256_packus_epi16(sss0, sss2);

    // Stores 32 bytes
    _mm256_storeu_si256((__m256i*)(lineOut + j), sss0);
  }

  // TODO: Do we also need block 4 ???
  // block 2
  const auto b2_usable_vec_stride = (8 / data_stride) * data_stride;
  for (; j < data_size - vec_size / 4; j += b2_usable_vec_stride) {
    auto sss0 = initial;
    auto sss1 = initial;
    int64_t i = 0;
    const auto * lineIn_min = lineIn + j + ids_min;

    for (; i < ids_size - 1; i += 2) {
      // Load 2 values from weight vector
      // mmk = [wl_0 wh_0 wl_1 wh_1  wl_0 wh_0 wl_1 wh_1  ... ]
      auto mmk = _mm_set1_epi32(*(int32_t*)&k[i]);

      // Load 2 pixels per line
      // RGBA: source1 = [
      //    r0 g0 b0 a0  r1 g1 b1 a1  0 0 0 0  0 0 0 0
      // ]
      // RGB: source1 = [
      //    r0 g0 b0 r1  g1 b1 r2 g2  0 0 0 0  0 0 0 0
      // ]
      auto source1 = _mm_loadl_epi64((__m128i *) (lineIn_min + i * data_size));
      auto source2 = _mm_loadl_epi64((__m128i *) (lineIn_min + (i + 1) * data_size));
      // Interleave source1 and source2 and cast the result to epi16
      // RGBA: pix = [
      //    r0 0 R0 0  g0 0 G0 0  b0 0 B0 0  a0 0 A0 0
      // ]
      // RGB: pix = [
      //    r0 0 R0 0  g0 0 G0 0  b0 0 B0 0  0 0 0 0
      // ]
      auto source = _mm_unpacklo_epi8(source1, source2);
      auto pix = _mm_unpacklo_epi8(source, zero);
      // Compute output value as C += w0 * c0 + w1 * C0 for each channel in 32-bit precision
      sss0 = _mm_add_epi32(sss0, _mm_madd_epi16(pix, mmk));
      // RGBA: pix = [
      //    r1 0 R1 0  g1 0 G1 0  b1 0 B1 0  a1 0 A1 0
      // ]
      // RGB: pix = [
      //    r1 0 R1 0  g1 0 G1 0  b1 0 B1 0  0 0 0 0
      // ]
      pix = _mm_unpackhi_epi8(source, zero);
      // Compute output value as C += w0 * c1 + w1 * C1 for each channel in 32-bit precision
      sss1 = _mm_add_epi32(sss1, _mm_madd_epi16(pix, mmk));
    }
    // Same processing as above but with a single weight value
    for (; i < ids_size; i += 1) {
      auto mmk = _mm_set1_epi32(k[i]);

      auto source1 = _mm_loadl_epi64((__m128i*) (lineIn_min + i * data_size));

      auto source = _mm_unpacklo_epi8(source1, zero);
      auto pix1 = _mm_unpacklo_epi8(source, zero);
      sss0 = _mm_add_epi32(sss0, _mm_madd_epi16(pix1, mmk));
      auto pix2 = _mm_unpackhi_epi8(source, zero);
      sss1 = _mm_add_epi32(sss1, _mm_madd_epi16(pix2, mmk));
    }
    // Shift back the output integer values: output = output >> weights_precision
    sss0 = _mm_srai_epi32(sss0, coefs_precision);
    sss1 = _mm_srai_epi32(sss1, coefs_precision);
    // Convert packed signed 32-bit integers to packed 16-bit integers using signed saturation
    // (a a a a b b b b | c c c c d d d d) -> (a a b b c c d d)
    sss0 = _mm_packs_epi32(sss0, sss1);
    // Convert packed signed 16-bit integers to packed 8-bit integers using unsigned saturation
    // (a a b b c c d d) -> (a b c d)
    sss0 = _mm_packus_epi16(sss0, sss0);
    // Store 2 pixels to the output
    _mm_storel_epi64((__m128i*)(lineOut + j), sss0);
  }

  // block 1
  const auto b1_usable_vec_stride = (4 / data_stride) * data_stride;
  for (; j < data_size - 4; j += b1_usable_vec_stride) {
    auto sss = initial;
    int64_t i = 0;
    const auto * lineIn_min = lineIn + j + ids_min;

    for (; i < ids_size - 1; i += 2) {
      // Load 2 values from weight vector
      // mmk = [wl_0 wh_0 wl_1 wh_1  wl_0 wh_0 wl_1 wh_1  ... ]
      auto mmk = _mm_set1_epi32(*(int32_t*)&k[i]);

      // Load one pixel per line
      // RGBA: source1 = [
      //    r0 g0 b0 a0  0 0 0 0  0 0 0 0  0 0 0 0
      // ]
      // RGB: source1 = [
      //    r0 g0 b0 r1  0 0 0 0  0 0 0 0  0 0 0 0
      // ]
      auto source1 = _mm_cvtsi32_si128(*(int32_t*)(lineIn_min + i * data_size));
      auto source2 = _mm_cvtsi32_si128(*(int32_t*)(lineIn_min + (i + 1) * data_size));

      // Interleave source1 and source2 and cast the result to epi16
      // RGBA: pix = [
      //    r0 0 R0 0  g0 0 G0 0  b0 0 B0 0  a0 0 A0 0
      // ]
      // RGB: pix = [
      //    r0 0 R0 0  g0 0 G0 0  b0 0 B0 0  0 0 0 0
      // ]
      auto source = _mm_unpacklo_epi8(source1, source2);
      auto pix = _mm_unpacklo_epi8(source, zero);
      // Compute output value as C += w0 * c0 + w1 * C0 for each channel in 32-bit precision
      sss = _mm_add_epi32(sss, _mm_madd_epi16(pix, mmk));
    }

    for (; i < ids_size; i++) {
      auto mmk = _mm_set1_epi32(k[i]);
      auto pix = mm_cvtepu8_epi32(lineIn_min + i * data_size);
      sss = _mm_add_epi32(sss, _mm_madd_epi16(pix, mmk));
    }
    sss = _mm_srai_epi32(sss, coefs_precision);
    sss = _mm_packs_epi32(sss, zero);
    sss = _mm_packus_epi16(sss, zero);

    auto o = _mm_cvtsi128_si32(sss);

    // Here we write 4 bytes to the output even if num_channels < 4, e.g o = {r,g,b,X} for num_channels=3
    // It is OK to write 4th byte (e.g. X) as on the next step we will overwrite it with new data.
    // We also wont go out of bounds of lineOut memory allocation
    std::memcpy(lineOut + j, (uint8_t *) &o, 4);
  }

  for (; j < data_size; j += data_stride) {
    auto sss = initial;
    int64_t i = 0;
    const auto * lineIn_min = lineIn + j + ids_min;
    // For RGBA we can use (ids_size - 1) as tighter limit but for RGB we can read outside memory boundary
    // for the last remaining line
    for (; i < ids_size - 2; i += 2) {
      // Load two coefficients at once
      auto mmk = _mm_set1_epi32(*(int32_t*)&k[i]);

      // Load 2 lines
      auto source1 = _mm_cvtsi32_si128(*(int32_t*)(lineIn_min + i * data_size));
      auto source2 = _mm_cvtsi32_si128(*(int32_t*)(lineIn_min + (i + 1) * data_size));

      auto source = _mm_unpacklo_epi8(source1, source2);
      auto pix = _mm_unpacklo_epi8(source, zero);
      sss = _mm_add_epi32(sss, _mm_madd_epi16(pix, mmk));
    }

    // Same processing as above but with a single weight value
    for (; i < ids_size; i++) {
      auto mmk = _mm_set1_epi32(k[i]);

      const uint8_t * p = lineIn_min + i * data_size;
      __m128i pix;
      // There is no much perf gain using more detailed condition like
      // num_channels == 3 && ids_min + j + data_size * i + 4 >= in_max_size
      // const int64_t in_max_size = data_size * in_ysize;
      if (num_channels == 3) {
        uint8_t input[4];
        std::memcpy(input, p, 3);
        pix = mm_cvtepu8_epi32(input);
      } else {
        pix = mm_cvtepu8_epi32(p);
      }
      sss = _mm_add_epi32(sss, _mm_madd_epi16(pix, mmk));
    }

    // Shift back the output integer values: output = output >> weights_precision
    sss = _mm_srai_epi32(sss, coefs_precision);
    // Convert packed signed 32-bit integers to packed 16-bit integers using signed saturation
    // (a a a a b b b b | c c c c d d d d) -> (a a b b c c d d)
    sss = _mm_packs_epi32(sss, zero);
    // Convert packed signed 16-bit integers to packed 8-bit integers using unsigned saturation
    // (a a b b c c d d) -> (a b c d)
    sss = _mm_packus_epi16(sss, zero);
    // Store one pixel to the output
    auto o = _mm_cvtsi128_si32(sss);
    if (num_channels == 3 && C10_UNLIKELY(j + 4 >= data_size)) {
      std::memcpy(lineOut + j, (uint8_t *) &o, 3);
    } else {
      std::memcpy(lineOut + j, (uint8_t *) &o, 4);
    }
  }
}


void foo(
    uint8_t* C10_RESTRICT lineOut,
    const uint8_t* C10_RESTRICT lineIn,
    int64_t xsize,
    int64_t ids_min,
    int64_t ids_size,
    const int16_t* k,
    unsigned int coefs_precision)
{
    int num_channels = 3;

    ImagingResampleVerticalConvolution8u(
        lineOut,
        lineIn,
        xsize,
        ids_min,
        ids_size,
        k,
        coefs_precision,
        num_channels);
}
```


- Assembly for source 1
  - compiler: x86-64 GCC 9.4
  - Flags: -std=c++17 -O3 -mavx -mfma -mavx2 -mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store -Wno-deprecated -fvisibility-inlines-hidden -fopenmp -O3 -fPIC -Wno-narrowing -Wall -Wextra -Werror=return-type -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-sign-compare -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function -Wno-unused-result -Wno-unused-local-typedefs -Wno-strict-overflow -Wno-strict-aliasing -Wno-error=deprecated-declarations -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=redundant-decls -Wno-error=old-style-cast -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Werror=cast-function-type -Wno-stringop-overflow
```
.LC0:
        .string "void ImagingResampleVerticalConvolution8u(uint8_t*, const uint8_t*, int64_t, int64_t, int64_t, const int16_t*, unsigned int, int64_t)"
.LC1:
        .string "/app/example.cpp"
.LC2:
        .string "stride == 3 || stride == 4"
ImagingResampleVerticalConvolution8u(unsigned char*, unsigned char const*, long, long, long, short const*, unsigned int, long):
        push    rbp
        mov     rbp, rsp
        push    r15
        push    r14
        push    r13
        push    r12
        push    rbx
        and     rsp, -32
        sub     rsp, 64
        mov     r10, QWORD PTR 24[rbp]
        mov     QWORD PTR 32[rsp], rsi
        lea     rax, -3[r10]
        mov     QWORD PTR 24[rsp], rcx
        cmp     rax, 1
        ja      .L62
        mov     eax, DWORD PTR 16[rbp]
        mov     rsi, rdx
        mov     r14, rdi
        imul    rsi, r10
        lea     ecx, -1[rax]
        mov     eax, 1
        sal     eax, cl
        vmovd   xmm2, eax
        vmovd   xmm10, eax
        mov     eax, 32
        lea     r15, -32[rsi]
        vpshufd xmm2, xmm2, 0
        vpbroadcastd    ymm10, xmm10
        cqo
        idiv    r10
        imul    rax, r10
        test    r15, r15
        jle     .L39
        lea     rdx, -2[r8]
        mov     QWORD PTR 24[rbp], r10
        lea     rdi, [rsi+rsi]
        vpxor   xmm5, xmm5, xmm5
        shr     rdx
        vmovd   xmm1, DWORD PTR 16[rbp]
        mov     r12, QWORD PTR 32[rsp]
        lea     rcx, 4[r9+rdx*4]
        lea     rdx, 2[rdx+rdx]
        add     r12, QWORD PTR 24[rsp]
        mov     QWORD PTR 40[rsp], rdx
        xor     edx, edx
.L9:
        cmp     r8, 1
        jle     .L63
        mov     r11, r12
        vmovdqa ymm8, ymm10
        vmovdqa ymm7, ymm10
        mov     r10, r9
        vmovdqa ymm9, ymm10
        vmovdqa ymm6, ymm10
.L4:
        vmovdqu ymm3, YMMWORD PTR [r11]
        vpunpcklbw      ymm0, ymm3, YMMWORD PTR [r11+rsi]
        add     r10, 4
        vpbroadcastd    ymm3, DWORD PTR -4[r10]
        vpunpcklbw      ymm4, ymm0, ymm5
        vpunpckhbw      ymm0, ymm0, ymm5
        vpmaddwd        ymm4, ymm4, ymm3
        vpmaddwd        ymm0, ymm0, ymm3
        vpaddd  ymm6, ymm4, ymm6
        vpaddd  ymm9, ymm0, ymm9
        vmovdqu ymm4, YMMWORD PTR [r11]
        vpunpckhbw      ymm0, ymm4, YMMWORD PTR [r11+rsi]
        add     r11, rdi
        vpunpcklbw      ymm4, ymm0, ymm5
        vpunpckhbw      ymm0, ymm0, ymm5
        vpmaddwd        ymm4, ymm4, ymm3
        vpmaddwd        ymm0, ymm0, ymm3
        vpaddd  ymm7, ymm4, ymm7
        vpaddd  ymm8, ymm0, ymm8
        cmp     r10, rcx
        jne     .L4
        mov     r11, QWORD PTR 40[rsp]
.L7:
        cmp     r8, r11
        jle     .L5
        mov     rbx, rsi
        movsx   r10d, WORD PTR [r9+r11*2]
        lea     r13, [r11+r11]
        imul    rbx, r11
        add     r11, 1
        vmovd   xmm11, r10d
        vpbroadcastd    ymm11, xmm11
        vmovdqu ymm3, YMMWORD PTR [r12+rbx]
        vmovdqu ymm0, YMMWORD PTR [r12+rbx]
        vpunpcklbw      ymm3, ymm3, ymm5
        vpunpckhbw      ymm0, ymm0, ymm5
        vpunpcklbw      ymm4, ymm3, ymm5
        vpunpcklbw      ymm12, ymm0, ymm5
        vpunpckhbw      ymm3, ymm3, ymm5
        vpunpckhbw      ymm0, ymm0, ymm5
        vpmaddwd        ymm4, ymm4, ymm11
        vpmaddwd        ymm3, ymm3, ymm11
        vpmaddwd        ymm12, ymm12, ymm11
        vpmaddwd        ymm0, ymm0, ymm11
        vpaddd  ymm4, ymm4, ymm6
        vpaddd  ymm3, ymm3, ymm9
        vpaddd  ymm12, ymm12, ymm7
        vpaddd  ymm0, ymm0, ymm8
        vmovdqa ymm6, ymm4
        vmovdqa ymm9, ymm3
        vmovdqa ymm7, ymm12
        vmovdqa ymm8, ymm0
        cmp     r8, r11
        jle     .L5
        lea     r11, [r12+rsi]
        movsx   r10d, WORD PTR 2[r9+r13]
        vmovdqu ymm8, YMMWORD PTR [r11+rbx]
        vmovd   xmm11, r10d
        vpunpcklbw      ymm9, ymm8, ymm5
        vpunpckhbw      ymm8, ymm8, ymm5
        vpbroadcastd    ymm11, xmm11
        vpunpcklbw      ymm6, ymm9, ymm5
        vpunpcklbw      ymm7, ymm8, ymm5
        vpunpckhbw      ymm9, ymm9, ymm5
        vpunpckhbw      ymm8, ymm8, ymm5
        vpmaddwd        ymm6, ymm6, ymm11
        vpmaddwd        ymm9, ymm9, ymm11
        vpmaddwd        ymm7, ymm7, ymm11
        vpmaddwd        ymm8, ymm8, ymm11
        vpaddd  ymm6, ymm6, ymm4
        vpaddd  ymm9, ymm9, ymm3
        vpaddd  ymm7, ymm7, ymm12
        vpaddd  ymm8, ymm8, ymm0
.L5:
        vpsrad  ymm6, ymm6, xmm1
        vpsrad  ymm9, ymm9, xmm1
        add     r12, rax
        vpsrad  ymm7, ymm7, xmm1
        vpsrad  ymm8, ymm8, xmm1
        vpackssdw       ymm6, ymm6, ymm9
        vpackssdw       ymm7, ymm7, ymm8
        vpackuswb       ymm6, ymm6, ymm7
        vmovdqu YMMWORD PTR [r14+rdx], ymm6
        add     rdx, rax
        cmp     rdx, r15
        jl      .L9
        mov     r10, QWORD PTR 24[rbp]
.L3:
        lea     r15, -8[rsi]
        lea     r12, [r10+r10]
        cmp     r15, rdx
        jle     .L18
        lea     rdi, -2[r8]
        mov     QWORD PTR 24[rbp], r10
        lea     rcx, [rsi+rsi]
        vpxor   xmm7, xmm7, xmm7
        shr     rdi
        mov     rax, QWORD PTR 24[rsp]
        vmovd   xmm1, DWORD PTR 16[rbp]
        lea     rbx, 2[rdi+rdi]
        mov     QWORD PTR 40[rsp], rbx
        lea     r11, [rdx+rax]
        lea     rax, 4[r9+rdi*4]
        add     r11, QWORD PTR 32[rsp]
.L19:
        vmovdqa xmm6, xmm2
        vmovdqa xmm5, xmm2
        xor     edi, edi
        cmp     r8, 1
        jle     .L16
        mov     r10, r11
        mov     rdi, r9
.L13:
        vmovq   xmm3, QWORD PTR [r10+rsi]
        vmovq   xmm0, QWORD PTR [r10]
        add     rdi, 4
        add     r10, rcx
        vpunpcklbw      xmm0, xmm0, xmm3
        vbroadcastss    xmm3, DWORD PTR -4[rdi]
        vpunpcklbw      xmm4, xmm0, xmm7
        vpunpckhbw      xmm0, xmm0, xmm7
        vpmaddwd        xmm4, xmm4, xmm3
        vpmaddwd        xmm0, xmm0, xmm3
        vpaddd  xmm5, xmm4, xmm5
        vpaddd  xmm6, xmm0, xmm6
        cmp     rax, rdi
        jne     .L13
        mov     rdi, QWORD PTR 40[rsp]
.L16:
        cmp     r8, rdi
        jle     .L14
        mov     rbx, rsi
        movsx   r10d, WORD PTR [r9+rdi*2]
        lea     r13, [rdi+rdi]
        imul    rbx, rdi
        add     rdi, 1
        vmovd   xmm4, r10d
        vpshufd xmm4, xmm4, 0
        vmovq   xmm0, QWORD PTR [r11+rbx]
        vpunpcklbw      xmm0, xmm0, xmm7
        vpunpcklbw      xmm3, xmm0, xmm7
        vpunpckhbw      xmm0, xmm0, xmm7
        vpmaddwd        xmm3, xmm3, xmm4
        vpmaddwd        xmm0, xmm0, xmm4
        vpaddd  xmm3, xmm3, xmm5
        vpaddd  xmm0, xmm0, xmm6
        vmovdqa xmm5, xmm3
        vmovdqa xmm6, xmm0
        cmp     r8, rdi
        jle     .L14
        lea     r10, [r11+rsi]
        movsx   edi, WORD PTR 2[r9+r13]
        vmovq   xmm6, QWORD PTR [r10+rbx]
        vmovd   xmm4, edi
        vpunpcklbw      xmm6, xmm6, xmm7
        vpshufd xmm4, xmm4, 0
        vpunpcklbw      xmm5, xmm6, xmm7
        vpunpckhbw      xmm6, xmm6, xmm7
        vpmaddwd        xmm5, xmm5, xmm4
        vpmaddwd        xmm6, xmm6, xmm4
        vpaddd  xmm5, xmm5, xmm3
        vpaddd  xmm6, xmm6, xmm0
.L14:
        vpsrad  xmm5, xmm5, xmm1
        vpsrad  xmm6, xmm6, xmm1
        add     r11, r12
        vpackssdw       xmm5, xmm5, xmm6
        vpackuswb       xmm5, xmm5, xmm5
        vmovq   QWORD PTR [r14+rdx], xmm5
        add     rdx, r12
        cmp     rdx, r15
        jl      .L19
        mov     r10, QWORD PTR 24[rbp]
.L18:
        lea     r13, -4[rsi]
        cmp     r13, rdx
        jle     .L11
        mov     rax, QWORD PTR 24[rsp]
        lea     rdi, -2[r8]
        lea     rcx, [rsi+rsi]
        vpxor   xmm5, xmm5, xmm5
        shr     rdi
        vmovd   xmm1, DWORD PTR 16[rbp]
        vpxor   xmm7, xmm7, xmm7
        vpxor   xmm6, xmm6, xmm6
        lea     r11, [rdx+rax]
        lea     r15, 2[rdi+rdi]
        add     r11, QWORD PTR 32[rsp]
        lea     rax, 4[r9+rdi*4]
.L26:
        mov     rbx, r11
        mov     rdi, r9
        vmovdqa xmm3, xmm2
        cmp     r8, 1
        jle     .L64
.L21:
        vmovd   xmm4, DWORD PTR [rbx+rsi]
        vmovd   xmm0, DWORD PTR [rbx]
        add     rdi, 4
        add     rbx, rcx
        vpunpcklbw      xmm0, xmm0, xmm4
        vbroadcastss    xmm4, DWORD PTR -4[rdi]
        vpunpcklbw      xmm0, xmm0, xmm5
        vpmaddwd        xmm0, xmm0, xmm4
        vpaddd  xmm3, xmm0, xmm3
        cmp     rax, rdi
        jne     .L21
        mov     rdi, r15
.L24:
        cmp     r8, rdi
        jle     .L22
        movsx   r12d, WORD PTR [r9+rdi*2]
        lea     rbx, [rdi+rdi]
        mov     DWORD PTR 40[rsp], r12d
        mov     r12, rsi
        vbroadcastss    xmm4, DWORD PTR 40[rsp]
        imul    r12, rdi
        add     rdi, 1
        vpmovzxbd       xmm0, DWORD PTR [r11+r12]
        vpmaddwd        xmm0, xmm0, xmm4
        vpaddd  xmm3, xmm0, xmm3
        cmp     r8, rdi
        jle     .L22
        movsx   edi, WORD PTR 2[r9+rbx]
        lea     rbx, [r11+rsi]
        vpmovzxbd       xmm0, DWORD PTR [rbx+r12]
        vmovd   xmm4, edi
        vpshufd xmm4, xmm4, 0
        vpmaddwd        xmm0, xmm0, xmm4
        vpaddd  xmm3, xmm3, xmm0
.L22:
        vpsrad  xmm3, xmm3, xmm1
        add     r11, r10
        vpackssdw       xmm3, xmm3, xmm7
        vpackuswb       xmm3, xmm3, xmm6
        vmovd   DWORD PTR [r14+rdx], xmm3
        add     rdx, r10
        cmp     rdx, r13
        jl      .L26
.L11:
        cmp     rsi, rdx
        jle     .L59
        lea     rbx, [r9+r8*2]
        mov     r11, QWORD PTR 24[rsp]
        lea     rdi, -3[r8]
        vmovd   xmm1, DWORD PTR 16[rbp]
        mov     QWORD PTR 24[rsp], rbx
        shr     rdi
        lea     rcx, [rsi+rsi]
        vpxor   xmm4, xmm4, xmm4
        lea     rax, 4[r9+rdi*4]
        vpxor   xmm6, xmm6, xmm6
        vpxor   xmm5, xmm5, xmm5
        add     r11, rdx
        lea     r13, 2[rdi+rdi]
        add     r11, QWORD PTR 32[rsp]
.L38:
        mov     rbx, r11
        mov     rdi, r9
        vmovdqa xmm3, xmm2
        cmp     r8, 2
        jle     .L65
.L27:
        vmovd   xmm7, DWORD PTR [rbx+rsi]
        vmovd   xmm0, DWORD PTR [rbx]
        add     rdi, 4
        add     rbx, rcx
        vpunpcklbw      xmm0, xmm0, xmm7
        vbroadcastss    xmm7, DWORD PTR -4[rdi]
        vpunpcklbw      xmm0, xmm0, xmm4
        vpmaddwd        xmm0, xmm0, xmm7
        vpaddd  xmm3, xmm0, xmm3
        cmp     rdi, rax
        jne     .L27
        mov     rdi, r13
.L30:
        cmp     r8, rdi
        jle     .L66
        cmp     r10, 3
        je      .L31
        mov     r12, rsi
        movsx   ebx, WORD PTR [r9+rdi*2]
        lea     r15, [rdi+rdi]
        imul    r12, rdi
        vmovd   xmm7, ebx
        lea     rbx, 1[rdi]
        vpshufd xmm7, xmm7, 0
        vpmovzxbd       xmm0, DWORD PTR [r11+r12]
        vpmaddwd        xmm0, xmm0, xmm7
        vpaddd  xmm3, xmm0, xmm3
        vmovdqa xmm0, xmm3
        cmp     r8, rbx
        jle     .L32
        movsx   ebx, WORD PTR 2[r9+r15]
        add     r12, rsi
        add     rdi, 2
        vpmovzxbd       xmm0, DWORD PTR [r11+r12]
        vmovd   xmm7, ebx
        vpshufd xmm7, xmm7, 0
        vpmaddwd        xmm0, xmm0, xmm7
        vpaddd  xmm3, xmm0, xmm3
        vmovdqa xmm0, xmm3
        cmp     r8, rdi
        jle     .L32
        movsx   edi, WORD PTR 4[r9+r15]
        lea     rbx, [r11+rsi]
        vpmovzxbd       xmm0, DWORD PTR [rbx+r12]
        vmovd   xmm7, edi
        vpshufd xmm7, xmm7, 0
        vpmaddwd        xmm0, xmm0, xmm7
        vpaddd  xmm0, xmm0, xmm3
.L32:
        vpsrad  xmm0, xmm0, xmm1
        vpackssdw       xmm0, xmm0, xmm6
        vpackuswb       xmm0, xmm0, xmm5
        vmovd   edi, xmm0
.L33:
        mov     DWORD PTR [r14+rdx], edi
.L36:
        add     rdx, r10
        add     r11, r10
        cmp     rsi, rdx
        jg      .L38
.L59:
        vzeroupper
        lea     rsp, -40[rbp]
        pop     rbx
        pop     r12
        pop     r13
        pop     r14
        pop     r15
        pop     rbp
        ret
.L66:
        vpsrad  xmm0, xmm3, xmm1
        lea     r12, [r14+rdx]
        vpackssdw       xmm0, xmm0, xmm6
        vpackuswb       xmm0, xmm0, xmm5
        vmovd   edi, xmm0
        vmovd   DWORD PTR 60[rsp], xmm0
        cmp     r10, 3
        jne     .L33
.L35:
        lea     rbx, 4[rdx]
        cmp     rsi, rbx
        jg      .L33
        movzx   edi, WORD PTR 60[rsp]
        mov     WORD PTR [r12], di
        movzx   edi, BYTE PTR 62[rsp]
        mov     BYTE PTR 2[r12], dil
        jmp     .L36
.L31:
        lea     r12, [r9+rdi*2]
        mov     QWORD PTR 32[rsp], rax
        mov     rbx, QWORD PTR 24[rsp]
        lea     r15, 60[rsp]
        imul    rdi, rsi
        add     rdi, r11
.L34:
        movsx   eax, WORD PTR [r12]
        add     r12, 2
        mov     DWORD PTR 40[rsp], eax
        movzx   eax, WORD PTR [rdi]
        vbroadcastss    xmm7, DWORD PTR 40[rsp]
        mov     WORD PTR [r15], ax
        movzx   eax, BYTE PTR 2[rdi]
        add     rdi, rsi
        mov     BYTE PTR 2[r15], al
        vpmovzxbd       xmm0, DWORD PTR 60[rsp]
        vpmaddwd        xmm0, xmm0, xmm7
        vpaddd  xmm0, xmm0, xmm3
        vmovdqa xmm3, xmm0
        cmp     rbx, r12
        jne     .L34
        vpsrad  xmm0, xmm0, xmm1
        mov     rax, QWORD PTR 32[rsp]
        lea     r12, [r14+rdx]
        vpackssdw       xmm0, xmm0, xmm6
        vpackuswb       xmm0, xmm0, xmm5
        vmovd   edi, xmm0
        vmovd   DWORD PTR 60[rsp], xmm0
        jmp     .L35
.L65:
        xor     edi, edi
        jmp     .L30
.L64:
        xor     edi, edi
        jmp     .L24
.L63:
        vmovdqa ymm8, ymm10
        vmovdqa ymm7, ymm10
        vmovdqa ymm9, ymm10
        xor     r11d, r11d
        vmovdqa ymm6, ymm10
        jmp     .L7
.L39:
        xor     edx, edx
        jmp     .L3
.L62:
        lea     rcx, .LC0[rip]
        mov     edx, 38
        lea     rsi, .LC1[rip]
        lea     rdi, .LC2[rip]
        call    __assert_fail@PLT
foo(unsigned char*, unsigned char const*, long, long, long, short const*, unsigned int):
        sub     rsp, 8
        push    3
        mov     eax, DWORD PTR 24[rsp]
        push    rax
        call    ImagingResampleVerticalConvolution8u(unsigned char*, unsigned char const*, long, long, long, short const*, unsigned int, long)@PLT
        add     rsp, 24
        ret
```

- Assembly for source 2
  - compiler: x86-64 GCC 9.4
  - Flags: -std=c++17 -O3 -mavx -mfma -mavx2 -mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store -Wno-deprecated -fvisibility-inlines-hidden -fopenmp -O3 -fPIC -Wno-narrowing -Wall -Wextra -Werror=return-type -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-sign-compare -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function -Wno-unused-result -Wno-unused-local-typedefs -Wno-strict-overflow -Wno-strict-aliasing -Wno-error=deprecated-declarations -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=redundant-decls -Wno-error=old-style-cast -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Werror=cast-function-type -Wno-stringop-overflow
```
.LC0:
        .string "void ImagingResampleVerticalConvolution8u(uint8_t*, const uint8_t*, int64_t, int64_t, int64_t, const int16_t*, unsigned int, int64_t)"
.LC1:
        .string "/app/example.cpp"
.LC2:
        .string "stride == 3 || stride == 4"
ImagingResampleVerticalConvolution8u(unsigned char*, unsigned char const*, long, long, long, short const*, unsigned int, long):
        push    rbp
        mov     rbp, rsp
        push    r15
        push    r14
        push    r13
        push    r12
        push    rbx
        and     rsp, -32
        sub     rsp, 64
        mov     r10, QWORD PTR 24[rbp]
        mov     QWORD PTR 32[rsp], rsi
        lea     rax, -3[r10]
        mov     QWORD PTR 24[rsp], rcx
        cmp     rax, 1
        ja      .L62
        mov     eax, DWORD PTR 16[rbp]
        mov     rsi, rdx
        mov     r14, rdi
        imul    rsi, r10
        lea     ecx, -1[rax]
        mov     eax, 1
        sal     eax, cl
        vmovd   xmm2, eax
        vmovd   xmm10, eax
        mov     eax, 32
        lea     r15, -32[rsi]
        vpshufd xmm2, xmm2, 0
        vpbroadcastd    ymm10, xmm10
        cqo
        idiv    r10
        imul    rax, r10
        test    r15, r15
        jle     .L39
        lea     rdx, -2[r8]
        mov     QWORD PTR 24[rbp], r10
        lea     rdi, [rsi+rsi]
        vpxor   xmm5, xmm5, xmm5
        shr     rdx
        vmovd   xmm1, DWORD PTR 16[rbp]
        mov     r12, QWORD PTR 32[rsp]
        lea     rcx, 4[r9+rdx*4]
        lea     rdx, 2[rdx+rdx]
        add     r12, QWORD PTR 24[rsp]
        mov     QWORD PTR 40[rsp], rdx
        xor     edx, edx
.L9:
        cmp     r8, 1
        jle     .L63
        mov     r11, r12
        vmovdqa ymm8, ymm10
        vmovdqa ymm7, ymm10
        mov     r10, r9
        vmovdqa ymm9, ymm10
        vmovdqa ymm6, ymm10
.L4:
        vmovdqu ymm3, YMMWORD PTR [r11]
        vpunpcklbw      ymm0, ymm3, YMMWORD PTR [r11+rsi]
        add     r10, 4
        vpbroadcastd    ymm3, DWORD PTR -4[r10]
        vpunpcklbw      ymm4, ymm0, ymm5
        vpunpckhbw      ymm0, ymm0, ymm5
        vpmaddwd        ymm4, ymm4, ymm3
        vpmaddwd        ymm0, ymm0, ymm3
        vpaddd  ymm6, ymm4, ymm6
        vpaddd  ymm9, ymm0, ymm9
        vmovdqu ymm4, YMMWORD PTR [r11]
        vpunpckhbw      ymm0, ymm4, YMMWORD PTR [r11+rsi]
        add     r11, rdi
        vpunpcklbw      ymm4, ymm0, ymm5
        vpunpckhbw      ymm0, ymm0, ymm5
        vpmaddwd        ymm4, ymm4, ymm3
        vpmaddwd        ymm0, ymm0, ymm3
        vpaddd  ymm7, ymm4, ymm7
        vpaddd  ymm8, ymm0, ymm8
        cmp     r10, rcx
        jne     .L4
        mov     r11, QWORD PTR 40[rsp]
.L7:
        cmp     r8, r11
        jle     .L5
        mov     rbx, rsi
        movsx   r10d, WORD PTR [r9+r11*2]
        lea     r13, [r11+r11]
        imul    rbx, r11
        add     r11, 1
        vmovd   xmm11, r10d
        vpbroadcastd    ymm11, xmm11
        vmovdqu ymm3, YMMWORD PTR [r12+rbx]
        vmovdqu ymm0, YMMWORD PTR [r12+rbx]
        vpunpcklbw      ymm3, ymm3, ymm5
        vpunpckhbw      ymm0, ymm0, ymm5
        vpunpcklbw      ymm4, ymm3, ymm5
        vpunpcklbw      ymm12, ymm0, ymm5
        vpunpckhbw      ymm3, ymm3, ymm5
        vpunpckhbw      ymm0, ymm0, ymm5
        vpmaddwd        ymm4, ymm4, ymm11
        vpmaddwd        ymm3, ymm3, ymm11
        vpmaddwd        ymm12, ymm12, ymm11
        vpmaddwd        ymm0, ymm0, ymm11
        vpaddd  ymm4, ymm4, ymm6
        vpaddd  ymm3, ymm3, ymm9
        vpaddd  ymm12, ymm12, ymm7
        vpaddd  ymm0, ymm0, ymm8
        vmovdqa ymm6, ymm4
        vmovdqa ymm9, ymm3
        vmovdqa ymm7, ymm12
        vmovdqa ymm8, ymm0
        cmp     r8, r11
        jle     .L5
        lea     r11, [r12+rsi]
        movsx   r10d, WORD PTR 2[r9+r13]
        vmovdqu ymm8, YMMWORD PTR [r11+rbx]
        vmovd   xmm11, r10d
        vpunpcklbw      ymm9, ymm8, ymm5
        vpunpckhbw      ymm8, ymm8, ymm5
        vpbroadcastd    ymm11, xmm11
        vpunpcklbw      ymm6, ymm9, ymm5
        vpunpcklbw      ymm7, ymm8, ymm5
        vpunpckhbw      ymm9, ymm9, ymm5
        vpunpckhbw      ymm8, ymm8, ymm5
        vpmaddwd        ymm6, ymm6, ymm11
        vpmaddwd        ymm9, ymm9, ymm11
        vpmaddwd        ymm7, ymm7, ymm11
        vpmaddwd        ymm8, ymm8, ymm11
        vpaddd  ymm6, ymm6, ymm4
        vpaddd  ymm9, ymm9, ymm3
        vpaddd  ymm7, ymm7, ymm12
        vpaddd  ymm8, ymm8, ymm0
.L5:
        vpsrad  ymm6, ymm6, xmm1
        vpsrad  ymm9, ymm9, xmm1
        add     r12, rax
        vpsrad  ymm7, ymm7, xmm1
        vpsrad  ymm8, ymm8, xmm1
        vpackssdw       ymm6, ymm6, ymm9
        vpackssdw       ymm7, ymm7, ymm8
        vpackuswb       ymm6, ymm6, ymm7
        vmovdqu YMMWORD PTR [r14+rdx], ymm6
        add     rdx, rax
        cmp     rdx, r15
        jl      .L9
        mov     r10, QWORD PTR 24[rbp]
.L3:
        lea     r15, -8[rsi]
        lea     r12, [r10+r10]
        cmp     r15, rdx
        jle     .L18
        lea     rdi, -2[r8]
        mov     QWORD PTR 24[rbp], r10
        lea     rcx, [rsi+rsi]
        vpxor   xmm7, xmm7, xmm7
        shr     rdi
        mov     rax, QWORD PTR 24[rsp]
        vmovd   xmm1, DWORD PTR 16[rbp]
        lea     rbx, 2[rdi+rdi]
        mov     QWORD PTR 40[rsp], rbx
        lea     r11, [rdx+rax]
        lea     rax, 4[r9+rdi*4]
        add     r11, QWORD PTR 32[rsp]
.L19:
        vmovdqa xmm6, xmm2
        vmovdqa xmm5, xmm2
        xor     edi, edi
        cmp     r8, 1
        jle     .L16
        mov     r10, r11
        mov     rdi, r9
.L13:
        vmovq   xmm3, QWORD PTR [r10+rsi]
        vmovq   xmm0, QWORD PTR [r10]
        add     rdi, 4
        add     r10, rcx
        vpunpcklbw      xmm0, xmm0, xmm3
        vbroadcastss    xmm3, DWORD PTR -4[rdi]
        vpunpcklbw      xmm4, xmm0, xmm7
        vpunpckhbw      xmm0, xmm0, xmm7
        vpmaddwd        xmm4, xmm4, xmm3
        vpmaddwd        xmm0, xmm0, xmm3
        vpaddd  xmm5, xmm4, xmm5
        vpaddd  xmm6, xmm0, xmm6
        cmp     rax, rdi
        jne     .L13
        mov     rdi, QWORD PTR 40[rsp]
.L16:
        cmp     r8, rdi
        jle     .L14
        mov     rbx, rsi
        movsx   r10d, WORD PTR [r9+rdi*2]
        lea     r13, [rdi+rdi]
        imul    rbx, rdi
        add     rdi, 1
        vmovd   xmm4, r10d
        vpshufd xmm4, xmm4, 0
        vmovq   xmm0, QWORD PTR [r11+rbx]
        vpunpcklbw      xmm0, xmm0, xmm7
        vpunpcklbw      xmm3, xmm0, xmm7
        vpunpckhbw      xmm0, xmm0, xmm7
        vpmaddwd        xmm3, xmm3, xmm4
        vpmaddwd        xmm0, xmm0, xmm4
        vpaddd  xmm3, xmm3, xmm5
        vpaddd  xmm0, xmm0, xmm6
        vmovdqa xmm5, xmm3
        vmovdqa xmm6, xmm0
        cmp     r8, rdi
        jle     .L14
        lea     r10, [r11+rsi]
        movsx   edi, WORD PTR 2[r9+r13]
        vmovq   xmm6, QWORD PTR [r10+rbx]
        vmovd   xmm4, edi
        vpunpcklbw      xmm6, xmm6, xmm7
        vpshufd xmm4, xmm4, 0
        vpunpcklbw      xmm5, xmm6, xmm7
        vpunpckhbw      xmm6, xmm6, xmm7
        vpmaddwd        xmm5, xmm5, xmm4
        vpmaddwd        xmm6, xmm6, xmm4
        vpaddd  xmm5, xmm5, xmm3
        vpaddd  xmm6, xmm6, xmm0
.L14:
        vpsrad  xmm5, xmm5, xmm1
        vpsrad  xmm6, xmm6, xmm1
        add     r11, r12
        vpackssdw       xmm5, xmm5, xmm6
        vpackuswb       xmm5, xmm5, xmm5
        vmovq   QWORD PTR [r14+rdx], xmm5
        add     rdx, r12
        cmp     rdx, r15
        jl      .L19
        mov     r10, QWORD PTR 24[rbp]
.L18:
        lea     r13, -4[rsi]
        cmp     r13, rdx
        jle     .L11
        mov     rax, QWORD PTR 24[rsp]
        lea     rdi, -2[r8]
        lea     rcx, [rsi+rsi]
        vpxor   xmm5, xmm5, xmm5
        shr     rdi
        vmovd   xmm1, DWORD PTR 16[rbp]
        vpxor   xmm7, xmm7, xmm7
        vpxor   xmm6, xmm6, xmm6
        lea     r11, [rdx+rax]
        lea     r15, 2[rdi+rdi]
        add     r11, QWORD PTR 32[rsp]
        lea     rax, 4[r9+rdi*4]
.L26:
        mov     rbx, r11
        mov     rdi, r9
        vmovdqa xmm3, xmm2
        cmp     r8, 1
        jle     .L64
.L21:
        vmovd   xmm4, DWORD PTR [rbx+rsi]
        vmovd   xmm0, DWORD PTR [rbx]
        add     rdi, 4
        add     rbx, rcx
        vpunpcklbw      xmm0, xmm0, xmm4
        vbroadcastss    xmm4, DWORD PTR -4[rdi]
        vpunpcklbw      xmm0, xmm0, xmm5
        vpmaddwd        xmm0, xmm0, xmm4
        vpaddd  xmm3, xmm0, xmm3
        cmp     rax, rdi
        jne     .L21
        mov     rdi, r15
.L24:
        cmp     r8, rdi
        jle     .L22
        movsx   r12d, WORD PTR [r9+rdi*2]
        lea     rbx, [rdi+rdi]
        mov     DWORD PTR 40[rsp], r12d
        mov     r12, rsi
        vbroadcastss    xmm4, DWORD PTR 40[rsp]
        imul    r12, rdi
        add     rdi, 1
        vpmovzxbd       xmm0, DWORD PTR [r11+r12]
        vpmaddwd        xmm0, xmm0, xmm4
        vpaddd  xmm3, xmm0, xmm3
        cmp     r8, rdi
        jle     .L22
        movsx   edi, WORD PTR 2[r9+rbx]
        lea     rbx, [r11+rsi]
        vpmovzxbd       xmm0, DWORD PTR [rbx+r12]
        vmovd   xmm4, edi
        vpshufd xmm4, xmm4, 0
        vpmaddwd        xmm0, xmm0, xmm4
        vpaddd  xmm3, xmm3, xmm0
.L22:
        vpsrad  xmm3, xmm3, xmm1
        add     r11, r10
        vpackssdw       xmm3, xmm3, xmm7
        vpackuswb       xmm3, xmm3, xmm6
        vmovd   DWORD PTR [r14+rdx], xmm3
        add     rdx, r10
        cmp     rdx, r13
        jl      .L26
.L11:
        cmp     rsi, rdx
        jle     .L59
        lea     rbx, [r9+r8*2]
        mov     r11, QWORD PTR 24[rsp]
        lea     rdi, -3[r8]
        vmovd   xmm1, DWORD PTR 16[rbp]
        mov     QWORD PTR 24[rsp], rbx
        shr     rdi
        lea     rcx, [rsi+rsi]
        vpxor   xmm4, xmm4, xmm4
        lea     rax, 4[r9+rdi*4]
        vpxor   xmm6, xmm6, xmm6
        vpxor   xmm5, xmm5, xmm5
        add     r11, rdx
        lea     r13, 2[rdi+rdi]
        add     r11, QWORD PTR 32[rsp]
.L38:
        mov     rbx, r11
        mov     rdi, r9
        vmovdqa xmm3, xmm2
        cmp     r8, 2
        jle     .L65
.L27:
        vmovd   xmm7, DWORD PTR [rbx+rsi]
        vmovd   xmm0, DWORD PTR [rbx]
        add     rdi, 4
        add     rbx, rcx
        vpunpcklbw      xmm0, xmm0, xmm7
        vbroadcastss    xmm7, DWORD PTR -4[rdi]
        vpunpcklbw      xmm0, xmm0, xmm4
        vpmaddwd        xmm0, xmm0, xmm7
        vpaddd  xmm3, xmm0, xmm3
        cmp     rdi, rax
        jne     .L27
        mov     rdi, r13
.L30:
        cmp     r8, rdi
        jle     .L66
        cmp     r10, 3
        je      .L31
        mov     r12, rsi
        movsx   ebx, WORD PTR [r9+rdi*2]
        lea     r15, [rdi+rdi]
        imul    r12, rdi
        vmovd   xmm7, ebx
        lea     rbx, 1[rdi]
        vpshufd xmm7, xmm7, 0
        vpmovzxbd       xmm0, DWORD PTR [r11+r12]
        vpmaddwd        xmm0, xmm0, xmm7
        vpaddd  xmm3, xmm0, xmm3
        vmovdqa xmm0, xmm3
        cmp     r8, rbx
        jle     .L32
        movsx   ebx, WORD PTR 2[r9+r15]
        add     r12, rsi
        add     rdi, 2
        vpmovzxbd       xmm0, DWORD PTR [r11+r12]
        vmovd   xmm7, ebx
        vpshufd xmm7, xmm7, 0
        vpmaddwd        xmm0, xmm0, xmm7
        vpaddd  xmm3, xmm0, xmm3
        vmovdqa xmm0, xmm3
        cmp     r8, rdi
        jle     .L32
        movsx   edi, WORD PTR 4[r9+r15]
        lea     rbx, [r11+rsi]
        vpmovzxbd       xmm0, DWORD PTR [rbx+r12]
        vmovd   xmm7, edi
        vpshufd xmm7, xmm7, 0
        vpmaddwd        xmm0, xmm0, xmm7
        vpaddd  xmm0, xmm0, xmm3
.L32:
        vpsrad  xmm0, xmm0, xmm1
        vpackssdw       xmm0, xmm0, xmm6
        vpackuswb       xmm0, xmm0, xmm5
        vmovd   edi, xmm0
.L33:
        mov     DWORD PTR [r14+rdx], edi
.L36:
        add     rdx, r10
        add     r11, r10
        cmp     rsi, rdx
        jg      .L38
.L59:
        vzeroupper
        lea     rsp, -40[rbp]
        pop     rbx
        pop     r12
        pop     r13
        pop     r14
        pop     r15
        pop     rbp
        ret
.L66:
        vpsrad  xmm0, xmm3, xmm1
        lea     r12, [r14+rdx]
        vpackssdw       xmm0, xmm0, xmm6
        vpackuswb       xmm0, xmm0, xmm5
        vmovd   edi, xmm0
        vmovd   DWORD PTR 60[rsp], xmm0
        cmp     r10, 3
        jne     .L33
.L35:
        lea     rbx, 4[rdx]
        cmp     rsi, rbx
        jg      .L33
        movzx   edi, WORD PTR 60[rsp]
        mov     WORD PTR [r12], di
        movzx   edi, BYTE PTR 62[rsp]
        mov     BYTE PTR 2[r12], dil
        jmp     .L36
.L31:
        lea     r12, [r9+rdi*2]
        mov     QWORD PTR 32[rsp], rax
        mov     rbx, QWORD PTR 24[rsp]
        lea     r15, 60[rsp]
        imul    rdi, rsi
        add     rdi, r11
.L34:
        movsx   eax, WORD PTR [r12]
        add     r12, 2
        mov     DWORD PTR 40[rsp], eax
        movzx   eax, WORD PTR [rdi]
        vbroadcastss    xmm7, DWORD PTR 40[rsp]
        mov     WORD PTR [r15], ax
        movzx   eax, BYTE PTR 2[rdi]
        add     rdi, rsi
        mov     BYTE PTR 2[r15], al
        vpmovzxbd       xmm0, DWORD PTR 60[rsp]
        vpmaddwd        xmm0, xmm0, xmm7
        vpaddd  xmm0, xmm0, xmm3
        vmovdqa xmm3, xmm0
        cmp     rbx, r12
        jne     .L34
        vpsrad  xmm0, xmm0, xmm1
        mov     rax, QWORD PTR 32[rsp]
        lea     r12, [r14+rdx]
        vpackssdw       xmm0, xmm0, xmm6
        vpackuswb       xmm0, xmm0, xmm5
        vmovd   edi, xmm0
        vmovd   DWORD PTR 60[rsp], xmm0
        jmp     .L35
.L65:
        xor     edi, edi
        jmp     .L30
.L64:
        xor     edi, edi
        jmp     .L24
.L63:
        vmovdqa ymm8, ymm10
        vmovdqa ymm7, ymm10
        vmovdqa ymm9, ymm10
        xor     r11d, r11d
        vmovdqa ymm6, ymm10
        jmp     .L7
.L39:
        xor     edx, edx
        jmp     .L3
.L62:
        lea     rcx, .LC0[rip]
        mov     edx, 38
        lea     rsi, .LC1[rip]
        lea     rdi, .LC2[rip]
        call    __assert_fail@PLT
foo(unsigned char*, unsigned char const*, long, long, long, short const*, unsigned int):
        sub     rsp, 8
        push    3
        mov     eax, DWORD PTR 24[rsp]
        push    rax
        call    ImagingResampleVerticalConvolution8u(unsigned char*, unsigned char const*, long, long, long, short const*, unsigned int, long)@PLT
        add     rsp, 24
        ret
```