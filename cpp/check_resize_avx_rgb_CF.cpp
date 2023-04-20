// Vectorized interpolation for uint8_t RGB Channels First

#include <iostream>
#include <immintrin.h>
#include <cassert>
#include <cstring>
#include <algorithm>


#define TORCH_INTERNAL_ASSERT assert
#define C10_RESTRICT __restrict
#define C10_UNLIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 0))

using uint8_t = unsigned char;
using uint32_t = unsigned int;


template<typename scalar_t=uint8_t>
void print_m256i(__m256i value, std::string tag="") {

    constexpr int64_t size = 256 / (sizeof(scalar_t) * 8);
    __attribute__ ((aligned (32))) scalar_t aligned_output[size];

    _mm256_storeu_si256((__m256i *)aligned_output, value);

    std::cout << "print_m256i " << tag << ": ";
    for (int i=0; i < size; i++) {
        std::cout << (int64_t) aligned_output[i] << " ";
    }
    std::cout << std::endl;
}


void print_m256(__m256 value, std::string tag="") {

    using scalar_t = float;

    constexpr int64_t size = 256 / (sizeof(scalar_t) * 8);
    __attribute__ ((aligned (32))) scalar_t aligned_output[size];

    _mm256_storeu_ps(aligned_output, value);

    std::cout << "print_m256 " << tag << ": ";
    for (int i=0; i < size; i++) {
        std::cout << (float) aligned_output[i] << " ";
    }
    std::cout << std::endl;
}


void print_m256d(__m256d value, std::string tag="") {

    constexpr int64_t size = 256 / (sizeof(double) * 8);
    __attribute__ ((aligned (64))) double aligned_output[size];

    _mm256_storeu_pd(aligned_output, value);

    std::cout << "print_m256d " << tag << ": ";
    for (int i=0; i < size; i++) {
        std::cout << (double) aligned_output[i] << " ";
    }
    std::cout << std::endl;
}

template<typename scalar_t=uint8_t>
void print_data(scalar_t * data, int size, std::string tag="") {
    std::cout << "\nprint_data " << tag << ": ";
    for (int i=0; i < size; i++) {
        std::cout << (int64_t) data[i] << " ";
    }
    std::cout << std::endl;
}


using INT32 = int;
using INT16 = short int;
using UINT32 = unsigned int;


void print_uint32(UINT32 value, const std::string & tag="") {
    unsigned char * data = (unsigned char *) &value;
    std::cout << "print_uint32 " << tag << ": ";
    for (int i=0; i < 4; i++) {
        std::cout << (int) data[i] << " ";
    }
    std::cout << std::endl;
}


static __m128i inline mm_cvtepu8_epi32(void *ptr) {
    return _mm_cvtepu8_epi32(_mm_cvtsi32_si128(*(INT32 *) ptr));
}


static __m256i inline mm256_cvtepu8_epi32(void *ptr) {
    return _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i *) ptr));
}


template<typename scalar_t=uint8_t>
void print_m128i(__m128i value, std::string tag="") {

    constexpr int64_t size = 128 / (sizeof(scalar_t) * 8);
    __attribute__ ((aligned (16))) scalar_t aligned_output[size];

    _mm_storeu_si128((__m128i *)aligned_output, value);

    std::cout << "print_m128i " << tag << ": ";
    for (int i=0; i < size; i++) {
        std::cout << (int64_t) aligned_output[i] << " ";
    }
    std::cout << std::endl;
}

static inline __m128i mm_cvt_si128(const uint8_t* C10_RESTRICT ptr, int n) {
  int32_t v;
  if (n == 2) {
    std::memcpy(&v, ptr, n);
  } else if (n == 3) {
    std::memcpy(&v, ptr, n);
  } else if (n == 4) {
    std::memcpy(&v, ptr, n);
  } else {
    TORCH_INTERNAL_ASSERT(false);
  }
  return _mm_cvtsi32_si128(v);
}

static inline __m128i mm_cvt_multi_si128(
    const uint8_t* C10_RESTRICT ptr0,
    const uint8_t* C10_RESTRICT ptr1,
    const uint8_t* C10_RESTRICT ptr2,
    const uint8_t* C10_RESTRICT ptr3) {

  uint8_t data[4] = {*ptr0, *ptr1, *ptr2, *ptr3};
  return _mm_cvtsi32_si128(*(int32_t *)data);
}

// Vertical pass: ImagingResampleVerticalConvolution8u
void test_ImagingResampleVerticalConvolution8u()
{
    // Vertical pass: ImagingResampleVerticalConvolution8u
    // Try to adapt the code for RGB Channels First data

    // h, w, c = 4, 20, 3
    // a = list(range(h * w * c))
    // s = w * h
    // for i in range(c):
    //     b = []
    //     for v in a[i::c]:
    //         b.append(v)
    //     print(b)

    constexpr int num_channels = 3;
    constexpr int width = 20;
    constexpr int height = 4;
    constexpr int out_height = 2;
    constexpr int out_width = width;

    unsigned char input[height * width * num_channels] = {
        0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63, 66, 69, 72, 75, 78, 81, 84, 87, 90, 93, 96, 99, 102, 105, 108, 111, 114, 117, 120, 123, 126, 129, 132, 135, 138, 141, 144, 147, 150, 153, 156, 159, 162, 165, 168, 171, 174, 177, 180, 183, 186, 189, 192, 195, 198, 201, 204, 207, 210, 213, 216, 219, 222, 225, 228, 231, 234, 237,
        1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 52, 55, 58, 61, 64, 67, 70, 73, 76, 79, 82, 85, 88, 91, 94, 97, 100, 103, 106, 109, 112, 115, 118, 121, 124, 127, 130, 133, 136, 139, 142, 145, 148, 151, 154, 157, 160, 163, 166, 169, 172, 175, 178, 181, 184, 187, 190, 193, 196, 199, 202, 205, 208, 211, 214, 217, 220, 223, 226, 229, 232, 235, 238,
        2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50, 53, 56, 59, 62, 65, 68, 71, 74, 77, 80, 83, 86, 89, 92, 95, 98, 101, 104, 107, 110, 113, 116, 119, 122, 125, 128, 131, 134, 137, 140, 143, 146, 149, 152, 155, 158, 161, 164, 167, 170, 173, 176, 179, 182, 185, 188, 191, 194, 197, 200, 203, 206, 209, 212, 215, 218, 221, 224, 227, 230, 233, 236, 239
    };

    unsigned char output[out_height * out_width * num_channels] = {};

    char * src = (char *) input;

    print_data(&input[0], height * width, "input, c0");
    print_data(&input[height * width], height * width, "input, c1");
    print_data(&input[2 * height * width], height * width, "input, c2");
    printf("\n");

    // (20, 4)
    // coefs_precision=16, ksize=5
    // x=0, xsize=20, xmin=0, xmax=3
    // k: 28087 28087 9362
    // x=0, xsize=20, xmin=1, xmax=3
    // k: 9362 28087 28087
    // (20, 2)

    // int xsize = 20;
    INT16 kk[5 * 4] = {
        28087, 28087, 9362, 0, 0,
        9362, 28087, 28087, 0, 0,
    };
    int ksize = 5;
    int kmax = ksize;

    int weights_precision = 16;

    for (int tt = 0; tt < num_channels * out_height; tt++) {

        std::cout << "--- tt= " << tt << std::endl;

        auto y = tt % out_height;
        std::cout << "    y= " << y << std::endl;
        auto c = tt / out_height;
        std::cout << "    c= " << c << std::endl;

        char * dst = (char *) output + tt * out_width;

        // int xmin, xmax, x;
        // index stride is constant for the given dimension
        const int64_t ids_stride = width;
        int64_t ids_min = y * width + c * width * height;
        int64_t ids_size = 3;

        // xsize = output width, xx = output x index
        // xmax = interpolation size, x = interpolation index (vertical <-> y dimension)
        // xmin = input y start index

        const int16_t *wts_ptr = &kk[y * ksize];
        // const int64_t wts_idx = *(int64_t*)&data[2 + 4][0];
        // const int16_t* wts_ptr = (int16_t*)&data[2 + 3][wts_idx];
        {
            std::cout << "Weights as int16: " << std::endl;
            for (int i=0; i < kmax; i++) {
                std::cout << (int) kk[i] << " ";
            }
            std::cout << std::endl;
            for (int i=kmax; i < 2 * kmax; i++) {
                std::cout << (int) kk[i] << " ";
            }
            std::cout << std::endl;

            std::cout << "Weights as uint8_t: " << std::endl;
            unsigned char * k_ui8 = (unsigned char *) kk;
            for (int i=0; i < 2 * kmax; i++) {
                std::cout << (int) k_ui8[i] << " ";
            }
            std::cout << std::endl;
            for (int i=kmax; i < 4 * kmax; i++) {
                std::cout << (int) k_ui8[i] << " ";
            }
            std::cout << std::endl;
        }

        const int64_t strides[2] = {sizeof(uint8_t), sizeof(uint8_t)};

        const int64_t n = out_width;
        constexpr auto vec_size = 256 / 8;

        int64_t i = 0;

        const auto initial = _mm_set1_epi32(1 << (weights_precision - 1));
        const auto initial_256 = _mm256_set1_epi32(1 << (weights_precision - 1));
        const auto zero = _mm_setzero_si128();
        const auto zero_256 = _mm256_setzero_si256();

        const auto mask_src_b4_1 = _mm_set_epi8(
            -1, -1, -1, 3, -1, -1, -1, 2, -1, -1, -1, 1, -1, -1, -1, 0);
        const auto mask_src_b8_1 = _mm256_set_epi8(
            -1, -1, -1, 7, -1, -1, -1, 6, -1, -1, -1, 5, -1, -1, -1, 4,
            -1, -1, -1, 3, -1, -1, -1, 2, -1, -1, -1, 1, -1, -1, -1, 0);
        const auto mask_src_b8_2 = _mm256_set_epi8(
            -1, 15, -1, 14, -1, 13, -1, 12, -1, 11, -1, 10, -1, 9, -1, 8,
            -1, 7, -1, 6, -1, 5, -1, 4, -1, 3, -1, 2, -1, 1, -1, 0);

        // Block 8: Read 8 values from input
        for (; i < n - 7; i += 8) {
          auto sss = initial_256;

          int64_t j = 0;
          char* src_min = src + i * strides[1] + ids_min;

          for (; j < ids_size - 1; j += 2) {
            // wts = [
            //    w0_l w0_h w1_l w1_h  w0_l w0_h w1_l w1_h ...
            //    w0_l w0_h w1_l w1_h  w0_l w0_h w1_l w1_h ...
            // ]
            auto wts = _mm256_set1_epi32(*(int32_t*)&wts_ptr[j]);

            std::cout << "b8 -- j=" << j << std::endl;
            print_m256i(wts, "wts");

            // Read 8 bytes, 4 bytes per line:
            // source1 = [r0 r1 r2 r3  r4 r5 r6 r7  0 0 0 0  0 0 0 0]
            // source2 = [R0 R1 R2 R3  R4 R5 R6 R7  0 0 0 0  0 0 0 0]
            // tmp = [r0 R0 r1 R1  r2 R2 r3 R3  r4 R4 r5 R5  r6 R6 r7 R7]
            // source = [
            //     r0 R0 r1 R1  r2 R2 r3 R3  r4 R4 r5 R5  r6 R6 r7 R7
            //     r0 R0 r1 R1  r2 R2 r3 R3  r4 R4 r5 R5  r6 R6 r7 R7
            // ]
            // pix = [r0 0 R0 0  r1 0 R1 0  r2 0 R2 0  r3 0 R3 0]
            auto source1 = _mm_loadl_epi64((__m128i *) &src_min[(j + 0) * ids_stride]);
            auto source2 = _mm_loadl_epi64((__m128i *) &src_min[(j + 1) * ids_stride]);
            auto tmp = _mm_unpacklo_epi8(source1, source2);
            print_m128i(tmp, "tmp");
            auto source = _mm256_inserti128_si256(_mm256_castsi128_si256(tmp), tmp, 1);
            print_m256i(source, "source");

            auto pix = _mm256_shuffle_epi8(source, mask_src_b8_2);
            print_m256i(pix, "pix");

            // sss = [
            //    (r0 * w0) + (R0 * w1)   as int16
            //    (r1 * w0) + (R1 * w1)   as int16
            //    (r2 * w0) + (R2 * w1)   as int16
            //    (r3 * w0) + (R3 * w1)   as int16
            // ]
            sss = _mm256_add_epi32(sss, _mm256_madd_epi16(pix, wts));
          }

          for (; j < ids_size; j++) {
            // wts = [
            //    w0_l w0_h 0 0  w0_l w0_h 0 0   ...
            //    w0_l w0_h 0 0  w0_l w0_h 0 0   ...
            // ]
            auto wts = _mm256_set1_epi32(wts_ptr[j]);
            std::cout << "b8 - j=" << j << std::endl;
            print_m256i(wts, "wts");

            // Read 8 bytes:
            // source = [
            //     r0 r1 r2 r3  r4 r5 r6 r7  0 0 0 0  0 0 0 0
            //     r0 r1 r2 r3  r4 r5 r6 r7  0 0 0 0  0 0 0 0
            // ]
            //
            // pix = [
            //    r0 0 0 0  r1 0 0 0  r2 0 0 0  r3 0 0 0
            //    r4 0 0 0  r5 0 0 0  r6 0 0 0  r7 0 0 0
            // ]
            auto tmp = _mm_loadl_epi64((__m128i *) &src_min[j * ids_stride]);
            auto source = _mm256_inserti128_si256(_mm256_castsi128_si256(tmp), tmp, 1);
            print_m256i(source, "source");

            auto pix = _mm256_shuffle_epi8(source, mask_src_b8_1);
            print_m256i(pix, "pix");

            sss = _mm256_add_epi32(sss, _mm256_madd_epi16(pix, wts));
            print_m256i(sss, "sss");
          }
          print_m256i(sss, "1 sss");
          sss = _mm256_srai_epi32(sss, weights_precision);
          print_m256i(sss, "2 sss");
          sss = _mm256_packs_epi32(sss, zero_256);
          print_m256i(sss, "3 sss");
          sss = _mm256_packus_epi16(sss, zero_256);
          print_m256i(sss, "4 sss");

          auto o1 = _mm_cvtsi128_si32(_mm256_extracti128_si256(sss, 0));
          auto o2 = _mm_cvtsi128_si32(_mm256_extracti128_si256(sss, 1));
          std::memcpy(&dst[(i + 0) * strides[0]], (uint8_t *) &o1, 4);
          std::memcpy(&dst[(i + 4) * strides[0]], (uint8_t *) &o2, 4);
        }

        // Block 4: Read 4 values from input
        for (; i < n - 3; i += 4) {
            auto sss = initial;

            int64_t j = 0;
            char* src_min = src + i * strides[1] + ids_min;

            for (; j < ids_size - 1; j += 2) {
              // wts = [
              //    w0_l w0_h w1_l w1_h  w0_l w0_h w1_l w1_h
              //    w0_l w0_h w1_l w1_h  w0_l w0_h w1_l w1_h
              // ]
              auto wts = _mm_set1_epi32(*(int32_t*)&wts_ptr[j]);
              std::cout << "b4 -- j=" << j << std::endl;
              print_m128i(wts, "wts");

              // Read 8 bytes, 4 bytes per line:
              // source1 = [r0 r1 r2 r3  0 0 0 0  0 0 0 0  0 0 0 0]
              // source2 = [R0 R1 R2 R3  0 0 0 0  0 0 0 0  0 0 0 0]
              // source = [r0 R0 r1 R1  r2 R2 r3 R3  0 0 0 0  0 0 0 0]
              // pix = [r0 0 R0 0  r1 0 R1 0  r2 0 R2 0  r3 0 R3 0]
              auto source1 = mm_cvt_si128((const uint8_t *) &src_min[(j + 0) * ids_stride], 4);
              auto source2 = mm_cvt_si128((const uint8_t *) &src_min[(j + 1) * ids_stride], 4);
              auto source = _mm_unpacklo_epi8(source1, source2);
              auto pix = _mm_unpacklo_epi8(source, zero);
              print_m128i(pix, "pix");
              // sss = [
              //    (r0 * w0) + (R0 * w1)   as int16
              //    (r1 * w0) + (R1 * w1)   as int16
              //    (r2 * w0) + (R2 * w1)   as int16
              //    (r3 * w0) + (R3 * w1)   as int16
              // ]
              sss = _mm_add_epi32(sss, _mm_madd_epi16(pix, wts));
            }

            for (; j < ids_size; j++) {

                // wts = [w0_l w0_h 0 0  w0_l w0_h 0 0  ...]
                auto wts = _mm_set1_epi32(wts_ptr[j]);

                std::cout << "b4 - j=" << j << std::endl;
                print_m128i(wts, "wts");
                // Read 4 bytes:
                // source = [r0 r1 r2 r3  0 0 0 0  0 0 0 0  0 0 0 0]
                // pix    = [r0 0 0 0    r1 0 0 0 r2 0 0 0 r3 0 0 0]
                auto source = mm_cvt_si128((const uint8_t *) &src_min[j * ids_stride], 4);
                print_m128i(source, "source");
                auto pix = _mm_shuffle_epi8(source, mask_src_b4_1);
                print_m128i(pix, "pix");

                // NOT SURE ABOUT THIS COMMENT
                // sss += [(r0 * w0) 0 0  (r1 * w0) 0 0  (r2 * w0) 0 0  (r3 * w0) 0 0]
                sss = _mm_add_epi32(sss, _mm_madd_epi16(pix, wts));
                print_m128i(sss, "sss");
            }
            print_m128i(sss, "1 sss");
            sss = _mm_srai_epi32(sss, weights_precision);
            print_m128i(sss, "2 sss");
            sss = _mm_packs_epi32(sss, zero);
            print_m128i(sss, "3 sss");
            sss = _mm_packus_epi16(sss, zero);
            print_m128i(sss, "4 sss");
            auto o = _mm_cvtsi128_si32(sss);

            std::memcpy(&dst[i * strides[0]], (uint8_t *) &o, 4);
        }

        // Block 1
        for (; i < n; i++) {

            char* src_min = src + i * strides[1] + ids_min;

            uint8_t t = *(uint8_t*)&src_min[0];
            int16_t wts = wts_ptr[0];
            // Intermediate computations are using integer type
            int output = 1 << (weights_precision - 1);  // accounts for the +0.5 part
            output += t * wts;
            for (int j=1; j < ids_size; j++) {
                wts = wts_ptr[j];
                t = *(uint8_t*)&src_min[j * ids_stride];
                output += t * wts;
            }
            *(uint8_t*)&dst[i * strides[0]] = (uint8_t)std::clamp(output >> weights_precision, 0, 255);
        }

        print_data(&output[0 * out_height * out_width], out_height * out_width, "output, c0");
        print_data(&output[1 * out_height * out_width], out_height * out_width, "output, c1");
        print_data(&output[2 * out_height * out_width], out_height * out_width, "output, c2");
        printf("\n");
    }


    // print_data output:
    // [43, 46, 49, 52, 55, 58, 61, 64, 67, 70, 73, 76, 79, 82, 85, 88, 91, 94, 97, 100,
    //    137, 140, 143, 146, 149, 152, 155, 158, 161, 164, 167, 170, 173, 176, 179, 182, 185, 188, 191, 194,
    //  44, 47, 50, 53, 56, 59, 62, 65, 68, 71, 74, 77, 80, 83, 86, 89, 92, 95, 98, 101,
    //    138, 141, 144, 147, 150, 153, 156, 159, 162, 165, 168, 171, 174, 177, 180, 183, 186, 189, 192, 195,
    //  45, 48, 51, 54, 57, 60, 63, 66, 69, 72, 75, 78, 81, 84, 87, 90, 93, 96, 99, 102,
    //    139, 142, 145, 148, 151, 154, 157, 160, 163, 166, 169, 172, 175, 178, 181, 184, 187, 190, 193, 196]
    unsigned char expected[out_height * out_width * num_channels] = {
        43, 46, 49, 52, 55, 58, 61, 64, 67, 70, 73, 76, 79, 82, 85, 88, 91, 94, 97, 100,                     // y=0
        137, 140, 143, 146, 149, 152, 155, 158, 161, 164, 167, 170, 173, 176, 179, 182, 185, 188, 191, 194,  // y=1

        44, 47, 50, 53, 56, 59, 62, 65, 68, 71, 74, 77, 80, 83, 86, 89, 92, 95, 98, 101,                     // y=0
        138, 141, 144, 147, 150, 153, 156, 159, 162, 165, 168, 171, 174, 177, 180, 183, 186, 189, 192, 195,  // y=1

        45, 48, 51, 54, 57, 60, 63, 66, 69, 72, 75, 78, 81, 84, 87, 90, 93, 96, 99, 102,                     // y=0
        139, 142, 145, 148, 151, 154, 157, 160, 163, 166, 169, 172, 175, 178, 181, 184, 187, 190, 193, 196   // y=1
    };
    auto c = 0;
    for (int k = 0; k < num_channels; k++) {
        for (int y = 0; y < out_height; y++) {
            for (int i = 0; i < out_width; i++) {
                assert(output[c] == expected[k * out_width * out_height + y * out_width + i]);
                c++;
            }
        }
    }

}

// // Horizontal pass: ImagingResampleHorizontalConvolution8u4x
// void test_ImagingResampleHorizontalConvolution8u4x()
// {
//     // Horizontal pass: ImagingResampleHorizontalConvolution8u4x
//     // Try to adapt the code for RGB data (not RGBA)

//     unsigned char *lineIn0, *lineIn1, *lineIn2, *lineIn3;
//     unsigned char *lineOut0, *lineOut1, *lineOut2, *lineOut3;

//     unsigned char data[8 * 9 * 3 + 5] = {
//         0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
//         27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
//         54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
//         81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107,

//         108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134,
//         135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161,
//         162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188,
//         189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215,
//         // ---- out of bounds ----
//         255, 255, 255, 255, 255
//     };

//     unsigned char output[8 * 2 * 3] = {
//         0, 0, 0, 0, 0, 0,  // line 0
//         0, 0, 0, 0, 0, 0,  // line 1
//         0, 0, 0, 0, 0, 0,  // line 2
//         0, 0, 0, 0, 0, 0,  // line 3

//         0, 0, 0, 0, 0, 0,  // line 0
//         0, 0, 0, 0, 0, 0,  // line 1
//         0, 0, 0, 0, 0, 0,  // line 2
//         0, 0, 0, 0, 0, 0   // line 3
//     };

//     constexpr int num_channels = 3;
//     auto stride = num_channels * 1;  // num channels * sizeof(uint8_t)

//     // (9, 8)
//     // xx=0, xsize=2, xmin=0, xmax=7, coefs_precision=17
//     // k: 20307 27691 31383 23999 16615 9230 1846
//     // xx=1, xsize=2, xmin=2, xmax=7, coefs_precision=17
//     // k: 1846 9230 16615 23999 31383 27691 20307
//     // xx=0, xsize=2, xmin=0, xmax=7, coefs_precision=17
//     // k: 20307 27691 31383 23999 16615 9230 1846
//     // xx=1, xsize=2, xmin=2, xmax=7, coefs_precision=17
//     // k: 1846 9230 16615 23999 31383 27691 20307
//     // (2, 8)

//     int xsize = 2;
//     int xbounds[2 * 2] = {0, 7, 2, 7};
//     INT16 kk[9 * 2] = {
//         20307, 27691, 31383, 23999, 16615, 9230, 1846, 0, 0,
//         1846, 9230, 16615, 23999, 31383, 27691, 20307, 0, 0
//     };
//     int kmax = 9;
//     int coefs_precision = 17;

//     for (int part=0; part < 2; part++) {

//         lineIn0 = &data[(4 * part + 0) * 9 * num_channels];
//         lineIn1 = &data[(4 * part + 1) * 9 * num_channels];
//         lineIn2 = &data[(4 * part + 2) * 9 * num_channels];
//         lineIn3 = &data[(4 * part + 3) * 9 * num_channels];

//         lineOut0 = &output[(4 * part + 0) * 2 * num_channels];
//         lineOut1 = &output[(4 * part + 1) * 2 * num_channels];
//         lineOut2 = &output[(4 * part + 2) * 2 * num_channels];
//         lineOut3 = &output[(4 * part + 3) * 2 * num_channels];

//         std::cout << "\n----- part = " << part << " -----" << std::endl;

//         print_uint32(((UINT32 *)lineIn0)[0], "lineIn0");
//         print_uint32(((UINT32 *)lineIn1)[0], "lineIn1");
//         print_uint32(((UINT32 *)lineIn2)[0], "lineIn2");
//         print_uint32(((UINT32 *)lineIn3)[0], "lineIn3");

//         int xmin, xmax, xx;
//         INT16 *k;

//         // xsize = output width, xx = output x index
//         // xmax = interpolation size, x = interpolation index (horizontal <-> x dimension)
//         // xmin = input x start index corresponding to output x index (xx)

//         {
//             std::cout << "Weights as int16: " << std::endl;
//             for (int i=0; i < kmax; i++) {
//                 std::cout << (int) kk[i] << " ";
//             }
//             std::cout << std::endl;
//             for (int i=kmax; i < 2 * kmax; i++) {
//                 std::cout << (int) kk[i] << " ";
//             }
//             std::cout << std::endl;

//             std::cout << "Weights as uint8_t: " << std::endl;
//             unsigned char * k_ui8 = (unsigned char *) kk;
//             for (int i=0; i < 2 * kmax; i++) {
//                 std::cout << (int) k_ui8[i] << " ";
//             }
//             std::cout << std::endl;
//             for (int i=kmax; i < 4 * kmax; i++) {
//                 std::cout << (int) k_ui8[i] << " ";
//             }
//             std::cout << std::endl;
//         }

//         __m256i mask_low, mask_high;
//         if (num_channels == 4) {
//             mask_low = _mm256_set_epi8(
//                 -1,7, -1,3, -1,6, -1,2, -1,5, -1,1, -1,4, -1,0,
//                 -1,7, -1,3, -1,6, -1,2, -1,5, -1,1, -1,4, -1,0
//             );
//             mask_high = _mm256_set_epi8(
//                 -1,15, -1,11, -1,14, -1,10, -1,13, -1,9, -1,12, -1,8,
//                 -1,15, -1,11, -1,14, -1,10, -1,13, -1,9, -1,12, -1,8
//             );
//         } else if (num_channels == 3) {
//             mask_low = _mm256_set_epi8(
//                 -1,-1, -1,-1, -1,5, -1,2, -1,4, -1,1, -1,3, -1,0,
//                 -1,-1, -1,-1, -1,5, -1,2, -1,4, -1,1, -1,3, -1,0
//             );
//             mask_high = _mm256_set_epi8(
//                 -1,-1, -1,-1, -1,11, -1,8, -1,10, -1,7, -1,9, -1,6,
//                 -1,-1, -1,-1, -1,11, -1,8, -1,10, -1,7, -1,9, -1,6
//             );
//         }

//         for (xx = 0; xx < xsize; xx++) {
//             xmin = xbounds[xx * 2 + 0];
//             xmax = xbounds[xx * 2 + 1];
//             k = &kk[xx * kmax];

//             int64_t x = 0;

//             __m256i sss0, sss1;
//             __m256i zero = _mm256_setzero_si256();
//             __m256i initial = _mm256_set1_epi32(1 << (coefs_precision-1));
//             sss0 = initial;
//             sss1 = initial;

//             std::cout << "-- xx=" << xx << std::endl;
//             print_m256i(sss0, "sss0");
//             // print_m256i sss0: 0 128 0 0 0 128 0 0 0 128 0 0 0 128 0 0 0 128 0 0 0 128 0 0 0 128 0 0 0 128 0 0
//             print_m256i(sss1, "sss1");
//             // print_m256i sss1: 0 128 0 0 0 128 0 0 0 128 0 0 0 128 0 0 0 128 0 0 0 128 0 0 0 128 0 0 0 128 0 0

//             // lineIn0 + stride * (x + xmin) + 16 <= lineIn0 + stride * (xmax + xmin)
//             // --> x <= xmax - 16.0 / stride
//             auto b4_xmax = xmax - 16.0 / stride + 1;

//             for (; x < b4_xmax; x += 4) {
//                 std::cout << "- block 4, x: " << x << std::endl;
//                 __m256i pix, mmk0, mmk1, source;

//                 // Load 2 weight values
//                 mmk0 = _mm256_set1_epi32(*(INT32 *) &k[x]);
//                 print_m256i(mmk0, "mmk0");
//                 // print_m256i mmk0: 183 45 0 64 183 45 0 64 183 45 0 64 183 45 0 64 183 45 0 64 183 45 0 64 183 45 0 64 183 45 0 64

//                 mmk1 = _mm256_set1_epi32(*(INT32 *) &k[x + 2]);
//                 print_m256i(mmk1, "mmk1");
//                 // print_m256i mmk1: 0 64 183 45 0 64 183 45 0 64 183 45 0 64 183 45 0 64 183 45 0 64 183 45 0 64 183 45 0 64 183 45

//                 // Load two lines into lanes
//                 source = _mm256_inserti128_si256(_mm256_castsi128_si256(
//                     _mm_loadu_si128((__m128i *) (lineIn0 + stride * (x + xmin)))),
//                     _mm_loadu_si128((__m128i *) (lineIn1 + stride * (x + xmin))), 1);
//                 print_m256i(source, "source");
//                 // print_m256i source: 0 1 2, 3 4 5, 6 7 8, 9 10 11, 12 13 14, 15 |  27 28 29, 30 31 32, 33 34 35, 36 37 38, 39 40 41, 42

//                 // Shuffle source to get pairs like (r0, r1); (g0, g1); (b0, b1)
//                 pix = _mm256_shuffle_epi8(source, mask_low);
//                 print_m256i(pix, "pix");
//                 // print_m256i pix: (0 0) (3 0) (1 0) (4 0) (2 0) (5 0) (0 0) (0 0) |  (27 0) (30 0) (28 0) (31 0) (29 0) (32 0) (0 0) (0 0)
//                 sss0 = _mm256_add_epi32(sss0, _mm256_madd_epi16(pix, mmk0));
//                 print_m256i(sss0, "sss0");
//                 // print_m256i sss0: 0 64 1 0 183 173 1 0 110 27 2 0 0 128 0 0 77 210 12 0 4 64 13 0 187 173 13 0 0 128 0 0

//                 pix = _mm256_shuffle_epi8(source, mask_high);
//                 print_m256i(pix, "pix");
//                 // print_m256i pix: (6 0) (9 0) (7 0) (10 0) (8 0) (11 0) (0 0) (0 0) |  (33 0) (36 0) (34 0) (37 0) (35 0) (38 0) (0 0) (0 0)
//                 sss0 = _mm256_add_epi32(sss0, _mm256_madd_epi16(pix, mmk1));
//                 print_m256i(sss0, "sss0");
//                 // print_m256i sss0: 111 91 4 0 221 54 5 0 75 18 6 0 0 128 0 0 9 128 27 0 119 91 28 0 229 54 29 0 0 128 0 0

//                 source = _mm256_inserti128_si256(_mm256_castsi128_si256(
//                     _mm_loadu_si128((__m128i *) (lineIn2 + stride * (x + xmin)))),
//                     _mm_loadu_si128((__m128i *) (lineIn3 + stride * (x + xmin))), 1);
//                 print_m256i(source, "source");
//                 // print_m256i source: 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96
//                 pix = _mm256_shuffle_epi8(source, mask_low);
//                 print_m256i(pix, "pix");
//                 // print_m256i pix: 54 0 57 0 55 0 58 0 56 0 59 0 0 0 0 0 81 0 84 0 82 0 85 0 83 0 86 0 0 0 0 0
//                 sss1 = _mm256_add_epi32(sss1, _mm256_madd_epi16(pix, mmk0));
//                 print_m256i(sss1, "sss1");
//                 // print_m256i sss1: 154 100 24 0 81 210 24 0 8 64 25 0 0 128 0 0 231 246 35 0 158 100 36 0 85 210 36 0 0 128 0 0
//                 pix = _mm256_shuffle_epi8(source, mask_high);
//                 print_m256i(pix, "pix");
//                 // print_m256i pix: 60 0 63 0 61 0 64 0 62 0 65 0 0 0 0 0 87 0 90 0 88 0 91 0 89 0 92 0 0 0 0 0

//                 sss1 = _mm256_add_epi32(sss1, _mm256_madd_epi16(pix, mmk1));
//                 print_m256i(sss1, "sss1");
//                 // print_m256i sss1: 163 164 50 0 17 128 51 0 127 91 52 0 0 128 0 0 61 201 73 0 171 164 74 0 25 128 75 0 0 128 0 0
//             }

//             // lineIn0 + stride * (x + xmin) + 16 <= lineIn0 + stride * (xmax + xmin)
//             // --> x <= xmax - 16.0 / stride
//             auto b2_xmax = xmax - 16.0 / stride + 1;

//             for (; x < b2_xmax; x += 2) {
//                 std::cout << "- block 2, x: " << x << std::endl;
//                 __m256i pix, mmk;

//                 // - Load int16 weights taking 2 values
//                 mmk = _mm256_set1_epi32(*(INT32 *) &k[x]);
//                 print_m256i(mmk, "mmk");
//                 // print_m256i mmk: 183 45 0 64 183 45 0 64 183 45 0 64 183 45 0 64 183 45 0 64 183 45 0 64 183 45 0 64 183 45 0 64

//                 auto a0 = _mm_loadl_epi64((__m128i *) (lineIn0 + stride * (x + xmin)));
//                 print_m128i(a0, "a0");
//                 // print_m128i a0: 0 1 2 3 4 5 6 7 0 0 0 0 0 0 0 0
//                 auto b0 = _mm256_castsi128_si256(a0);
//                 print_m256i(b0, "b0");
//                 // print_m256i b0: 0 1 2 3 4 5 6 7 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
//                 auto a1 = _mm_loadl_epi64((__m128i *) (lineIn1 + stride * (x + xmin)));
//                 print_m128i(a1, "a1");
//                 // print_m128i a1: 27 28 29 30 31 32 33 34 0 0 0 0 0 0 0 0
//                 pix = _mm256_inserti128_si256(b0, a1, 1);
//                 print_m256i(pix, "1 pix");
//                 // print_m256i 1 pix: 0 1 2 3 4 5 6 7 0 0 0 0 0 0 0 0 27 28 29 30 31 32 33 34 0 0 0 0 0 0 0 0
//                 auto m0 = mask_low;
//                 print_m256i(m0, "m0");
//                 // print_m256i m0: 0 255 3 255 1 255 4 255 2 255 5 255 255 255 255 255 0 255 3 255 1 255 4 255 2 255 5 255 255 255 255 25
//                 pix = _mm256_shuffle_epi8(pix, m0);
//                 print_m256i(pix, "2 pix");
//                 // print_m256i 2 pix: 0 0 3 0 1 0 4 0 2 0 5 0 0 0 0 0 27 0 30 0 28 0 31 0 29 0 32 0 0 0 0 0
//                 // - Interleave pixels from the same line and expand to uint16 (reusing zeros)
//                 // (
//                 //   line0_pixel0_R, 0, line0_pixel1_R, 0, line0_pixel0_G, 0, line0_pixel1_G, 0, line0_pixel0_B, 0, line0_pixel1_B, 0, 255, 0, 255, 0,
//                 //   line1_pixel0_R, 0, line1_pixel1_R, 0, line1_pixel0_G, 0, line1_pixel1_G, 0, line1_pixel0_B, 0, line1_pixel1_B, 0, 255, 0, 255, 0,
//                 // )
//                 auto tmp0 = _mm256_madd_epi16(pix, mmk);
//                 print_m256i(tmp0, "tmp0");
//                 // print_m256i tmp0: 111 27 3 0 38 137 3 0 221 246 3 0 0 0 0 0 188 173 14 0 115 27 15 0 42 137 15 0 0 0 0 0
//                 // tmp0 = pix * mmk
//                 // tmp0 = (t1, t2, t3, t4, t5, t6, ..., t31, t32)
//                 //
//                 // (t1, t2, t3, t4) = (line0_pixel0_R, 0) * (mmk[0], mmk[1]) + (line0_pixel1_R, 0) * (mmk[2], mmk[3])
//                 //                  = (0, 0) * (183 45) + (3 0) * (0 64) = 0 * 11703 + 3 * 64 * 256 = 192 * 256 <=> (0, 192, 0, 0)
//                 // (t5, t6, t7, t8) = (line0_pixel0_G, 0) * (mmk[4], mmk[5]) + (line0_pixel1_G, 0) * (mmk[6], mmk[7])
//                 //                  = (1, 0) * (183 45) + (4 0) * (0 64) = 1 * 11703 + 4 * 64 * 256 = 77239 => (183 45 1 0)
//                 // ...
//                 sss0 = _mm256_add_epi32(sss0, tmp0);
//                 // sss0 = initial + pix * mmk + ...
//                 print_m256i(sss0, "sss0");
//                 // print_m256i sss0: 111 91 4 0 221 54 5 0 75 18 6 0 0 128 0 0 9 128 27 0 119 91 28 0 229 54 29 0 0 128 0 0

//                 pix = _mm256_inserti128_si256(_mm256_castsi128_si256(
//                     _mm_loadl_epi64((__m128i *) (lineIn2 + stride * (x + xmin)))),
//                     _mm_loadl_epi64((__m128i *) (lineIn3 + stride * (x + xmin))), 1);
//                 pix = _mm256_shuffle_epi8(pix, mask_low);
//                 sss1 = _mm256_add_epi32(sss1, _mm256_madd_epi16(pix, mmk));
//                 print_m256i(sss1, "sss1");
//                 // print_m256i sss1: 163 164 50 0 17 128 51 0 127 91 52 0 0 128 0 0 61 201 73 0 171 164 74 0 25 128 75 0 0 128 0 0
//             }

//             for (; x < xmax - 1; x++) {
//                 std::cout << "- block 1, x: " << x << std::endl;
//                 __m256i pix, mmk;

//                 mmk = _mm256_set1_epi32(k[x]);
//                 // 123 34 0 0
//                 print_m256i(mmk, "mmk");
//                 // print_m256i mmk: 123 34 0 0 123 34 0 0 123 34 0 0 123 34 0 0 123 34 0 0 123 34 0 0 123 34 0 0

//                 auto a0 = mm_cvtepu8_epi32(lineIn0 + stride * (x + xmin));
//                 print_m128i(a0, "a0");
//                 // print_m128i a0: 0 0 0 0 1 0 0 0 2 0 0 0 3 0 0 0
//                 auto b0 = _mm256_castsi128_si256(a0);
//                 print_m256i(b0, "b0");
//                 // print_m256i b0: 0 0 0 0 1 0 0 0 2 0 0 0 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
//                 auto a1 = mm_cvtepu8_epi32(lineIn1 + stride * (x + xmin));
//                 print_m128i(a1, "a1");
//                 // print_m128i a1: 18 0 0 0 19 0 0 0 20 0 0 0 3 0 0 0
//                 pix = _mm256_inserti128_si256(b0, a1, 1);
//                 print_m256i(pix, "pix");
//                 // print_m256i pix: 0 0 0 0 1 0 0 0 2 0 0 0 3 0 0 0 18 0 0 0 19 0 0 0 20 0 0 0 21 0 0 0
//                 auto tmp0 = _mm256_madd_epi16(pix, mmk);
//                 print_m256i(tmp0, "tmp0");
//                 sss0 = _mm256_add_epi32(sss0, tmp0);
//                 print_m256i(sss0, "sss0");
//                 // sss0 = initial + pix[0] * mmk[0] + pix[1] * mmk[1] + ...
//                 // for lines 0 and 1


//                 pix = _mm256_inserti128_si256(
//                     _mm256_castsi128_si256(
//                         mm_cvtepu8_epi32(
//                             lineIn2 + stride * (x + xmin)
//                         )
//                     ),
//                     mm_cvtepu8_epi32(
//                         lineIn3 + stride * (x + xmin)
//                     ),
//                     1
//                 );
//                 print_m256i(pix, "pix");
//                 auto tmp1 = _mm256_madd_epi16(pix, mmk);
//                 print_m256i(tmp1, "tmp1");
//                 sss1 = _mm256_add_epi32(sss1, tmp1);
//                 print_m256i(sss1, "sss1");
//                 // sss1 = initial + pix[0] * mmk[0] + pix[1] * mmk[1] + ...
//                 // for lines 2 and 3
//             }

//             // last element x = xmax - 1
//             {
//                 x = xmax - 1;
//                 std::cout << "- block 0, x: " << x << std::endl;
//                 int output0[8] = {0, 0, 0, 0, 0, 0, 0, 0};
//                 int output1[8] = {0, 0, 0, 0, 0, 0, 0, 0};

//                 auto p0 = lineIn0 + stride * (x + xmin);
//                 auto p1 = lineIn1 + stride * (x + xmin);
//                 auto p2 = lineIn2 + stride * (x + xmin);
//                 auto p3 = lineIn3 + stride * (x + xmin);
//                 for (int ch=0; ch < num_channels; ch++) {
//                     output0[0 + ch] += (*(p0 + ch) * k[x]);
//                     output0[4 + ch] += (*(p1 + ch) * k[x]);
//                     output1[0 + ch] += (*(p2 + ch) * k[x]);
//                     output1[4 + ch] += (*(p3 + ch) * k[x]);
//                 }

//                 auto t0 = _mm256_load_si256((__m256i *)output0);
//                 auto t1 = _mm256_load_si256((__m256i *)output1);
//                 print_m256i(t0, "t0");
//                 print_m256i(t1, "t1");

//                 sss0 = _mm256_add_epi32(sss0, _mm256_load_si256((__m256i *)output0));
//                 sss1 = _mm256_add_epi32(sss1, _mm256_load_si256((__m256i *)output1));
//             }

//             std::cout << "--" << std::endl;

//             // Shift packed 32-bit integers in a right by imm8 while shifting in sign bits, and store the results in dst.
//             print_m256i(sss0, "sss0 ->");
//             //
//             sss0 = _mm256_srai_epi32(sss0, coefs_precision);
//             print_m256i(sss0, "-> sss0");
//             // print_m256i -> sss0: 15 0 0 0 16 0 0 0 17 0 0 0 18 0 0 0 39 0 0 0 40 0 0 0 41 0 0 0 42 0 0 0
//             print_m256i(sss1, "sss1 ->");
//             sss1 = _mm256_srai_epi32(sss1, coefs_precision);
//             print_m256i(sss1, "-> sss1");
//             // print_m256i -> sss1: 63 0 0 0 64 0 0 0 65 0 0 0 66 0 0 0 87 0 0 0 88 0 0 0 89 0 0 0 90 0 0 0

//             // Convert packed signed 32-bit integers from a and b to packed 16-bit integers using signed saturation,
//             // and store the results in dst.
//             // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#ig_expand=5381,5381,6389,2316,2310,6688,5148,5178,5148&text=_mm256_packs_epi32
//             sss0 = _mm256_packs_epi32(sss0, zero);
//             print_m256i(sss0, "sss0");
//             // print_m256i sss0: 15 0 16 0 17 0 X 0 0 0 0 0 0 0 0 0 39 0 40 0 41 0 X 0 0 0 0 0 0 0 0 0

//             sss1 = _mm256_packs_epi32(sss1, zero);
//             print_m256i(sss1, "sss1");
//             // print_m256i sss1: 63 0 64 0 65 0 X 0 0 0 0 0 0 0 0 0 87 0 88 0 89 0 X 0 0 0 0 0 0 0 0 0

//             // Convert packed signed 16-bit integers from a and b to packed 8-bit integers using unsigned saturation,
//             // and store the results in dst.
//             // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#ig_expand=5381,5381,6389,2316,2310,6688,5148,5178&text=_mm256_packus_epi16
//             sss0 = _mm256_packus_epi16(sss0, zero);
//             print_m256i(sss0, "sss0");
//             // print_m256i sss0: 15 16 17 X 0 0 0 0 0 0 0 0 0 0 0 0 39 40 41 X 0 0 0 0 0 0 0 0 0 0 0 0

//             sss1 = _mm256_packus_epi16(sss1, zero);
//             print_m256i(sss1, "sss1");
//             // print_m256i sss1: 63 64 65 X 0 0 0 0 0 0 0 0 0 0 0 0 87 88 89 X 0 0 0 0 0 0 0 0 0 0 0 0

//             if (num_channels == 3) {
//                 // replace X by 0
//                 auto mask = _mm256_set_epi8(
//                     -1, -1, -1, -1,  -1, -1, -1, -1,  -1, -1, -1, -1,  -1, 2, 1, 0,
//                     -1, -1, -1, -1,  -1, -1, -1, -1,  -1, -1, -1, -1,  -1, 2, 1, 0
//                 );
//                 sss0 = _mm256_shuffle_epi8(sss0, mask);
//                 sss1 = _mm256_shuffle_epi8(sss1, mask);
//             }
//             // print_m256i sss0: 15 16 17 0 0 0 0 0 0 0 0 0 0 0 0 0 39 40 41 0 0 0 0 0 0 0 0 0 0 0 0 0
//             // print_m256i sss1: 63 64 65 0 0 0 0 0 0 0 0 0 0 0 0 0 87 88 89 0 0 0 0 0 0 0 0 0 0 0 0 0

//             // _mm_cvtsi128_si32: Copy the lower 32-bit integer in a to dst.
//             // _mm256_extracti128_si256: Extract 128 bits (composed of integer data) from a, selected with imm8, and store the result in dst.
//             auto e0 = _mm256_extracti128_si256(sss0, 0);
//             print_m128i(e0, "e0");
//             // print_m128i e0: 15 16 17 0 0 0 0 0 0 0 0 0 0 0 0 0

//             auto e1 = _mm256_extracti128_si256(sss0, 1);
//             print_m128i(e1, "e1");
//             // print_m128i e1: 39 40 41 0 0 0 0 0 0 0 0 0 0 0 0 0

//             auto e2 = _mm256_extracti128_si256(sss1, 0);
//             print_m128i(e2, "e2");
//             // print_m128i e2: 63 64 65 0 0 0 0 0 0 0 0 0 0 0 0 0

//             auto e3 = _mm256_extracti128_si256(sss1, 1);
//             print_m128i(e3, "e3");
//             // print_m128i e3: 87 88 89 0 0 0 0 0 0 0 0 0 0 0 0 0

//             auto o0 = _mm_cvtsi128_si32(e0);
//             auto o1 = _mm_cvtsi128_si32(e1);
//             auto o2 = _mm_cvtsi128_si32(e2);
//             auto o3 = _mm_cvtsi128_si32(e3);
//             for (int ch=0; ch<num_channels; ch++) {
//                 (lineOut0 + stride * xx)[ch] = ((unsigned char *) &o0)[ch];
//                 (lineOut1 + stride * xx)[ch] = ((unsigned char *) &o1)[ch];
//                 (lineOut2 + stride * xx)[ch] = ((unsigned char *) &o2)[ch];
//                 (lineOut3 + stride * xx)[ch] = ((unsigned char *) &o3)[ch];
//             }
//         }

//         print_data(output + part * (4 * 2 * 3), 4 * 2 * 3, "- output");
//     }

//     print_data(output, 8 * 2 * 3, "output");
//     // print_data output:
//     // Expected: [[[7, 8, 9, 17, 18, 19, 34, 35, 36, 44, 45, 46, 61, 62, 63, 71, 72, 73, 88, 89, 90, 98, 99, 100, 115, 116, 117, 125, 126, 127, 142, 143, 144, 152, 153, 154, 169, 170, 171, 179, 180, 181, 196, 197, 198, 206, 207, 208]]]

//     unsigned char expected[8 * 2 * 3] = {
//         7, 8, 9, 17, 18, 19, 34, 35, 36, 44, 45, 46, 61, 62, 63, 71, 72, 73, 88, 89, 90, 98, 99, 100, 115, 116, 117, 125, 126, 127, 142, 143, 144, 152, 153, 154, 169, 170, 171, 179, 180, 181, 196, 197, 198, 206, 207, 208
//     };
//     for (int i=0; i<(8 * 2 * 3); i++) {
//         assert(output[i] == expected[i]);
//     }
// }

// Horizontal pass: ImagingResampleHorizontalConvolution8u
void test_ImagingResampleHorizontalConvolution8u()
{

    // Horizontal pass: ImagingResampleHorizontalConvolution8u
    // Try to adapt the code for RGB Channels First data

    constexpr int num_channels = 3;
    constexpr int width = 18;
    constexpr int height = 2;
    constexpr int out_height = 2;
    constexpr int out_width = 2;

    unsigned char input[height * width * num_channels] = {
        0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51,
        54, 57, 60, 63, 66, 69, 72, 75, 78, 81, 84, 87, 90, 93, 96, 99, 102, 105,

        1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 52,
        55, 58, 61, 64, 67, 70, 73, 76, 79, 82, 85, 88, 91, 94, 97, 100, 103, 106,

        2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50, 53,
        56, 59, 62, 65, 68, 71, 74, 77, 80, 83, 86, 89, 92, 95, 98, 101, 104, 107
    };

    unsigned char output[out_height * out_width * num_channels] = {};

    print_data(&input[0], height * width, "input, c0");
    print_data(&input[height * width], height * width, "input, c1");
    print_data(&input[2 * height * width], height * width, "input, c2");
    printf("\n");

    // (18, 2)
    // xx=0, xsize=2, xmin=0, xmax=14, coefs_precision=17
    // k: 9230 11077 12923 14769 16615 14769 12923 11077 9230 7384 5538 3692 1846 0
    // xx=1, xsize=2, xmin=5, xmax=13, coefs_precision=17
    // k: 1846 3692 5538 7384 9230 11077 12923 14769 16615 14769 12923 11077 9230
    // xx=0, xsize=2, xmin=0, xmax=14, coefs_precision=17
    // k: 9230 11077 12923 14769 16615 14769 12923 11077 9230 7384 5538 3692 1846 0
    // xx=1, xsize=2, xmin=5, xmax=13, coefs_precision=17
    // k: 1846 3692 5538 7384 9230 11077 12923 14769 16615 14769 12923 11077 9230
    // (2, 2)

    int xsize = 2;
    int xbounds[2 * 2] = {0, 14, 5, 13};
    INT16 kk[14 * 2] = {
        9230, 11077, 12923, 14769, 16615, 14769, 12923, 11077, 9230, 7384, 5538, 3692, 1846, 0,
        1846, 3692, 5538, 7384, 9230, 11077, 12923, 14769, 16615, 14769, 12923, 11077, 9230, 0
    };
    int kmax = 14;
    int ksize = kmax;
    int weights_precision = 17;

    for (int tt = 0; tt < num_channels * out_height; tt++) {

      std::cout << "--- tt= " << tt << std::endl;

      auto y = tt % out_height;
      std::cout << "    y= " << y << std::endl;
      auto c = tt / out_height;
      std::cout << "    c= " << c << std::endl;

      char * src = (char *) input + tt * width;
      char * dst = (char *) output + tt * out_width;

      // int xmin, xmax, x;
      // index stride is constant for the given dimension
      const int64_t ids_stride = 1;

    //   const int16_t *wts_ptr = &kk[y * ksize];
      const int64_t strides[2] = {sizeof(uint8_t), 0};
      const int64_t n = out_width;

      int64_t i = 0;


      // Block 4
      for (; i < n - 3; i += 4) {
        int64_t ids_min = xbounds[i * xsize + 0];
        int64_t ids_size = xbounds[i * xsize + 1];

        auto sss = initial;

        int64_t j = 0;
        char* src_min = src + i * strides[1] + ids_min;

        // for (; j < ids_size - 1; j += 2) {
        //   // wts = [
        //   //    w0_l w0_h w1_l w1_h  w0_l w0_h w1_l w1_h
        //   //    w0_l w0_h w1_l w1_h  w0_l w0_h w1_l w1_h
        //   // ]
        //   auto wts = _mm_set1_epi32(*(int32_t*)&wts_ptr[j]);
        //   std::cout << "b4 -- j=" << j << std::endl;
        //   print_m128i(wts, "wts");

        //   // Read 8 bytes, 4 bytes per line:
        //   // source1 = [r0 r1 r2 r3  0 0 0 0  0 0 0 0  0 0 0 0]
        //   // source2 = [R0 R1 R2 R3  0 0 0 0  0 0 0 0  0 0 0 0]
        //   // source = [r0 R0 r1 R1  r2 R2 r3 R3  0 0 0 0  0 0 0 0]
        //   // pix = [r0 0 R0 0  r1 0 R1 0  r2 0 R2 0  r3 0 R3 0]
        //   auto source1 = mm_cvt_si128((const uint8_t *) &src_min[(j + 0) * ids_stride], 4);
        //   auto source2 = mm_cvt_si128((const uint8_t *) &src_min[(j + 1) * ids_stride], 4);
        //   auto source = _mm_unpacklo_epi8(source1, source2);
        //   auto pix = _mm_unpacklo_epi8(source, zero);
        //   print_m128i(pix, "pix");
        //   // sss = [
        //   //    (r0 * w0) + (R0 * w1)   as int16
        //   //    (r1 * w0) + (R1 * w1)   as int16
        //   //    (r2 * w0) + (R2 * w1)   as int16
        //   //    (r3 * w0) + (R3 * w1)   as int16
        //   // ]
        //   sss = _mm_add_epi32(sss, _mm_madd_epi16(pix, wts));
        // }

          for (; j < ids_size; j++) {
            // wts = [w0_l w0_h 0 0  w0_l w0_h 0 0  ...]
            auto wts = _mm_set1_epi32(wts_ptr[j]);

            std::cout << "b4 - j=" << j << std::endl;
            print_m128i(wts, "wts");
            // Read 4 bytes:
            // source = [r00 r10 r20 r30  0 0 0 0  0 0 0 0  0 0 0 0]
            // pix    = [r00 0 0 0    r10 0 0 0 r20 0 0 0 r30 0 0 0]
            auto source = mm_cvt_multi_si128(
                (const uint8_t *) &src_min[j * ids_stride + 0 * width],
                (const uint8_t *) &src_min[j * ids_stride + 1 * width],
                (const uint8_t *) &src_min[j * ids_stride + 2 * width,]
                (const uint8_t *) &src_min[j * ids_stride + 2 * width]);
            print_m128i(source, "source");
            auto pix = _mm_shuffle_epi8(source, mask_src_b4_1);
            print_m128i(pix, "pix");

            // NOT SURE ABOUT THIS COMMENT
            // sss += [(r0 * w0) 0 0  (r1 * w0) 0 0  (r2 * w0) 0 0  (r3 * w0) 0 0]
            sss = _mm_add_epi32(sss, _mm_madd_epi16(pix, wts));
            print_m128i(sss, "sss");
          }
          print_m128i(sss, "1 sss");
          sss = _mm_srai_epi32(sss, weights_precision);
          print_m128i(sss, "2 sss");
          sss = _mm_packs_epi32(sss, zero);
          print_m128i(sss, "3 sss");
          sss = _mm_packus_epi16(sss, zero);
          print_m128i(sss, "4 sss");
          auto o = _mm_cvtsi128_si32(sss);

          std::memcpy(&dst[i * strides[0]], (uint8_t *) &o, 4);
        }

      // Block 1
      for (; i < n; i++) {
        int64_t ids_min = xbounds[i * xsize + 0];
        int64_t ids_size = xbounds[i * xsize + 1];

        char* src_min = src + i * strides[1] + ids_min;

        const int16_t *wts_ptr = &kk[i * ksize];

        uint8_t t = *(uint8_t*)&src_min[0];
        int16_t wts = wts_ptr[0];
        // Intermediate computations are using integer type
        int output = 1 << (weights_precision - 1);  // accounts for the +0.5 part
        output += t * wts;
        for (int j=1; j < ids_size; j++) {
          wts = wts_ptr[j];
          t = *(uint8_t*)&src_min[j * ids_stride];
          output += t * wts;
        }
        *(uint8_t*)&dst[i * strides[0]] = (uint8_t)std::clamp(output >> weights_precision, 0, 255);
      }

      print_data(&output[0 * out_height * out_width], out_height * out_width, "output, c0");
      print_data(&output[1 * out_height * out_width], out_height * out_width, "output, c1");
      print_data(&output[2 * out_height * out_width], out_height * out_width, "output, c2");
      printf("\n");

    }

    // Expected:
    // [15, 36, 69, 90]
    // [16, 37, 70, 91]
    // [17, 38, 71, 92]
    unsigned char expected[out_height * out_width * num_channels] = {
        15, 36,
        69, 90,

        16, 37,
        70, 91,

        17, 38,
        71, 92
    };
    auto c = 0;
    for (int k = 0; k < num_channels; k++) {
        for (int y = 0; y < out_height; y++) {
            for (int i = 0; i < out_width; i++) {
                assert(output[c] == expected[k * out_width * out_height + y * out_width + i]);
                c++;
            }
        }
    }

}

int main() {

    // test_ImagingResampleVerticalConvolution8u();

    test_ImagingResampleHorizontalConvolution8u();

    return 0;
}
