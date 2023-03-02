// STD
#include <iostream>
#include <cstring>
#include <cassert>

// AVX
#include <immintrin.h>


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


static __m128i inline mm_cvtepu8_epi32(void *ptr) {
    return _mm_cvtepu8_epi32(_mm_cvtsi32_si128(*(INT32 *) ptr));
}


static __m256i inline mm256_cvtepu8_epi32(void *ptr) {
    return _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i *) ptr));
}


void test_1()
{

    unsigned char data[8 * 9 * 3] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
        27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
        54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
        81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107,
        108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134,
        135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161,
        162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188,
        189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215
    };

    UINT32 *lineIn0, *lineIn1, *lineIn2, *lineIn3;

    lineIn0 = (UINT32 *) &data[0 * 9 * 3];
    lineIn1 = (UINT32 *) &data[1 * 9 * 3];
    lineIn2 = (UINT32 *) &data[2 * 9 * 3];
    lineIn3 = (UINT32 *) &data[3 * 9 * 3];

    int coefs_precision = 16;

    int xx = 0, x= 0, xmin=0;

    __m256i sss0, sss1;
    __m256i zero = _mm256_setzero_si256();
    __m256i initial = _mm256_set1_epi32(1 << (coefs_precision-1));
    sss0 = initial;
    sss1 = initial;

    std::cout << "-- xx=" << xx << std::endl;
    print_m256i(sss0, "sss0");
    // print_m256i sss0: 0 128 0 0 0 128 0 0 0 128 0 0 0 128 0 0 0 128 0 0 0 128 0 0 0 128 0 0 0 128 0 0
    print_m256i(sss1, "sss1");
    // print_m256i sss1: 0 128 0 0 0 128 0 0 0 128 0 0 0 128 0 0 0 128 0 0 0 128 0 0 0 128 0 0 0 128 0 0

    std::cout << "- block 4, x: " << x << std::endl;
    __m256i pix, source;
    {
        // Load two lines into lanes
        source = _mm256_inserti128_si256(_mm256_castsi128_si256(
            _mm_loadu_si128((__m128i *) &lineIn0[x + xmin])),
            _mm_loadu_si128((__m128i *) &lineIn1[x + xmin]), 1);
        print_m256i(source, "source");
        // print_m256i source: 0 1 2, 3 4 5, 6 7 8, 9 10 11, 12 13 14, 15 |  27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42

        // Shuffle source to get pairs like (r0, r1); (g0, g1); (b0, b1)
        pix = _mm256_shuffle_epi8(source, _mm256_set_epi8(
            -1,-1, -1,-1, -1,5, -1,2, -1,4, -1,1, -1,3, -1,0,
            -1,-1, -1,-1, -1,5, -1,2, -1,4, -1,1, -1,3, -1,0
        ));
        print_m256i(pix, "pix");
        // print_m256i pix: (0 0) (3 0) (1 0) (4 0) (2 0) (5 0) (0 0) (0 0) |  (27 0) (30 0) (28 0) (31 0) (29 0) (32 0) (0 0) (0 0)
    }

    std::cout << "- block 2, x: " << x << std::endl;
    {
        auto a0 = _mm_loadl_epi64((__m128i *) &lineIn0[x + xmin]);
        print_m128i(a0, "a0");
        // print_m128i a0: 0 1 2 3 4 5 6 7 0 0 0 0 0 0 0 0
        auto b0 = _mm256_castsi128_si256(a0);
        print_m256i(b0, "b0");
        // print_m256i b0: 0 1 2 3 4 5 6 7 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        auto a1 = _mm_loadl_epi64((__m128i *) &lineIn1[x + xmin]);
        print_m128i(a1, "a1");
        // print_m128i a1: 27 28 29 30 31 32 33 34 0 0 0 0 0 0 0 0
        pix = _mm256_inserti128_si256(b0, a1, 1);
        print_m256i(pix, "1 pix");
        // print_m256i 1 pix: 0 1 2 3 4 5 6 7 0 0 0 0 0 0 0 0 27 28 29 30 31 32 33 34 0 0 0 0 0 0 0 0
        auto m0 = _mm256_set_epi8(
            -1,-1, -1,-1, -1,5, -1,2, -1,4, -1,1, -1,3, -1,0,
            -1,-1, -1,-1, -1,5, -1,2, -1,4, -1,1, -1,3, -1,0);
        print_m256i(m0, "m0");
        // print_m256i m0: 0 255 3 255 1 255 4 255 2 255 5 255 255 255 255 255 0 255 3 255 1 255 4 255 2 255 5 255 255 255 255 255
        pix = _mm256_shuffle_epi8(pix, m0);
        print_m256i(pix, "2 pix");
        // print_m256i 2 pix: 0 0 3 0 1 0 4 0 2 0 5 0 0 0 0 0 27 0 30 0 28 0 31 0 29 0 32 0 0 0 0 0
    }


    std::cout << "- block 1, x: " << x << std::endl;
    {
        auto a0 = mm_cvtepu8_epi32(&lineIn0[x + xmin]);
        print_m128i(a0, "a0");
        // print_m128i a0: 0 0 0 0 1 0 0 0 2 0 0 0 3 0 0 0
        auto b0 = _mm256_castsi128_si256(a0);
        print_m256i(b0, "b0");
        // print_m256i b0: 0 0 0 0 1 0 0 0 2 0 0 0 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        auto a1 = mm_cvtepu8_epi32(&lineIn1[x + xmin]);
        print_m128i(a1, "a1");
        // print_m128i a1: 18 0 0 0 19 0 0 0 20 0 0 0 3 0 0 0
        pix = _mm256_inserti128_si256(b0, a1, 1);
        print_m256i(pix, "pix");
    }
}


void test_2() {


    auto d = _mm256_set_epi8(
        0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,
        0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  4, 3, 2, 1
    );

    auto mask = _mm256_set_epi8(
        -1, -1, -1, -1,  -1, -1, -1, -1,  -1, -1, -1, -1,  -1, 2, 1, 0,
        -1, -1, -1, -1,  -1, -1, -1, -1,  -1, -1, -1, -1,  -1, 2, 1, 0
    );

    d = _mm256_shuffle_epi8(d, mask);

    print_m256i(d, "d");

}


void unpack_rgb_channels_last(uint8_t* unpacked, const uint8_t* packed, int64_t num_channels, int64_t num_pixels) {

  constexpr int rgba_size = 4;
  constexpr int stride = 32 / rgba_size;

  auto i = 0;

//   auto mask = _mm256_set_epi8();
  // R = 0, 1, 2; 3,  4, 5; 6, 7,  8; 9, 10, 11;  12, 13, 14; 15 | 12, 13, 14; 15,  16, 17; 18, 19,  20; 21, 22, 23;  24, 25, 26, 27;
  // -> apply mask on data as 3 channels
  // R = 0, 1, 2, X;  3, 4, 5, X;  6, 7, 8, X;    9, 10, 11, X;  | 12, 13, 14, X;   15, 16, 17, X;   18, 19, 20, X;   21, 22, 23, X

  for (;i < num_pixels - stride; i += stride) {
      auto input_vec = _mm256_loadu_si256((__m256i *) (packed + num_channels * i));


      _mm256_storeu_si256((__m256i *)(unpacked + rgba_size * i), input_vec);
  }

  for (;i < num_pixels; i += 1) {
    for (int j=0; j<4; j++) {
      unpacked[rgba_size * i + j] = (j < num_channels) ? packed[num_channels * i + j] : 0;
    }
  }


}


void test_3() {
    unsigned char input[8 * 9 * 3] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
        27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
        54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
        81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107,
        108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134,
        135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161,
        162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188,
        189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215
    };
    unsigned char output[8 * 9 * 4] = {};

    print_data(input, 8 * 9 * 3, "input");
    unpack_rgb_channels_last(output, input, 3, 8 * 9);
    print_data(output, 8 * 9 * 4, "output");
}

void test_mask_load() {

    unsigned char input[4 * 2] = {
        0, 1, 2, 3, 4, 5, 6, 7
    };

    auto d1 = _mm_loadl_epi64((__m128i *) input);
    print_m128i(d1, "d1");

    auto d2 = _mm_loadl_epi64((__m128i *) input);
    auto mask = _mm_set_epi8(
        -1, 11, 10, 9,  -1, 8, 7, 6,  -1, 5, 4, 3,  -1, 2, 1, 0
    );
    d2 = _mm_shuffle_epi8(d2, mask);
    print_m128i(d2, "d2");
}

// static __m128i inline mm_cvtepu8_epi32(void *ptr) {
//     return _mm_cvtepu8_epi32(_mm_cvtsi32_si128(*(INT32 *) ptr));
// }

void test_mm_cvtepu8_epi32() {
    unsigned char input[4 * 2] = {
        0, 1, 2, 3, 4, 5, 6, 7
    };

    unsigned char data[4] = {
        input[0], input[1], input[2], 0
    };

    auto d1 = _mm_cvtsi32_si128(*(int *) data);
    print_m128i(d1, "d1");

    auto d2 = _mm_cvtepu8_epi32(d1);
    print_m128i(d2, "d2");

    int data2[4] = {234, 345, 123, 0};

    auto d3 = _mm_load_si128((__m128i *)data2);
    print_m128i(d3, "d3");

}


void test_mm256_load_si256() {
    int data2[8] = {1, 2, 3, 4, 5, 6, 7, 8};

    auto d3 = _mm256_load_si256((__m256i *)data2);
    print_m256i(d3, "d3");
}


int main(int argc, char** argv)
{
    // test_1();

    // test_2();

    // test_3();

    // test_mask_load();

    // test_mm_cvtepu8_epi32();

    test_mm256_load_si256();

    // std::cout << 32.0 / 3 << std::endl;

    return 0;
}
