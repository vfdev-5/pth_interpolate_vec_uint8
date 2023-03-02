#include <iostream>
#include <immintrin.h>
#include <cassert>


using uint8 = unsigned char;

template<typename scalar_t=uint8>
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


template<typename scalar_t=uint8>
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


// Vertical pass: ImagingResampleVerticalConvolution8u
void test_ImagingResampleVerticalConvolution8u()
{
    // Vertical pass: ImagingResampleVerticalConvolution8u

    unsigned char *lineIn;
    unsigned char *lineOut;

    // a = list(range(4 * 20 * 3))
    // s = 20 * 3
    // for i in range(4):
    //     b = []
    //     for i, v in enumerate(a[s * i:s * (i + 1)]):
    //         if i > 0 and i % 3 == 0:
    //             b.append(255)
    //             b.append(v)
    //         else:
    //             b.append(v)
    //     b.append(255)
    //     print(b)

    constexpr int num_channels = 4;

    unsigned char data[4 * 20 * num_channels] = {
        0, 1, 2, 255, 3, 4, 5, 255, 6, 7, 8, 255, 9, 10, 11, 255, 12, 13, 14, 255, 15, 16, 17, 255, 18, 19, 20, 255, 21, 22, 23, 255, 24, 25, 26, 255, 27, 28, 29, 255, 30, 31, 32, 255, 33, 34, 35, 255, 36, 37, 38, 255, 39, 40, 41, 255, 42, 43, 44, 255, 45, 46, 47, 255, 48, 49, 50, 255, 51, 52, 53, 255, 54, 55, 56, 255, 57, 58, 59, 255,
        60, 61, 62, 255, 63, 64, 65, 255, 66, 67, 68, 255, 69, 70, 71, 255, 72, 73, 74, 255, 75, 76, 77, 255, 78, 79, 80, 255, 81, 82, 83, 255, 84, 85, 86, 255, 87, 88, 89, 255, 90, 91, 92, 255, 93, 94, 95, 255, 96, 97, 98, 255, 99, 100, 101, 255, 102, 103, 104, 255, 105, 106, 107, 255, 108, 109, 110, 255, 111, 112, 113, 255, 114, 115, 116, 255, 117, 118, 119, 255,
        120, 121, 122, 255, 123, 124, 125, 255, 126, 127, 128, 255, 129, 130, 131, 255, 132, 133, 134, 255, 135, 136, 137, 255, 138, 139, 140, 255, 141, 142, 143, 255, 144, 145, 146, 255, 147, 148, 149, 255, 150, 151, 152, 255, 153, 154, 155, 255, 156, 157, 158, 255, 159, 160, 161, 255, 162, 163, 164, 255, 165, 166, 167, 255, 168, 169, 170, 255, 171, 172, 173, 255, 174, 175, 176, 255, 177, 178, 179, 255,
        180, 181, 182, 255, 183, 184, 185, 255, 186, 187, 188, 255, 189, 190, 191, 255, 192, 193, 194, 255, 195, 196, 197, 255, 198, 199, 200, 255, 201, 202, 203, 255, 204, 205, 206, 255, 207, 208, 209, 255, 210, 211, 212, 255, 213, 214, 215, 255, 216, 217, 218, 255, 219, 220, 221, 255, 222, 223, 224, 255, 225, 226, 227, 255, 228, 229, 230, 255, 231, 232, 233, 255, 234, 235, 236, 255, 237, 238, 239, 255
    };

    unsigned char output[2 * 20 * num_channels] = {};

    lineIn = &data[0];
    lineOut = &output[0];

    print_data(data, 1 * 20 * num_channels, "input");

    auto stride = num_channels * 1;  // num channels * sizeof(uint8)

    // (20, 4)
    // coefs_precision=16, ksize=5
    // x=0, xsize=20, xmin=0, xmax=3
    // k: 28087 28087 9362
    // x=0, xsize=20, xmin=1, xmax=3
    // k: 9362 28087 28087
    // (20, 2)

    int xsize = 20;
    INT16 kk[5 * 4] = {
        28087, 28087, 9362, 0, 0,
        9362, 28087, 28087, 0, 0,
    };
    int ksize = 5;
    int kmax = ksize;

    int coefs_precision = 16;

    // int y = 0;
    int y = 1;

    int xmin, xmax, x;
    if (y == 1) {
        xmin = 1;
        xmax = 3;
    } else if (y == 0) {
        xmin = 0;
        xmax = 3;
    }

    // xsize = output width, xx = output x index
    // xmax = interpolation size, x = interpolation index (vertical <-> y dimension)
    // xmin = input y start index

    INT16 *k = &kk[y * ksize];
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

        std::cout << "Weights as uint8: " << std::endl;
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

    const int64_t data_size = xsize * stride;
    const int64_t data_stride = stride;
    constexpr auto vec_size = 256 / 8;

    int64_t i = 0;

    __m128i initial = _mm_set1_epi32(1 << (coefs_precision - 1));
    __m256i initial_256 = _mm256_set1_epi32(1 << (coefs_precision - 1));
    auto zero = _mm_setzero_si128();
    auto zero_256 = _mm256_setzero_si256();

    const auto b8_usable_vec_stride = (vec_size / data_stride) * data_stride;

    for (; i < data_size - vec_size; i += b8_usable_vec_stride) {
        __m256i sss0 = initial_256;
        __m256i sss1 = initial_256;
        __m256i sss2 = initial_256;
        __m256i sss3 = initial_256;
        x = 0;

        std::cout << "-- B8 i=" << i << std::endl;
        print_m256i(sss0, "sss0");
        print_m256i(sss1, "sss1");
        print_m256i(sss2, "sss2");
        print_m256i(sss3, "sss3");

        for (; x < xmax - 1; x += 2) {
            std::cout << "- B8 block 2, x: " << x << std::endl;
            __m256i source, source1, source2;
            __m256i pix, mmk;

            // Load two coefficients at once
            mmk = _mm256_set1_epi32(*(int32_t*)&k[x]);

            // Load 2 lines
            source1 =
                _mm256_loadu_si256((__m256i*)(lineIn + i + data_size * (x + xmin)));
            source2 =
                _mm256_loadu_si256((__m256i*)(lineIn + i + data_size * (x + 1 + xmin)));

            source = _mm256_unpacklo_epi8(source1, source2);
            pix = _mm256_unpacklo_epi8(source, zero_256);
            sss0 = _mm256_add_epi32(sss0, _mm256_madd_epi16(pix, mmk));
            pix = _mm256_unpackhi_epi8(source, zero_256);
            sss1 = _mm256_add_epi32(sss1, _mm256_madd_epi16(pix, mmk));

            source = _mm256_unpackhi_epi8(source1, source2);
            pix = _mm256_unpacklo_epi8(source, zero_256);
            sss2 = _mm256_add_epi32(sss2, _mm256_madd_epi16(pix, mmk));
            pix = _mm256_unpackhi_epi8(source, zero_256);
            sss3 = _mm256_add_epi32(sss3, _mm256_madd_epi16(pix, mmk));
        }
        for (; x < xmax; x += 1) {
            std::cout << "- B8 block 1, x: " << x << std::endl;

            __m256i source, source1, pix, mmk;
            mmk = _mm256_set1_epi32(k[x]);
            print_m256i(mmk, "mmk");

            source1 = _mm256_loadu_si256((__m256i*)(lineIn + i + data_size * (x + xmin)));
            print_m256i(source1, "source1");

            source = _mm256_unpacklo_epi8(source1, zero_256);
            print_m256i(source, "source");
            pix = _mm256_unpacklo_epi8(source, zero_256);
            print_m256i(pix, "pix");
            sss0 = _mm256_add_epi32(sss0, _mm256_madd_epi16(pix, mmk));
            print_m256i(sss0, "sss0");

            pix = _mm256_unpackhi_epi8(source, zero_256);
            print_m256i(pix, "pix");
            sss1 = _mm256_add_epi32(sss1, _mm256_madd_epi16(pix, mmk));
            print_m256i(sss1, "sss1");

            source = _mm256_unpackhi_epi8(source1, zero_256);
            print_m256i(source, "source");
            pix = _mm256_unpacklo_epi8(source, zero_256);
            print_m256i(pix, "pix");
            sss2 = _mm256_add_epi32(sss2, _mm256_madd_epi16(pix, mmk));
            print_m256i(sss1, "sss1");

            pix = _mm256_unpackhi_epi8(source, zero_256);
            print_m256i(pix, "pix");
            sss3 = _mm256_add_epi32(sss3, _mm256_madd_epi16(pix, mmk));
            print_m256i(sss3, "sss3");
        }

        std::cout << "--" << std::endl;

        sss0 = _mm256_srai_epi32(sss0, coefs_precision);
        print_m256i(sss0, "sss0");
        sss1 = _mm256_srai_epi32(sss1, coefs_precision);
        print_m256i(sss1, "sss1");
        sss2 = _mm256_srai_epi32(sss2, coefs_precision);
        print_m256i(sss2, "sss2");
        sss3 = _mm256_srai_epi32(sss3, coefs_precision);
        print_m256i(sss3, "sss3");

        sss0 = _mm256_packs_epi32(sss0, sss1);
        print_m256i(sss0, "sss0");
        sss2 = _mm256_packs_epi32(sss2, sss3);
        print_m256i(sss2, "sss2");
        sss0 = _mm256_packus_epi16(sss0, sss2);
        print_m256i(sss0, "sss0");

        // Stores 32 bytes
        _mm256_storeu_si256((__m256i*)(lineOut + i), sss0);
        print_data(output, 20 * 1 * 3, "B8 output");
    }

    const auto b2_usable_vec_stride = (8 / data_stride) * data_stride;
    // TODO: can we make b2_usable_vec_stride as (16 / data_stride) * data_stride ?
    for (; i < data_size - vec_size / 2; i += b2_usable_vec_stride) {
        __m128i sss0 = initial; // left row
        __m128i sss1 = initial; // right row
        x = 0;
        std::cout << "-- B2 i=" << i << std::endl;
        print_m128i(sss0, "sss0");
        print_m128i(sss1, "sss1");

        for (; x < xmax - 1; x += 2) {
            std::cout << "- B2 block 2, x: " << x << std::endl;
            __m128i source, source1, source2;
            __m128i pix, mmk;

            // Load two coefficients at once
            mmk = _mm_set1_epi32(*(int32_t*)&k[x]);

            // Load 2 lines
            source1 = _mm_loadl_epi64((__m128i*)(lineIn + i + data_size * (x + xmin)));
            source2 = _mm_loadl_epi64((__m128i*)(lineIn + i + data_size * (x + 1 + xmin)));

            source = _mm_unpacklo_epi8(source1, source2);
            pix = _mm_unpacklo_epi8(source, zero);
            sss0 = _mm_add_epi32(sss0, _mm_madd_epi16(pix, mmk));
            pix = _mm_unpackhi_epi8(source, zero);
            sss1 = _mm_add_epi32(sss1, _mm_madd_epi16(pix, mmk));
        }
        for (; x < xmax; x += 1) {
            std::cout << "- B2 block 1, x: " << x << std::endl;

            __m128i source, source1, pix, mmk;
            mmk = _mm_set1_epi32(k[x]);
            print_m128i(mmk, "mmk");

            source1 = _mm_loadl_epi64((__m128i*)(lineIn + i + data_size * (x + xmin)));
            print_m128i(source1, "source1");

            source = _mm_unpacklo_epi8(source1, zero);
            print_m128i(source, "source");

            pix = _mm_unpacklo_epi8(source, zero);
            print_m128i(pix, "pix");
            sss0 = _mm_add_epi32(sss0, _mm_madd_epi16(pix, mmk));
            print_m128i(sss0, "sss0");

            pix = _mm_unpackhi_epi8(source, zero);
            print_m128i(pix, "pix");
            sss1 = _mm_add_epi32(sss1, _mm_madd_epi16(pix, mmk));
            print_m128i(sss1, "sss1");

        }

        std::cout << "--" << std::endl;

        sss0 = _mm_srai_epi32(sss0, coefs_precision);
        print_m128i(sss0, "sss0");
        sss1 = _mm_srai_epi32(sss1, coefs_precision);
        print_m128i(sss1, "sss1");

        sss0 = _mm_packs_epi32(sss0, sss1);
        print_m128i(sss0, "sss0");
        sss0 = _mm_packus_epi16(sss0, zero);
        print_m128i(sss0, "sss0");

        _mm_storel_epi64((__m128i*)(lineOut + i), sss0);

        print_data(output, 20 * 1 * 3, "B2 output");
    }

    const auto b1_usable_vec_stride = (4 / data_stride) * data_stride;

    for (; i < data_size - 4; i += b1_usable_vec_stride) {
        __m128i sss = initial;
        x = 0;

        std::cout << "-- B1 i=" << i << std::endl;
        print_m128i(sss, "sss");

        for (; x < xmax - 1; x += 2) {
            std::cout << "- B1 block 2, x: " << x << std::endl;
            __m128i source, source1, source2;
            __m128i pix, mmk;

            // Load two coefficients at once
            mmk = _mm_set1_epi32(*(int32_t*)&k[x]);
            print_m128i(mmk, "mmk");

            // Load 2 lines
            source1 = _mm_cvtsi32_si128(*(int*)(lineIn + i + data_size * (x + xmin)));
            print_m128i(source1, "source1");
            source2 = _mm_cvtsi32_si128(*(int*)(lineIn + i + data_size * (x + 1 + xmin)));
            print_m128i(source2, "source2");

            source = _mm_unpacklo_epi8(source1, source2);
            print_m128i(source, "source");
            pix = _mm_unpacklo_epi8(source, zero);
            print_m128i(pix, "pix");
            sss = _mm_add_epi32(sss, _mm_madd_epi16(pix, mmk));
            print_m128i(sss, "sss");
        }

        for (; x < xmax; x++) {
            std::cout << "- B1 block 1, x: " << x << std::endl;

            __m128i pix = mm_cvtepu8_epi32(lineIn + i + data_size * (x + xmin));
            print_m128i(pix, "pix");
            __m128i mmk = _mm_set1_epi32(k[x]);
            print_m128i(mmk, "mmk");
            sss = _mm_add_epi32(sss, _mm_madd_epi16(pix, mmk));
            print_m128i(sss, "sss");
        }

        std::cout << "--" << std::endl;

        sss = _mm_srai_epi32(sss, coefs_precision);
        print_m128i(sss, "sss");
        // print_m128i sss: 25 0 0 0 26 0 0 0 27 0 0 0 X 0 0 0
        sss = _mm_packs_epi32(sss, zero);
        print_m128i(sss, "sss");
        // print_m128i sss: 25 0 26 0 27 0 X 0 0 0 0 0 0 0 0 0
        sss = _mm_packus_epi16(sss, zero);
        print_m128i(sss, "sss");
        // print_m128i sss: 25 26 27 X 0 0 0 0 0 0 0 0 0 0 0 0

        if (num_channels == 3)
        {
            // replace X by 0
            auto mask = _mm_set_epi8(
                -1, -1, -1, -1,  -1, -1, -1, -1,  -1, -1, -1, -1,  -1, 2, 1, 0
            );
            sss = _mm_shuffle_epi8(sss, mask);
        }
        print_m128i(sss, "sss");
        // print_m128i sss: 25 26 27 0 0 0 0 0 0 0 0 0 0 0 0 0

        auto o = _mm_cvtsi128_si32(sss);
        for (int ch=0; ch < num_channels; ch++) {
            (lineOut + i)[ch] = ((unsigned char *) &o)[ch];
        }
    }

    for (; i < data_size; i += data_stride) {
        __m128i sss = initial;
        x = 0;
        std::cout << "-- B0 i=" << i << std::endl;

        for (; x < xmax - 1; x += 2) {
            std::cout << "- B0 block 2, x: " << x << std::endl;
            __m128i source, source1, source2;
            __m128i pix, mmk;

            // Load two coefficients at once
            mmk = _mm_set1_epi32(*(int32_t*)&k[x]);
            print_m128i(mmk, "mmk");

            // Load 2 lines
            source1 = _mm_cvtsi32_si128(*(int*)(lineIn + i + data_size * (x + xmin)));
            print_m128i(source1, "source1");
            source2 = _mm_cvtsi32_si128(*(int*)(lineIn + i + data_size * (x + 1 + xmin)));
            print_m128i(source2, "source2");

            source = _mm_unpacklo_epi8(source1, source2);
            print_m128i(source, "source");
            pix = _mm_unpacklo_epi8(source, zero);
            print_m128i(pix, "pix");
            sss = _mm_add_epi32(sss, _mm_madd_epi16(pix, mmk));
            print_m128i(sss, "sss");
        }

        int output[4] = {0, 0, 0, 0};
        for (; x < xmax; x++) {
            std::cout << "- B0 block 1, x: " << x << std::endl;
            auto p = lineIn + i + data_size * (x + xmin);
            for (int ch=0; ch < num_channels; ch++) {
                output[ch] += (*(p + ch) * k[x]);
            }
        }

        sss = _mm_add_epi32(sss, _mm_load_si128((__m128i *)output));

        std::cout << "--" << std::endl;

        sss = _mm_srai_epi32(sss, coefs_precision);
        print_m128i(sss, "sss");
        // print_m128i sss: 25 0 0 0 26 0 0 0 27 0 0 0 X 0 0 0
        sss = _mm_packs_epi32(sss, zero);
        print_m128i(sss, "sss");
        // print_m128i sss: 25 0 26 0 27 0 X 0 0 0 0 0 0 0 0 0
        sss = _mm_packus_epi16(sss, zero);
        print_m128i(sss, "sss");
        // print_m128i sss: 25 26 27 X 0 0 0 0 0 0 0 0 0 0 0 0

        if (num_channels == 3)
        {
            // replace X by 0
            auto mask = _mm_set_epi8(
                -1, -1, -1, -1,  -1, -1, -1, -1,  -1, -1, -1, -1,  -1, 2, 1, 0
            );
            sss = _mm_shuffle_epi8(sss, mask);
        }
        print_m128i(sss, "sss");
        // print_m128i sss: 25 26 27 0 0 0 0 0 0 0 0 0 0 0 0 0

        auto o = _mm_cvtsi128_si32(sss);
        for (int ch=0; ch < num_channels; ch++) {
            (lineOut + i)[ch] = ((unsigned char *) &o)[ch];
        }
    }

    print_data(output, 20 * 1 * 4, "output");
    // print_data output: 43 44 45 255 46 47 48 255 49 50 51 255 52 53 54 255 55 56 57 255 58 59 60 255 61 62 63 255 64 65 66 255 67 68 69 255 70 71 72 255 73 74 75 255 76 77 78 255 79 80 81 255 82 83 84 255 85 86 87 255 88 89 90 255 91 92 93 255 94 95 96 255 97 98 99 255 100 101 102 255

    // print_data output:
    // Expected: [[43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102]]
    unsigned char expected[20 * 2 * 3] = {
        43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102,
        137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196
    };
    int j = 0;
    for (int i=0; i<(20 * 1 * 3); i++) {
        assert(output[j] == expected[i + y * 20 * 3]);
        j++;
        if ((j + 1) % 4 == 0) {
            j++;
        }
    }

}

// Horizontal pass: ImagingResampleHorizontalConvolution8u4x
void test_ImagingResampleHorizontalConvolution8u4x()
{

    // What does ImagingResampleHorizontalConvolution8u4x exactly ?
    // https://github.com/uploadcare/pillow-simd/blob/668aa48d12305b8f093958792a5e4f690c2583d6/src/libImaging/ResampleSIMDHorizontalConv.c

    unsigned char *lineIn0, *lineIn1, *lineIn2, *lineIn3;
    unsigned char *lineOut0, *lineOut1, *lineOut2, *lineOut3;

    // We need 4 lines with 8 values
    // a = list(range(8 * 9 * 3))
    // s = 9 * 3
    // for i in range(8):
    //     b = []
    //     for i, v in enumerate(a[s * i:s * (i + 1)]):
    //         if i > 0 and i % 3 == 0:
    //             b.append(255)
    //             b.append(v)
    //         else:
    //             b.append(v)
    //     b.append(255)
    //     print(b)

    unsigned char data[8 * 9 * 4] = {
        0, 1, 2, 255, 3, 4, 5, 255, 6, 7, 8, 255, 9, 10, 11, 255, 12, 13, 14, 255, 15, 16, 17, 255, 18, 19, 20, 255, 21, 22, 23, 255, 24, 25, 26, 255,
        27, 28, 29, 255, 30, 31, 32, 255, 33, 34, 35, 255, 36, 37, 38, 255, 39, 40, 41, 255, 42, 43, 44, 255, 45, 46, 47, 255, 48, 49, 50, 255, 51, 52, 53, 255,
        54, 55, 56, 255, 57, 58, 59, 255, 60, 61, 62, 255, 63, 64, 65, 255, 66, 67, 68, 255, 69, 70, 71, 255, 72, 73, 74, 255, 75, 76, 77, 255, 78, 79, 80, 255,
        81, 82, 83, 255, 84, 85, 86, 255, 87, 88, 89, 255, 90, 91, 92, 255, 93, 94, 95, 255, 96, 97, 98, 255, 99, 100, 101, 255, 102, 103, 104, 255, 105, 106, 107, 255,
        108, 109, 110, 255, 111, 112, 113, 255, 114, 115, 116, 255, 117, 118, 119, 255, 120, 121, 122, 255, 123, 124, 125, 255, 126, 127, 128, 255, 129, 130, 131, 255, 132, 133, 134, 255,
        135, 136, 137, 255, 138, 139, 140, 255, 141, 142, 143, 255, 144, 145, 146, 255, 147, 148, 149, 255, 150, 151, 152, 255, 153, 154, 155, 255, 156, 157, 158, 255, 159, 160, 161, 255,
        162, 163, 164, 255, 165, 166, 167, 255, 168, 169, 170, 255, 171, 172, 173, 255, 174, 175, 176, 255, 177, 178, 179, 255, 180, 181, 182, 255, 183, 184, 185, 255, 186, 187, 188, 255,
        189, 190, 191, 255, 192, 193, 194, 255, 195, 196, 197, 255, 198, 199, 200, 255, 201, 202, 203, 255, 204, 205, 206, 255, 207, 208, 209, 255, 210, 211, 212, 255, 213, 214, 215, 255
    };

    unsigned char output[8 * 2 * 4] = {
        0, 0, 0, 0, 0, 0, 0, 0,  // line 0
        0, 0, 0, 0, 0, 0, 0, 0,  // line 1
        0, 0, 0, 0, 0, 0, 0, 0,  // line 2
        0, 0, 0, 0, 0, 0, 0, 0,  // line 3

        0, 0, 0, 0, 0, 0, 0, 0,  // line 0
        0, 0, 0, 0, 0, 0, 0, 0,  // line 1
        0, 0, 0, 0, 0, 0, 0, 0,  // line 2
        0, 0, 0, 0, 0, 0, 0, 0   // line 3
    };

    constexpr int num_channels = 4;
    auto stride = num_channels * 1;  // num channels * sizeof(uint8)

    // (9, 8)
    // xx=0, xsize=2, xmin=0, xmax=7, coefs_precision=17
    // k: 20307 27691 31383 23999 16615 9230 1846
    // xx=1, xsize=2, xmin=2, xmax=7, coefs_precision=17
    // k: 1846 9230 16615 23999 31383 27691 20307
    // xx=0, xsize=2, xmin=0, xmax=7, coefs_precision=17
    // k: 20307 27691 31383 23999 16615 9230 1846
    // xx=1, xsize=2, xmin=2, xmax=7, coefs_precision=17
    // k: 1846 9230 16615 23999 31383 27691 20307
    // (2, 8)

    int xsize = 2;
    int xbounds[2 * 2] = {0, 7, 2, 7};
    INT16 kk[9 * 2] = {
        20307, 27691, 31383, 23999, 16615, 9230, 1846, 0, 0,
        1846, 9230, 16615, 23999, 31383, 27691, 20307, 0, 0
    };
    int kmax = 9;
    int coefs_precision = 17;

    for (int part=0; part < 2; part++) {

        lineIn0 = &data[(4 * part + 0) * 9 * num_channels];
        lineIn1 = &data[(4 * part + 1) * 9 * num_channels];
        lineIn2 = &data[(4 * part + 2) * 9 * num_channels];
        lineIn3 = &data[(4 * part + 3) * 9 * num_channels];

        lineOut0 = &output[(4 * part + 0) * 2 * num_channels];
        lineOut1 = &output[(4 * part + 1) * 2 * num_channels];
        lineOut2 = &output[(4 * part + 2) * 2 * num_channels];
        lineOut3 = &output[(4 * part + 3) * 2 * num_channels];

        std::cout << "\n----- part = " << part << " -----" << std::endl;

        print_uint32(((UINT32 *)lineIn0)[0], "lineIn0");
        print_uint32(((UINT32 *)lineIn1)[0], "lineIn1");
        print_uint32(((UINT32 *)lineIn2)[0], "lineIn2");
        print_uint32(((UINT32 *)lineIn3)[0], "lineIn3");

        int xmin, xmax, xx;
        INT16 *k;

        // xsize = output width, xx = output x index
        // xmax = interpolation size, x = interpolation index (horizontal <-> x dimension)
        // xmin = input x start index corresponding to output x index (xx)

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

            std::cout << "Weights as uint8: " << std::endl;
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

        __m256i mask_low, mask_high;
        if (num_channels == 4) {
            mask_low = _mm256_set_epi8(
                -1,7, -1,3, -1,6, -1,2, -1,5, -1,1, -1,4, -1,0,
                -1,7, -1,3, -1,6, -1,2, -1,5, -1,1, -1,4, -1,0
            );
            mask_high = _mm256_set_epi8(
                -1,15, -1,11, -1,14, -1,10, -1,13, -1,9, -1,12, -1,8,
                -1,15, -1,11, -1,14, -1,10, -1,13, -1,9, -1,12, -1,8
            );
        } else if (num_channels == 3) {
            mask_low = _mm256_set_epi8(
                -1,-1, -1,-1, -1,5, -1,2, -1,4, -1,1, -1,3, -1,0,
                -1,-1, -1,-1, -1,5, -1,2, -1,4, -1,1, -1,3, -1,0
            );
            mask_high = _mm256_set_epi8(
                -1,-1, -1,-1, -1,11, -1,8, -1,10, -1,7, -1,9, -1,6,
                -1,-1, -1,-1, -1,11, -1,8, -1,10, -1,7, -1,9, -1,6
            );
        }

        for (xx = 0; xx < xsize; xx++) {
            xmin = xbounds[xx * 2 + 0];
            xmax = xbounds[xx * 2 + 1];
            k = &kk[xx * kmax];

            int64_t x = 0;

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

            // lineIn0 + stride * (x + xmin) + 16 <= lineIn0 + stride * (xmax + xmin)
            // --> x <= xmax - 16.0 / stride
            auto b4_xmax = xmax - 16.0 / stride + 1;

            for (; x < b4_xmax; x += 4) {
                std::cout << "- block 4, x: " << x << std::endl;
                __m256i pix, mmk0, mmk1, source;

                // Load 2 weight values
                mmk0 = _mm256_set1_epi32(*(INT32 *) &k[x]);
                print_m256i(mmk0, "mmk0");
                mmk1 = _mm256_set1_epi32(*(INT32 *) &k[x + 2]);
                print_m256i(mmk1, "mmk1");

                source = _mm256_inserti128_si256(_mm256_castsi128_si256(
                    _mm_loadu_si128((__m128i *) (lineIn0 + stride * (x + xmin)))),
                    _mm_loadu_si128((__m128i *) (lineIn1 + stride * (x + xmin))), 1);
                print_m256i(source, "source");

                pix = _mm256_shuffle_epi8(source, mask_low);
                print_m256i(pix, "pix");
                sss0 = _mm256_add_epi32(sss0, _mm256_madd_epi16(pix, mmk0));
                print_m256i(sss0, "sss0");

                pix = _mm256_shuffle_epi8(source, mask_high);
                print_m256i(pix, "pix");
                sss0 = _mm256_add_epi32(sss0, _mm256_madd_epi16(pix, mmk1));
                print_m256i(sss0, "sss0");

                source = _mm256_inserti128_si256(_mm256_castsi128_si256(
                    _mm_loadu_si128((__m128i *) (lineIn2 + stride * (x + xmin)))),
                    _mm_loadu_si128((__m128i *) (lineIn3 + stride * (x + xmin))), 1);
                print_m256i(source, "source");
                pix = _mm256_shuffle_epi8(source, mask_low);
                print_m256i(pix, "pix");
                sss1 = _mm256_add_epi32(sss1, _mm256_madd_epi16(pix, mmk0));
                print_m256i(sss1, "sss1");
                pix = _mm256_shuffle_epi8(source, mask_high);
                print_m256i(pix, "pix");
                sss1 = _mm256_add_epi32(sss1, _mm256_madd_epi16(pix, mmk1));
                print_m256i(sss1, "sss1");
            }

            // lineIn0 + stride * (x + xmin) + 16 <= lineIn0 + stride * (xmax + xmin)
            // --> x <= xmax - 16.0 / stride
            auto b2_xmax = xmax - 16.0 / stride + 1;

            for (; x < b2_xmax; x += 2) {
                std::cout << "- block 2, x: " << x << std::endl;
                __m256i pix, mmk;

                // - Load int16 weights taking 2 values
                mmk = _mm256_set1_epi32(*(INT32 *) &k[x]);
                print_m256i(mmk, "mmk");
                // print_m256i mmk: 183 45 0 64 183 45 0 64 183 45 0 64 183 45 0 64 183 45 0 64 183 45 0 64 183 45 0 64 183 45 0 64
                // (183 45 0 64) <-> 183 + 256 * 45 = 11703, 64 * 256 = 16384
                // mmk ~ {kk[0], kk[1], kk[0], kk[1], ... kk[0], kk[1]}

                auto a0 = _mm_loadl_epi64((__m128i *) (lineIn0 + stride * (x + xmin)));
                print_m128i(a0, "a0");
                // print_m128i a0: 0 1 2 255 3 4 5 255 0 0 0 0 0 0 0 0
                auto b0 = _mm256_castsi128_si256(a0);
                print_m256i(b0, "b0");
                // print_m256i b0: 0 1 2 255 3 4 5 255 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                auto a1 = _mm_loadl_epi64((__m128i *) (lineIn1 + stride * (x + xmin)));
                print_m128i(a1, "a1");
                // print_m128i a1: 24 25 26 255 27 28 29 255 0 0 0 0 0 0 0 0
                pix = _mm256_inserti128_si256(b0, a1, 1);
                print_m256i(pix, "1 pix");
                // print_m256i 1 pix: 0 1 2 255 3 4 5 255 0 0 0 0 0 0 0 0 24 25 26 255 27 28 29 255 0 0 0 0 0 0 0 0
                auto m0 = mask_low;
                print_m256i(m0, "m0");
                // print_m256i m0: 0 255 4 255 1 255 5 255 2 255 6 255 3 255 7 255 0 255 4 255 1 255 5 255 2 255 6 255 3 255 7 255
                pix = _mm256_shuffle_epi8(pix, m0);
                print_m256i(pix, "2 pix");
                // print_m256i 2 pix: 0 0 3 0 1 0 4 0 2 0 5 0 255 0 255 0 24 0 27 0 25 0 28 0 26 0 29 0 255 0 255 0
                // - Interleave pixels from the same line and expand to uint16 (reusing zeros)
                // (
                //   line0_pixel0_R, 0, line0_pixel1_R, 0, line0_pixel0_G, 0, line0_pixel1_G, 0, line0_pixel0_B, 0, line0_pixel1_B, 0, 255, 0, 255, 0,
                //   line1_pixel0_R, 0, line1_pixel1_R, 0, line1_pixel0_G, 0, line1_pixel1_G, 0, line1_pixel0_B, 0, line1_pixel1_B, 0, 255, 0, 255, 0,
                // )

                auto tmp0 = _mm256_madd_epi16(pix, mmk);
                print_m256i(tmp0, "tmp0");
                // print_m256i tmp0: 0 192 0 0 183 45 1 0 110 155 1 0 73 73 109 0 40 9 11 0 223 118 11 0 150 228 11 0 73 73 109 0
                // tmp0 = pix * mmk
                // tmp0 = (t1, t2, t3, t4, t5, t6, ..., t31, t32)
                //
                // (t1, t2, t3, t4) = (line0_pixel0_R, 0) * (mmk[0], mmk[1]) + (line0_pixel1_R, 0) * (mmk[2], mmk[3])
                //                  = (0, 0) * (183 45) + (3 0) * (0 64) = 0 * 11703 + 3 * 64 * 256 = 192 * 256 <=> (0, 192, 0, 0)
                // (t5, t6, t7, t8) = (line0_pixel0_G, 0) * (mmk[4], mmk[5]) + (line0_pixel1_G, 0) * (mmk[6], mmk[7])
                //                  = (1, 0) * (183 45) + (4 0) * (0 64) = 1 * 11703 + 4 * 64 * 256 = 77239 => (183 45 1 0)
                // ...
                sss0 = _mm256_add_epi32(sss0, tmp0);
                // sss0 = initial + pix * mmk + ...
                print_m256i(sss0, "sss0");

                pix = _mm256_inserti128_si256(_mm256_castsi128_si256(
                    _mm_loadl_epi64((__m128i *) (lineIn2 + stride * (x + xmin)))),
                    _mm_loadl_epi64((__m128i *) (lineIn3 + stride * (x + xmin))), 1);
                pix = _mm256_shuffle_epi8(pix, mask_low);
                sss1 = _mm256_add_epi32(sss1, _mm256_madd_epi16(pix, mmk));
                print_m256i(sss1, "sss1");
            }

            for (; x < xmax - 1; x++) {
                std::cout << "- block 1, x: " << x << std::endl;
                __m256i pix, mmk;

                mmk = _mm256_set1_epi32(k[x]);
                // 123 34 0 0
                print_m256i(mmk, "mmk");
                // print_m256i mmk: 123 34 0 0 123 34 0 0 123 34 0 0 123 34 0 0 123 34 0 0 123 34 0 0 123 34 0 0

                auto a0 = mm_cvtepu8_epi32(lineIn0 + stride * (x + xmin));
                print_m128i(a0, "a0");
                // print_m128i a0: 0 0 0 0 1 0 0 0 2 0 0 0 255 0 0 0
                auto b0 = _mm256_castsi128_si256(a0);
                print_m256i(b0, "b0");
                // print_m256i b0: 0 0 0 0 1 0 0 0 2 0 0 0 255 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                auto a1 = mm_cvtepu8_epi32(lineIn1 + stride * (x + xmin));
                print_m128i(a1, "a1");
                // print_m128i a1: 18 0 0 0 19 0 0 0 20 0 0 0 255 0 0 0
                pix = _mm256_inserti128_si256(b0, a1, 1);
                print_m256i(pix, "pix");
                // print_m256i pix: 0 0 0 0 1 0 0 0 2 0 0 0 255 0 0 0 18 0 0 0 19 0 0 0 20 0 0 0 255 0 0 0
                auto tmp0 = _mm256_madd_epi16(pix, mmk);
                // For example:
                //   pix: (1 0 100 0) == 1, 100
                //   mmk: (58 1 0 125) == 314, 32000
                //   tmp: 1 * 314 + 100 * 32000 = 3200314 <-> 58 213 48 0 <--> 58 + 213 * 256 + 48 * 256 * 256
                print_m256i(tmp0, "tmp0");
                // print_m256i tmp0: 0 0 0 0 183 45 0 0 110 91 0 0 73 137 45 0 222 54 3 0 149 100 3 0 76 146 3 0 73 137 45 0
                sss0 = _mm256_add_epi32(sss0, tmp0);
                print_m256i(sss0, "sss0");
                // print_m256i sss0: 0 128 0 0 183 173 0 0 110 219 0 0 73 9 46 0 40 201 4 0 223 246 4 0 150 36 5 0 73 9 46 0
                // sss0 = initial + pix[0] * mmk[0] + pix[1] * mmk[1] + ...
                // for lines 0 and 1

                pix = _mm256_inserti128_si256(
                    _mm256_castsi128_si256(
                        mm_cvtepu8_epi32(
                            lineIn2 + stride * (x + xmin)
                        )
                    ),
                    mm_cvtepu8_epi32(
                        lineIn3 + stride * (x + xmin)
                    ),
                    1
                );
                print_m256i(pix, "pix");
                auto tmp1 = _mm256_madd_epi16(pix, mmk);
                print_m256i(tmp1, "tmp1");
                sss1 = _mm256_add_epi32(sss1, tmp1);
                print_m256i(sss1, "sss1");
                // sss1 = initial + pix[0] * mmk[0] + pix[1] * mmk[1] + ...
                // for lines 2 and 3
            }

            // last element x = xmax - 1
            {
                x = xmax - 1;
                std::cout << "- block 0, x: " << x << std::endl;
                int output0[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                int output1[8] = {0, 0, 0, 0, 0, 0, 0, 0};

                auto p0 = lineIn0 + stride * (x + xmin);
                auto p1 = lineIn1 + stride * (x + xmin);
                auto p2 = lineIn2 + stride * (x + xmin);
                auto p3 = lineIn3 + stride * (x + xmin);
                for (int ch=0; ch < num_channels; ch++) {
                    output0[0 + ch] += (*(p0 + ch) * k[x]);
                    output0[4 + ch] += (*(p1 + ch) * k[x]);
                    output1[0 + ch] += (*(p2 + ch) * k[x]);
                    output1[4 + ch] += (*(p3 + ch) * k[x]);
                }

                auto t0 = _mm256_load_si256((__m256i *)output0);
                auto t1 = _mm256_load_si256((__m256i *)output1);
                print_m256i(t0, "t0");
                print_m256i(t1, "t1");

                sss0 = _mm256_add_epi32(sss0, _mm256_load_si256((__m256i *)output0));
                sss1 = _mm256_add_epi32(sss1, _mm256_load_si256((__m256i *)output1));
            }

            std::cout << "--" << std::endl;

            // Shift packed 32-bit integers in a right by imm8 while shifting in sign bits, and store the results in dst.
            print_m256i(sss0, "sss0 ->");
            //
            sss0 = _mm256_srai_epi32(sss0, coefs_precision);
            print_m256i(sss0, "-> sss0");
            // print_m256i -> sss0: 15 0 0 0 16 0 0 0 17 0 0 0 255 0 0 0 39 0 0 0 40 0 0 0 41 0 0 0 255 0 0 0
            print_m256i(sss1, "sss1 ->");
            sss1 = _mm256_srai_epi32(sss1, coefs_precision);
            print_m256i(sss1, "-> sss1");
            // print_m256i -> sss1: 63 0 0 0 64 0 0 0 65 0 0 0 255 0 0 0 87 0 0 0 88 0 0 0 89 0 0 0 255 0 0 0

            // Convert packed signed 32-bit integers from a and b to packed 16-bit integers using signed saturation,
            // and store the results in dst.
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#ig_expand=5381,5381,6389,2316,2310,6688,5148,5178,5148&text=_mm256_packs_epi32
            sss0 = _mm256_packs_epi32(sss0, zero);
            print_m256i(sss0, "sss0");
            // print_m256i sss0: 15 0 16 0 17 0 255 0 0 0 0 0 0 0 0 0 39 0 40 0 41 0 255 0 0 0 0 0 0 0 0 0

            sss1 = _mm256_packs_epi32(sss1, zero);
            print_m256i(sss1, "sss1");
            // print_m256i sss1: 63 0 64 0 65 0 255 0 0 0 0 0 0 0 0 0 87 0 88 0 89 0 255 0 0 0 0 0 0 0 0 0

            // Convert packed signed 16-bit integers from a and b to packed 8-bit integers using unsigned saturation,
            // and store the results in dst.
            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#ig_expand=5381,5381,6389,2316,2310,6688,5148,5178&text=_mm256_packus_epi16
            sss0 = _mm256_packus_epi16(sss0, zero);
            print_m256i(sss0, "sss0");
            // print_m256i sss0: 15 16 17 255 0 0 0 0 0 0 0 0 0 0 0 0 39 40 41 255 0 0 0 0 0 0 0 0 0 0 0 0

            sss1 = _mm256_packus_epi16(sss1, zero);
            print_m256i(sss1, "sss1");
            // print_m256i sss1: 63 64 65 255 0 0 0 0 0 0 0 0 0 0 0 0 87 88 89 255 0 0 0 0 0 0 0 0 0 0 0 0

            if (num_channels == 3) {
                // replace X by 0
                auto mask = _mm256_set_epi8(
                    -1, -1, -1, -1,  -1, -1, -1, -1,  -1, -1, -1, -1,  -1, 2, 1, 0,
                    -1, -1, -1, -1,  -1, -1, -1, -1,  -1, -1, -1, -1,  -1, 2, 1, 0
                );
                sss0 = _mm256_shuffle_epi8(sss0, mask);
                sss1 = _mm256_shuffle_epi8(sss1, mask);
            }

            // _mm_cvtsi128_si32: Copy the lower 32-bit integer in a to dst.
            // _mm256_extracti128_si256: Extract 128 bits (composed of integer data) from a, selected with imm8, and store the result in dst.
            auto e0 = _mm256_extracti128_si256(sss0, 0);
            print_m128i(e0, "e0");
            // print_m128i e0: 15 16 17 255 0 0 0 0 0 0 0 0 0 0 0 0
            auto e1 = _mm256_extracti128_si256(sss0, 1);
            print_m128i(e1, "e1");
            // print_m128i e1: 39 40 41 255 0 0 0 0 0 0 0 0 0 0 0 0

            auto e2 = _mm256_extracti128_si256(sss1, 0);
            print_m128i(e2, "e2");
            // print_m128i e2: 63 64 65 255 0 0 0 0 0 0 0 0 0 0 0 0

            auto e3 = _mm256_extracti128_si256(sss1, 1);
            print_m128i(e3, "e3");
            // print_m128i e3: 87 88 89 255 0 0 0 0 0 0 0 0 0 0 0 0

            auto o0 = _mm_cvtsi128_si32(e0);
            auto o1 = _mm_cvtsi128_si32(e1);
            auto o2 = _mm_cvtsi128_si32(e2);
            auto o3 = _mm_cvtsi128_si32(e3);
            for (int ch=0; ch<num_channels; ch++) {
                (lineOut0 + stride * xx)[ch] = ((unsigned char *) &o0)[ch];
                (lineOut1 + stride * xx)[ch] = ((unsigned char *) &o1)[ch];
                (lineOut2 + stride * xx)[ch] = ((unsigned char *) &o2)[ch];
                (lineOut3 + stride * xx)[ch] = ((unsigned char *) &o3)[ch];
            }
        }

        print_data(output + part * (4 * 2 * 4), 4 * 2 * 4, "- output");
    }

    print_data(output, 8 * 2 * 4, "output");
    // print_data output:
    // Expected: [[[7, 8, 9, 17, 18, 19, 34, 35, 36, 44, 45, 46, 61, 62, 63, 71, 72, 73, 88, 89, 90, 98, 99, 100, 115, 116, 117, 125, 126, 127, 142, 143, 144, 152, 153, 154, 169, 170, 171, 179, 180, 181, 196, 197, 198, 206, 207, 208]]]

    unsigned char expected[8 * 2 * 3] = {
        7, 8, 9, 17, 18, 19, 34, 35, 36, 44, 45, 46, 61, 62, 63, 71, 72, 73, 88, 89, 90, 98, 99, 100, 115, 116, 117, 125, 126, 127, 142, 143, 144, 152, 153, 154, 169, 170, 171, 179, 180, 181, 196, 197, 198, 206, 207, 208
    };
    int j = 0;
    for (int i=0; i<(8 * 2 * 3); i++) {
        assert(output[j] == expected[i]);
        j++;
        if ((j + 1) % 4 == 0) {
            j++;
        }
    }

}

// Horizontal pass: ImagingResampleHorizontalConvolution8u
void test_ImagingResampleHorizontalConvolution8u()
{

    unsigned char *lineIn;
    unsigned char *lineOut;

    unsigned char data[2 * 18 * 4] = {
        0, 1, 2, 255, 3, 4, 5, 255, 6, 7, 8, 255, 9, 10, 11, 255, 12, 13, 14, 255, 15, 16, 17, 255, 18, 19, 20, 255, 21, 22, 23, 255, 24, 25, 26, 255,
        27, 28, 29, 255, 30, 31, 32, 255, 33, 34, 35, 255, 36, 37, 38, 255, 39, 40, 41, 255, 42, 43, 44, 255, 45, 46, 47, 255, 48, 49, 50, 255, 51, 52, 53, 255,
        54, 55, 56, 255, 57, 58, 59, 255, 60, 61, 62, 255, 63, 64, 65, 255, 66, 67, 68, 255, 69, 70, 71, 255, 72, 73, 74, 255, 75, 76, 77, 255, 78, 79, 80, 255,
        81, 82, 83, 255, 84, 85, 86, 255, 87, 88, 89, 255, 90, 91, 92, 255, 93, 94, 95, 255, 96, 97, 98, 255, 99, 100, 101, 255, 102, 103, 104, 255, 105, 106, 107, 255
    };

    unsigned char output[2 * 2 * 4] = {
        0, 0, 0, 0, 0, 0, 0, 0,  // line 0
        0, 0, 0, 0, 0, 0, 0, 0  // line 1
    };

    constexpr int num_channels = 4;
    auto stride = num_channels * 1;  // num channels * sizeof(uint8)

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
    int coefs_precision = 17;

    auto zero = _mm_setzero_si128();

    for (int part=0; part < 2; part++) {

        lineIn = &data[part * 18 * num_channels];
        lineOut = &output[part * 2 * num_channels];

        std::cout << "\n----- part = " << part << " -----" << std::endl;

        print_uint32(((UINT32 *)lineIn)[0], "lineIn");

        int xmin, xmax, xx;
        INT16 *k;

        // xsize = output width, xx = output x index
        // xmax = interpolation size, x = interpolation index (horizontal <-> x dimension)
        // xmin = input x start index corresponding to output x index (xx)

        __m256i mask_low, mask_high, mask_hl;
        __m128i mask_low128;
        if (num_channels == 4) {
            mask_low = _mm256_set_epi8(
                -1,7, -1,3, -1,6, -1,2, -1,5, -1,1, -1,4, -1,0,
                -1,7, -1,3, -1,6, -1,2, -1,5, -1,1, -1,4, -1,0
            );
            mask_high = _mm256_set_epi8(
                -1,15, -1,11, -1,14, -1,10, -1,13, -1,9, -1,12, -1,8,
                -1,15, -1,11, -1,14, -1,10, -1,13, -1,9, -1,12, -1,8
            );
            mask_hl = _mm256_set_epi8(
                -1,15, -1,11, -1,14, -1,10, -1,13, -1,9, -1,12, -1,8,
                -1,7, -1,3, -1,6, -1,2, -1,5, -1,1, -1,4, -1,0
            );
            mask_low128 = _mm_set_epi8(
                -1, 7, -1, 3, -1, 6, -1, 2, -1, 5, -1, 1, -1, 4, -1, 0
            );
        } else if (num_channels == 3) {
            mask_low = _mm256_set_epi8(
                -1,-1, -1,-1, -1,5, -1,2, -1,4, -1,1, -1,3, -1,0,
                -1,-1, -1,-1, -1,5, -1,2, -1,4, -1,1, -1,3, -1,0
            );
            mask_high = _mm256_set_epi8(
                -1,-1, -1,-1, -1,11, -1,8, -1,10, -1,7, -1,9, -1,6,
                -1,-1, -1,-1, -1,11, -1,8, -1,10, -1,7, -1,9, -1,6
            );
            mask_hl = _mm256_set_epi8(
                -1,-1, -1,-1, -1,11, -1,8, -1,10, -1,7, -1,9, -1,6,
                -1,-1, -1,-1, -1,5, -1,2, -1,4, -1,1, -1,3, -1,0
            );
            mask_low128 = _mm_set_epi8(
                -1,-1, -1,-1, -1,5, -1,2, -1,4, -1,1, -1,3, -1,0
            );
        }

        for (xx = 0; xx < xsize; xx++) {
            __m128i sss;
            xmin = xbounds[xx * 2 + 0];
            xmax = xbounds[xx * 2 + 1];
            k = &kk[xx * kmax];

            int64_t x = 0;

            std::cout << "-- xx=" << xx << std::endl;

            if (xmax < 8) {
                sss = _mm_set1_epi32(1 << (coefs_precision - 1));
            }
            else {
                // Lower part will be added to higher, use only half of the error
                __m256i sss256 = _mm256_set1_epi32(1 << (coefs_precision - 2));

                print_m256i(sss256, "sss256");
                // print_m256i sss256: 0 128 0 0 0 128 0 0 0 128 0 0 0 128 0 0 0 128 0 0 0 128 0 0 0 128 0 0 0 128 0 0

                // // lineIn + stride * (x + xmin) + 32 <= lineIn + stride * (xmax + xmin)
                // // --> x <= xmax - 32.0 / stride
                // auto b8_xmax = xmax - 32.0 / stride + 1;

                // for (; x < b8_xmax; x += 8) {
                //     std::cout << "- block 8, x: " << x << std::endl;

                //     __m256i pix, mmk, source;
                //     __m128i tmp = _mm_loadu_si128((__m128i*)&k[x]);
                //     __m256i ksource = _mm256_insertf128_si256(_mm256_castsi128_si256(tmp), tmp, 1);
                //     print_m256i(ksource, "ksource");

                //     source = _mm256_inserti128_si256(_mm256_castsi128_si256(
                //         _mm_loadu_si128((__m128i *) (lineIn + stride * (x + 0 + xmin)))),
                //         _mm_loadu_si128((__m128i *) (lineIn + stride * (x + 4 + xmin))), 1);
                //     print_m256i(source, "source");

                //     pix = _mm256_shuffle_epi8(source, mask_low);
                //     print_m256i(pix, "pix");
                //     mmk = _mm256_shuffle_epi8(ksource, _mm256_set_epi8(
                //         11,10, 9,8, 11,10, 9,8, 11,10, 9,8, 11,10, 9,8,
                //         3,2, 1,0, 3,2, 1,0, 3,2, 1,0, 3,2, 1,0)
                //     );
                //     sss256 = _mm256_add_epi32(sss256, _mm256_madd_epi16(pix, mmk));
                //     print_m256i(sss256, "sss256");

                //     pix = _mm256_shuffle_epi8(source, mask_high);
                //     print_m256i(pix, "pix");
                //     mmk = _mm256_shuffle_epi8(ksource, _mm256_set_epi8(
                //         15,14, 13,12, 15,14, 13,12, 15,14, 13,12, 15,14, 13,12,
                //         7,6, 5,4, 7,6, 5,4, 7,6, 5,4, 7,6, 5,4));
                //     sss256 = _mm256_add_epi32(sss256, _mm256_madd_epi16(pix, mmk));
                //     print_m256i(sss256, "sss256");
                // }

                // lineIn + stride * (x + xmin) + 16 <= lineIn0 + stride * (xmax + xmin)
                // --> x <= xmax - 16.0 / stride
                auto b4_xmax = xmax - 16.0 / stride + 1;

                for (; x < b4_xmax; x += 4) {
                    std::cout << "- block 4, x: " << x << std::endl;
                    __m256i pix, mmk, source;
                    __m128i tmp = _mm_loadl_epi64((__m128i*)&k[x]);
                    print_m128i(tmp, "1 tmp");
                    __m256i ksource = _mm256_insertf128_si256(_mm256_castsi128_si256(tmp), tmp, 1);
                    print_m256i(ksource, "ksource");

                    tmp = _mm_loadu_si128((__m128i*) (lineIn + stride * (x + xmin)));
                    print_m128i(tmp, "2 tmp");

                    source = _mm256_insertf128_si256(_mm256_castsi128_si256(tmp), tmp, 1);
                    print_m256i(source, "source");

                    pix = _mm256_shuffle_epi8(source, mask_hl);
                    print_m256i(pix, "pix");
                    mmk = _mm256_shuffle_epi8(ksource, _mm256_set_epi8(
                        7,6, 5,4, 7,6, 5,4, 7,6, 5,4, 7,6, 5,4,
                        3,2, 1,0, 3,2, 1,0, 3,2, 1,0, 3,2, 1,0));
                    print_m256i(mmk, "mmk");
                    sss256 = _mm256_add_epi32(sss256, _mm256_madd_epi16(pix, mmk));
                    print_m256i(sss256, "sss256");
                }

                sss = _mm_add_epi32(
                    _mm256_extracti128_si256(sss256, 0),
                    _mm256_extracti128_si256(sss256, 1));

                print_m128i(sss, "sss");
            }

            // lineIn0 + stride * (x + xmin) + 16 <= lineIn0 + stride * (xmax + xmin)
            // --> x <= xmax - 16.0 / stride
            auto b2_xmax = xmax - 16.0 / stride + 1;

            for (; x < b2_xmax; x += 2) {
                std::cout << "- block 2, x: " << x << std::endl;
                __m128i mmk = _mm_set1_epi32(*(int32_t*)&k[x]);
                __m128i source = _mm_loadl_epi64((__m128i*) (lineIn + stride * (x + xmin)));
                print_m128i(source, "source");

                __m128i pix = _mm_shuffle_epi8(source, mask_low128);
                sss = _mm_add_epi32(sss, _mm_madd_epi16(pix, mmk));
                print_m128i(sss, "sss");
            }

            for (; x < xmax - 1; x++) {
                std::cout << "- block 1, x: " << x << std::endl;
                __m128i pix = mm_cvtepu8_epi32(lineIn + stride * (x + xmin));
                print_m128i(pix, "pix");
                __m128i mmk = _mm_set1_epi32(k[x]);
                sss = _mm_add_epi32(sss, _mm_madd_epi16(pix, mmk));
                print_m128i(sss, "sss");
            }

            // last element x = xmax - 1
            {
                x = xmax - 1;
                std::cout << "- block 0, x: " << x << std::endl;
                int output[4] = {0, 0, 0, 0};
                auto p = lineIn + stride * (x + xmin);
                for (int ch=0; ch < num_channels; ch++) {
                    output[ch] += (*(p + ch) * k[x]);
                }
                sss = _mm_add_epi32(sss, _mm_load_si128((__m128i *)output));
                print_m128i(sss, "sss");
            }

            std::cout << "--" << std::endl;

            sss = _mm_srai_epi32(sss, coefs_precision);
            print_m128i(sss, "sss");
            sss = _mm_packs_epi32(sss, zero);
            print_m128i(sss, "sss");
            sss = _mm_packus_epi16(sss, zero);
            print_m128i(sss, "1 sss");

            if (num_channels == 3)
            {
                // replace X by 0
                auto mask = _mm_set_epi8(
                    -1, -1, -1, -1,  -1, -1, -1, -1,  -1, -1, -1, -1,  -1, 2, 1, 0
                );
                sss = _mm_shuffle_epi8(sss, mask);
                print_m128i(sss, "2 sss");
            }

            auto o = _mm_cvtsi128_si32(sss);
            for (int ch=0; ch < num_channels; ch++) {
                (lineOut + stride * xx)[ch] = ((unsigned char *) &o)[ch];
            }
        }
    }

    print_data(output, 2 * 2 * 4, "output");
    // print_data output:
    // Expected: [[[15, 16, 17, 36, 37, 38, 69, 70, 71, 90, 91, 92]]]

    unsigned char expected[2 * 2 * 3] = {
        15, 16, 17, 36, 37, 38, 69, 70, 71, 90, 91, 92
    };
    int j = 0;
    for (int i=0; i<(2 * 2 * 3); i++) {
        assert(output[j] == expected[i]);
        j++;
        if ((j + 1) % 4 == 0) {
            j++;
        }
    }

}

int main() {

    // test_ImagingResampleVerticalConvolution8u();

    // test_ImagingResampleHorizontalConvolution8u4x();

    test_ImagingResampleHorizontalConvolution8u();

    return 0;
}
