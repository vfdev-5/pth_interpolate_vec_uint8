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


#if 0
int main(int argc, char** argv)
{

    /// WHAT DOES https://github.com/uploadcare/pillow-simd/blob/668aa48d12305b8f093958792a5e4f690c2583d6/src/libImaging/ResampleSIMDVerticalConv.c#L14-L74

    int coefs_precision = 4;
    // random weights
    short k[10] = {6, -13, 19, -26, 32, -102, 173, -243, 314, 23};

    // size = sizeof(uint8) * 32
    unsigned char top_line[32] = {};
    unsigned char bottom_line[32] = {};
    for (int i=0; i<32; i++) {
        top_line[i] = i + 1;
        bottom_line[i] = i + 100;
    }

    __m256i initial_256 = _mm256_set1_epi32(1 << (coefs_precision-1));
    print_m256i(initial_256, "initial_256");
    // print_m256i initial_256: 32 0 0 0 32 0 0 0 32 0 0 0 32 0 0 0 32 0 0 0 32 0 0 0 32 0 0 0 32 0 0 0

    __m256i sss0 = initial_256;
    __m256i sss1 = initial_256;
    __m256i sss2 = initial_256;
    __m256i sss3 = initial_256;

    int x = 8;  // x goes from 0 to xmax == kernel size

    __m256i source, source1, source2, tmp;
    __m256i pix, mmk;
    // - Load int16 weights taking 2 values
    mmk = _mm256_set1_epi32(*(int *) &k[x]);
    print_m256i(mmk, "mmk");
    // print_m256i mmk: 58 1 0 125 58 1 0 125 58 1 0 125 58 1 0 125 58 1 0 125 58 1 0 125 58 1 0 125 58 1 0 125
    // (58 1) <-> 1 * 256 + 58 = 314
    // (0 125) <-> 125 * 256 + 0 = 32000

    // - Load two lines of image:
    source1 = _mm256_loadu_si256(  // top line
        (__m256i *) top_line);
    print_m256i(source1, "source1");
    // print_m256i source1: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32

    source2 = _mm256_loadu_si256(  // bottom line
        (__m256i *) bottom_line);
    print_m256i(source2, "source2");
    // print_m256i source2: 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131

    std::cout << " " << std::endl;

    // - Interleave source1 and source2 and expand to int16 range (appending zeros)
    source = _mm256_unpacklo_epi8(source1, source2);
    print_m256i(source, "source");
    // print_m256i source: 1 100 2 101 3 102 4 103 5 104 6 105 7 106 8 107 17 116 18 117 19 118 20 119 21 120 22 121 23 122 24 123

    std::cout << " " << std::endl;

    pix = _mm256_unpacklo_epi8(source, _mm256_setzero_si256());
    print_m256i(pix, "pix");
    // print_m256i pix: 1 0 100 0 2 0 101 0 3 0 102 0 4 0 103 0 17 0 116 0 18 0 117 0 19 0 118 0 20 0 119 0

    // - Multiply and sum pix, mmk
    //   - We need to compute: out_pixel(y, x) = k(y, x + xmin) * in(y, x + xmin) + k(y, x + xmin + 1) * in(y, x + xmin + 1) + ...
    //   - _mm256_madd_epi16 does k(y, x + xmin) * in(y, x + xmin) + k(y, x + xmin + 1) * in(y, x + xmin + 1)
    // _mm256_madd_epi16 = Multiplies signed packed 16-bit integer data elements of two vectors.
    // https://www.intel.com/content/www/us/en/develop/documentation/cpp-compiler-developer-guide-and-reference/top/compiler-reference/intrinsics/intrinsics-for-avx2/intrinsics-for-arithmetic-operations-2/mm256-madd-epi16.html
    // For example:
    //   pix: (1 0 100 0) == 1, 100
    //   mmk: (58 1 0 125) == 314, 32000
    //   tmp: 1 * 314 + 100 * 32000 = 3200314 <-> 58 213 48 0 <--> 58 + 213 * 256 + 48 * 256 * 256

    tmp = _mm256_madd_epi16(pix, mmk);
    print_m256i(tmp, "tmp");
    // print_m256i tmp: 58 213 48 0 116 83 49 0 174 209 49 0 232 79 50 0 218 184 56 0 20 55 57 0 78 181 57 0 136 51 58 0

    // sss0: 32 0 0 0 32 0 0 0 32 0 0 0 32 0 0 0 32 0 0 0 32 0 0 0 32 0 0 0 32 0 0 0
    sss0 = _mm256_add_epi32(sss0, tmp);
    print_m256i(sss0, "sss0");
    // sss0 = sss0 + tmp
    // print_m256i sss0: 90 213 48 0 148 83 49 0 206 209 49 0 8 80 50 0 250 184 56 0 52 55 57 0 110 181 57 0 168 51 58 0

    std::cout << " " << std::endl;

    pix = _mm256_unpackhi_epi8(source, _mm256_setzero_si256());
    print_m256i(pix, "pix");
    // print_m256i pix: 5 0 104 0 6 0 105 0 7 0 106 0 8 0 107 0 21 0 120 0 22 0 121 0 23 0 122 0 24 0 123 0

    tmp = _mm256_madd_epi16(pix, mmk);
    print_m256i(tmp, "tmp");
    // print_m256i tmp: 34 206 50 0 92 76 51 0 150 202 51 0 208 72 52 0 194 177 58 0 252 47 59 0 54 174 59 0 112 44 60 0
    //   pix: (5 0 104 0) == 5, 104
    //   mmk: (58 1 0 125) == 314, 32000
    //   tmp: 5 * 314 + 104 * 32000 = 3329570 <-> (34 206 50 0) <--> 34 + 206 * 256 + 50 * 256 * 256

    sss1 = _mm256_add_epi32(sss1, tmp);
    print_m256i(sss1, "sss1");
    // print_m256i sss1: 66 206 50 0 124 76 51 0 182 202 51 0 240 72 52 0 226 177 58 0 28 48 59 0 86 174 59 0 144 44 60 0
    // sss1 = sss1 + tmp

    std::cout << " " << std::endl;

    source = _mm256_unpackhi_epi8(source1, source2);
    print_m256i(source, "source");
    // print_m256i source: 9 108 10 109 11 110 12 111 13 112 14 113 15 114 16 115 25 124 26 125 27 126 28 127 29 128 30 129 31 130 32 131

    std::cout << " " << std::endl;

    pix = _mm256_unpacklo_epi8(source, _mm256_setzero_si256());
    print_m256i(pix, "pix");
    // print_m256i pix: 9 0 108 0 10 0 109 0 11 0 110 0 12 0 111 0 25 0 124 0 26 0 125 0 27 0 126 0 28 0 127 0

    tmp = _mm256_madd_epi16(pix, mmk);
    print_m256i(tmp, "tmp");
    // print_m256i tmp: 10 199 52 0 68 69 53 0 126 195 53 0 184 65 54 0 170 170 60 0 228 40 61 0 30 167 61 0 88 37 62 0
    //   pix: (9 0 108 0) == 9, 108
    //   mmk: (58 1 0 125) == 314, 32000
    //   tmp: 9 * 314 + 108 * 32000 = 3458826 <-> (10 199 52 0) <--> 10 + 199 * 256 + 52 * 256 * 256

    sss2 = _mm256_add_epi32(sss2, tmp);
    print_m256i(sss2, "sss2");
    // print_m256i sss2: 42 199 52 0 100 69 53 0 158 195 53 0 216 65 54 0 202 170 60 0 4 41 61 0 62 167 61 0 120 37 62 0
    // sss2 = sss2 + tmp

    std::cout << " " << std::endl;

    pix = _mm256_unpackhi_epi8(source, _mm256_setzero_si256());
    print_m256i(pix, "pix");
    // print_m256i pix: 13 0 112 0 14 0 113 0 15 0 114 0 16 0 115 0 29 0 128 0 30 0 129 0 31 0 130 0 32 0 131 0

    tmp = _mm256_madd_epi16(pix, mmk);
    print_m256i(tmp, "tmp");
    // print_m256i tmp: 242 191 54 0 44 62 55 0 102 188 55 0 160 58 56 0 146 163 62 0 204 33 63 0 6 160 63 0 64 30 64 0
    //   pix: (13 0 112 0) == 13, 112
    //   mmk: (58 1 0 125) == 314, 32000
    //   tmp: 13 * 314 + 112 * 32000 = 3588082 <-> (242 191 54 0) <--> 242 + 191 * 256 + 54 * 256 * 256

    sss3 = _mm256_add_epi32(sss3, tmp);
    print_m256i(sss3, "sss3");
    // print_m256i sss3: 18 192 54 0 76 62 55 0 134 188 55 0 192 58 56 0 178 163 62 0 236 33 63 0 38 160 63 0 96 30 64 0
    // sss3 = sss3 + tmp

    // ....
    std::cout << " " << std::endl;

    // Shift packed 32-bit integers in a right by imm8 while shifting in sign bits, and store the results in dst.
    // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#ig_expand=6387,6375,6378,6378,6389,6389,6378,6389,5234,6389,4374,6389,6378,6442,6378,6389,6433,5264,5239,5234,6389,6235,5381,5312,5372,5366,6688&text=_mm256_srai_epi32
    print_m256i(sss0, "before sss0");
    sss0 = _mm256_srai_epi32(sss0, coefs_precision);
    print_m256i(sss0, "after sss0");
    // coefs_precision = 6
    // print_m256i before sss0: 90 213 48 0 148 83 49 0 206 209 49 0 8 80 50 0 250 184 56 0 52 55 57 0 110 181 57 0 168 51 58 0
    // print_m256i after sss0: 85 195 0 0 78 197 0 0 71 199 0 0 64 201 0 0 227 226 0 0 220 228 0 0 213 230 0 0 206 232 0 0
    //

    print_m256i(sss1, "before sss1");
    sss1 = _mm256_srai_epi32(sss1, coefs_precision);
    print_m256i(sss1, "after sss1");
    //

    print_m256i(sss2, "before sss2");
    sss2 = _mm256_srai_epi32(sss2, coefs_precision);
    print_m256i(sss2, "after sss2");
    //

    print_m256i(sss3, "before sss3");
    sss3 = _mm256_srai_epi32(sss3, coefs_precision);
    print_m256i(sss3, "after sss3");
    //

    std::cout << " " << std::endl;

    sss0 = _mm256_packs_epi32(sss0, sss1);
    print_m256i(sss0, "sss0");
    //
    sss2 = _mm256_packs_epi32(sss2, sss3);
    print_m256i(sss2, "sss2");
    // Convert packed signed 16-bit integers from a and b to packed 8-bit integers using unsigned saturation, and store the results in dst.
    // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#ig_expand=5381,5381,6389,2316,2310,6688,5148,5178,5148,2310,3041,7415,7358,5178&text=_mm256_packus_epi16
    sss0 = _mm256_packus_epi16(sss0, sss2);
    print_m256i(sss0, "sss0");

    std::cout << " " << std::endl;

    unsigned char output[32] = {};
    _mm256_storeu_si256((__m256i *) output, sss0);

    std::cout << "output: ";
    for (int i=0; i<32; i++) {
        std::cout << (int) output[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
#endif


#if 1
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// #undef INT32
// #define INT32 int

// #undef INT16
// #define INT16 short

// #undef UINT32
// #define UINT32 unsigned int


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


int main(int argc, char** argv)
{
    // Vertical pass: ImagingResampleVerticalConvolution8u

    UINT32 *lineIn;
    UINT32 *lineOut;

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

    unsigned char data[4 * 20 * 4] = {
        0, 1, 2, 255, 3, 4, 5, 255, 6, 7, 8, 255, 9, 10, 11, 255, 12, 13, 14, 255, 15, 16, 17, 255, 18, 19, 20, 255, 21, 22, 23, 255, 24, 25, 26, 255, 27, 28, 29, 255, 30, 31, 32, 255, 33, 34, 35, 255, 36, 37, 38, 255, 39, 40, 41, 255, 42, 43, 44, 255, 45, 46, 47, 255, 48, 49, 50, 255, 51, 52, 53, 255, 54, 55, 56, 255, 57, 58, 59, 255,
        60, 61, 62, 255, 63, 64, 65, 255, 66, 67, 68, 255, 69, 70, 71, 255, 72, 73, 74, 255, 75, 76, 77, 255, 78, 79, 80, 255, 81, 82, 83, 255, 84, 85, 86, 255, 87, 88, 89, 255, 90, 91, 92, 255, 93, 94, 95, 255, 96, 97, 98, 255, 99, 100, 101, 255, 102, 103, 104, 255, 105, 106, 107, 255, 108, 109, 110, 255, 111, 112, 113, 255, 114, 115, 116, 255, 117, 118, 119, 255,
        120, 121, 122, 255, 123, 124, 125, 255, 126, 127, 128, 255, 129, 130, 131, 255, 132, 133, 134, 255, 135, 136, 137, 255, 138, 139, 140, 255, 141, 142, 143, 255, 144, 145, 146, 255, 147, 148, 149, 255, 150, 151, 152, 255, 153, 154, 155, 255, 156, 157, 158, 255, 159, 160, 161, 255, 162, 163, 164, 255, 165, 166, 167, 255, 168, 169, 170, 255, 171, 172, 173, 255, 174, 175, 176, 255, 177, 178, 179, 255,
        180, 181, 182, 255, 183, 184, 185, 255, 186, 187, 188, 255, 189, 190, 191, 255, 192, 193, 194, 255, 195, 196, 197, 255, 198, 199, 200, 255, 201, 202, 203, 255, 204, 205, 206, 255, 207, 208, 209, 255, 210, 211, 212, 255, 213, 214, 215, 255, 216, 217, 218, 255, 219, 220, 221, 255, 222, 223, 224, 255, 225, 226, 227, 255, 228, 229, 230, 255, 231, 232, 233, 255, 234, 235, 236, 255, 237, 238, 239, 255
    };

    unsigned char output[2 * 20 * 4] = {};

    lineIn = (UINT32 *) &data[0];
    lineOut = (UINT32 *) &output[0];

    print_data(data, 1 * 9 * 4, "input");

    // (20, 4)
    // coefs_precision=16, ksize=5
    // x=0, xsize=20, xmin=0, xmax=3
    // k: 28087 28087 9362
    // x=0, xsize=20, xmin=1, xmax=3
    // k: 9362 28087 28087
    // (20, 2)

    int xsize = 20;
    int xin = xsize;
    INT16 kk[5 * 4] = {
        28087, 28087, 9362, 0, 0,
        9362, 28087, 28087, 0, 0,
    };
    int ksize = 5;
    int kmax = ksize;

    int coefs_precision = 16;
    int xmin=0, xmax=3, xx=0, x;

    // xsize = output width, xx = output x index
    // xmax = interpolation size, x = interpolation index (vertical <-> y dimension)
    // xmin = input y start index

    INT16 *k = &kk[0 * ksize];
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

    __m128i initial = _mm_set1_epi32(1 << (coefs_precision - 1));
    __m256i initial_256 = _mm256_set1_epi32(1 << (coefs_precision - 1));

    for (; xx < xsize - 7; xx += 8) {
        __m256i sss0 = initial_256;
        __m256i sss1 = initial_256;
        __m256i sss2 = initial_256;
        __m256i sss3 = initial_256;
        x = 0;
        for (; x < xmax - 1; x += 2) {
            __m256i source, source1, source2;
            __m256i pix, mmk;

            // Load two coefficients at once
            mmk = _mm256_set1_epi32(*(int32_t*)&k[x]);

            // Load 2 lines
            //                           (__m256i *) &lineIn->image32[x + xmin][xx]
            source1 = _mm256_loadu_si256((__m256i*)(lineIn + (x + xmin) * xin + xx));
            //                           (__m256i *) &lineIn->image32[x + 1 + xmin][xx]
            source2 =
                _mm256_loadu_si256((__m256i*)(lineIn + (x + 1 + xmin) * xin + xx));

            source = _mm256_unpacklo_epi8(source1, source2);
            pix = _mm256_unpacklo_epi8(source, _mm256_setzero_si256());
            sss0 = _mm256_add_epi32(sss0, _mm256_madd_epi16(pix, mmk));
            pix = _mm256_unpackhi_epi8(source, _mm256_setzero_si256());
            sss1 = _mm256_add_epi32(sss1, _mm256_madd_epi16(pix, mmk));

            source = _mm256_unpackhi_epi8(source1, source2);
            pix = _mm256_unpacklo_epi8(source, _mm256_setzero_si256());
            sss2 = _mm256_add_epi32(sss2, _mm256_madd_epi16(pix, mmk));
            pix = _mm256_unpackhi_epi8(source, _mm256_setzero_si256());
            sss3 = _mm256_add_epi32(sss3, _mm256_madd_epi16(pix, mmk));
        }
        for (; x < xmax; x += 1) {
            __m256i source, source1, pix, mmk;
            mmk = _mm256_set1_epi32(k[x]);

            //                           (__m256i *) &lineIn->image32[x + xmin][xx])
            source1 = _mm256_loadu_si256((__m256i*)(lineIn + (x + xmin) * xin + xx));

            source = _mm256_unpacklo_epi8(source1, _mm256_setzero_si256());
            pix = _mm256_unpacklo_epi8(source, _mm256_setzero_si256());
            sss0 = _mm256_add_epi32(sss0, _mm256_madd_epi16(pix, mmk));
            pix = _mm256_unpackhi_epi8(source, _mm256_setzero_si256());
            sss1 = _mm256_add_epi32(sss1, _mm256_madd_epi16(pix, mmk));

            source = _mm256_unpackhi_epi8(source1, _mm256_setzero_si256());
            pix = _mm256_unpacklo_epi8(source, _mm256_setzero_si256());
            sss2 = _mm256_add_epi32(sss2, _mm256_madd_epi16(pix, mmk));
            pix = _mm256_unpackhi_epi8(source, _mm256_setzero_si256());
            sss3 = _mm256_add_epi32(sss3, _mm256_madd_epi16(pix, mmk));
        }
        sss0 = _mm256_srai_epi32(sss0, coefs_precision);
        sss1 = _mm256_srai_epi32(sss1, coefs_precision);
        sss2 = _mm256_srai_epi32(sss2, coefs_precision);
        sss3 = _mm256_srai_epi32(sss3, coefs_precision);

        sss0 = _mm256_packs_epi32(sss0, sss1);
        sss2 = _mm256_packs_epi32(sss2, sss3);
        sss0 = _mm256_packus_epi16(sss0, sss2);
        _mm256_storeu_si256((__m256i*)&lineOut[xx], sss0);
    }

    for (; xx < xsize - 1; xx += 2) {
        __m128i sss0 = initial; // left row
        __m128i sss1 = initial; // right row
        x = 0;
        for (; x < xmax - 1; x += 2) {
            __m128i source, source1, source2;
            __m128i pix, mmk;

            // Load two coefficients at once
            mmk = _mm_set1_epi32(*(int32_t*)&k[x]);

            // Load 2 lines
            //                        (__m128i *) &lineIn->image32[x + xmin][xx])
            source1 = _mm_loadl_epi64((__m128i*)(lineIn + (x + xmin) * xin + xx));
            //                        (__m128i *) &lineIn->image32[x + 1 + xmin][xx]
            source2 = _mm_loadl_epi64((__m128i*)(lineIn + (x + 1 + xmin) * xin + xx));

            source = _mm_unpacklo_epi8(source1, source2);
            pix = _mm_unpacklo_epi8(source, _mm_setzero_si128());
            sss0 = _mm_add_epi32(sss0, _mm_madd_epi16(pix, mmk));
            pix = _mm_unpackhi_epi8(source, _mm_setzero_si128());
            sss1 = _mm_add_epi32(sss1, _mm_madd_epi16(pix, mmk));
        }
        for (; x < xmax; x += 1) {
            __m128i source, source1, pix, mmk;
            mmk = _mm_set1_epi32(k[x]);

            //                        (__m128i *) &lineIn->image32[x + xmin][xx]);
            source1 = _mm_loadl_epi64((__m128i*)(lineIn + (x + xmin) * xin + xx));

            source = _mm_unpacklo_epi8(source1, _mm_setzero_si128());
            pix = _mm_unpacklo_epi8(source, _mm_setzero_si128());
            sss0 = _mm_add_epi32(sss0, _mm_madd_epi16(pix, mmk));
            pix = _mm_unpackhi_epi8(source, _mm_setzero_si128());
            sss1 = _mm_add_epi32(sss1, _mm_madd_epi16(pix, mmk));
        }
        sss0 = _mm_srai_epi32(sss0, coefs_precision);
        sss1 = _mm_srai_epi32(sss1, coefs_precision);

        sss0 = _mm_packs_epi32(sss0, sss1);
        sss0 = _mm_packus_epi16(sss0, sss0);
        _mm_storel_epi64((__m128i*)&lineOut[xx], sss0);
    }

    for (; xx < xsize; xx++) {
        __m128i sss = initial;
        x = 0;
        for (; x < xmax - 1; x += 2) {
            __m128i source, source1, source2;
            __m128i pix, mmk;

            // Load two coefficients at once
            mmk = _mm_set1_epi32(*(int32_t*)&k[x]);

            // Load 2 lines
            //                           *(int *) &lineIn->image32[x + xmin][xx]
            source1 = _mm_cvtsi32_si128(*(int*)(lineIn + (x + xmin) * xin + xx));
            //                          *(int *) &lineIn->image32[x + 1 + xmin][xx]
            source2 = _mm_cvtsi32_si128(*(int*)(lineIn + (x + 1 + xmin) * xin + xx));

            source = _mm_unpacklo_epi8(source1, source2);
            pix = _mm_unpacklo_epi8(source, _mm_setzero_si128());
            sss = _mm_add_epi32(sss, _mm_madd_epi16(pix, mmk));
        }

        for (; x < xmax; x++) {
            //                             &lineIn->image32[x + xmin][xx]
            __m128i pix = mm_cvtepu8_epi32(lineIn + (x + xmin) * xin + xx);
            __m128i mmk = _mm_set1_epi32(k[x]);
            sss = _mm_add_epi32(sss, _mm_madd_epi16(pix, mmk));
        }
        sss = _mm_srai_epi32(sss, coefs_precision);
        sss = _mm_packs_epi32(sss, sss);
        lineOut[xx] = _mm_cvtsi128_si32(_mm_packus_epi16(sss, sss));
    }

    print_data(output, 20 * 1 * 4, "output");
    // print_data output: 43 44 45 255 46 47 48 255 49 50 51 255 52 53 54 255 55 56 57 255 58 59 60 255 61 62 63 255 64 65 66 255 67 68 69 255 70 71 72 255 73 74 75 255 76 77 78 255 79 80 81 255 82 83 84 255 85 86 87 255 88 89 90 255 91 92 93 255 94 95 96 255 97 98 99 255 100 101 102 255

    return 0;
}

#endif


#if 0
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// #undef INT32
// #define INT32 int

// #undef INT16
// #define INT16 short

// #undef UINT32
// #define UINT32 unsigned int


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


int main(int argc, char** argv)
{

    // What does ImagingResampleHorizontalConvolution8u4x exactly ?
    // https://github.com/uploadcare/pillow-simd/blob/668aa48d12305b8f093958792a5e4f690c2583d6/src/libImaging/ResampleSIMDHorizontalConv.c

    UINT32 *lineIn0, *lineIn1, *lineIn2, *lineIn3;
    UINT32 *lineOut0, *lineOut1, *lineOut2, *lineOut3;

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

    unsigned char output[4 * 2 * 4] = {
        0, 0, 0, 0, 0, 0, 0, 0,  // line 0
        0, 0, 0, 0, 0, 0, 0, 0,  // line 1
        0, 0, 0, 0, 0, 0, 0, 0,  // line 2
        0, 0, 0, 0, 0, 0, 0, 0,  // line 3
    };

    lineIn0 = (UINT32 *) &data[0 * 9 * 4];
    lineIn1 = (UINT32 *) &data[1 * 9 * 4];
    lineIn2 = (UINT32 *) &data[2 * 9 * 4];
    lineIn3 = (UINT32 *) &data[3 * 9 * 4];

    lineOut0 = (UINT32 *) &output[0 * 2 * 4];
    lineOut1 = (UINT32 *) &output[1 * 2 * 4];
    lineOut2 = (UINT32 *) &output[2 * 2 * 4];
    lineOut3 = (UINT32 *) &output[3 * 2 * 4];

    print_uint32(lineIn0[0], "lineIn0");
    print_uint32(lineIn1[0], "lineIn1");
    print_uint32(lineIn2[0], "lineIn2");
    print_uint32(lineIn3[0], "lineIn3");

    int xsize = 2;
    int xbounds[2 * 2] = {0, 7, 2, 7};
    INT16 kk[9 * 2] = {
        11703, 16384, 16384, 11703, 7022, 2341, 1235, 0, 0,
        2341, 7022, 11703, 16384, 16384, 11703, 1235, 0, 0
    };
    int kmax = 9;
    int coefs_precision = 16;

    int xmin, xmax, xx, x;
    INT16 *k;

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

    // xsize = output width, xx = output x index
    // xmax = interpolation size, x = interpolation index (horizontal <-> x dimension)
    // xmin = input x start index corresponding to output x index (xx)

    for (xx = 0; xx < xsize; xx++) {
        xmin = xbounds[xx * 2 + 0];
        xmax = xbounds[xx * 2 + 1];
        k = &kk[xx * kmax];
        x = 0;

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

        for (; x < xmax - 3; x += 4) {
            std::cout << "- block 4, x: " << x << std::endl;
            __m256i pix, mmk0, mmk1, source;

            mmk0 = _mm256_set1_epi32(*(INT32 *) &k[x]);
            print_m256i(mmk0, "mmk0");
            mmk1 = _mm256_set1_epi32(*(INT32 *) &k[x + 2]);
            print_m256i(mmk1, "mmk1");

            source = _mm256_inserti128_si256(_mm256_castsi128_si256(
                _mm_loadu_si128((__m128i *) &lineIn0[x + xmin])),
                _mm_loadu_si128((__m128i *) &lineIn1[x + xmin]), 1);
            print_m256i(source, "source");

            pix = _mm256_shuffle_epi8(source, _mm256_set_epi8(
                -1,7, -1,3, -1,6, -1,2, -1,5, -1,1, -1,4, -1,0,
                -1,7, -1,3, -1,6, -1,2, -1,5, -1,1, -1,4, -1,0));
            print_m256i(pix, "pix");
            sss0 = _mm256_add_epi32(sss0, _mm256_madd_epi16(pix, mmk0));
            print_m256i(sss0, "sss0");

            pix = _mm256_shuffle_epi8(source, _mm256_set_epi8(
                -1,15, -1,11, -1,14, -1,10, -1,13, -1,9, -1,12, -1,8,
                -1,15, -1,11, -1,14, -1,10, -1,13, -1,9, -1,12, -1,8));
            print_m256i(pix, "pix");
            sss0 = _mm256_add_epi32(sss0, _mm256_madd_epi16(pix, mmk1));
            print_m256i(sss0, "sss0");

            source = _mm256_inserti128_si256(_mm256_castsi128_si256(
                _mm_loadu_si128((__m128i *) &lineIn2[x + xmin])),
                _mm_loadu_si128((__m128i *) &lineIn3[x + xmin]), 1);
            print_m256i(source, "source");
            pix = _mm256_shuffle_epi8(source, _mm256_set_epi8(
                -1,7, -1,3, -1,6, -1,2, -1,5, -1,1, -1,4, -1,0,
                -1,7, -1,3, -1,6, -1,2, -1,5, -1,1, -1,4, -1,0));
            print_m256i(pix, "pix");
            sss1 = _mm256_add_epi32(sss1, _mm256_madd_epi16(pix, mmk0));
            print_m256i(sss1, "sss1");
            pix = _mm256_shuffle_epi8(source, _mm256_set_epi8(
                -1,15, -1,11, -1,14, -1,10, -1,13, -1,9, -1,12, -1,8,
                -1,15, -1,11, -1,14, -1,10, -1,13, -1,9, -1,12, -1,8));
            print_m256i(pix, "pix");
            sss1 = _mm256_add_epi32(sss1, _mm256_madd_epi16(pix, mmk1));
            print_m256i(sss1, "sss1");
        }

        for (; x < xmax - 1; x += 2) {
            std::cout << "- block 2, x: " << x << std::endl;
            __m256i pix, mmk;

            // - Load int16 weights taking 2 values
            mmk = _mm256_set1_epi32(*(INT32 *) &k[x]);
            print_m256i(mmk, "mmk");
            // print_m256i mmk: 183 45 0 64 183 45 0 64 183 45 0 64 183 45 0 64 183 45 0 64 183 45 0 64 183 45 0 64 183 45 0 64
            // (183 45 0 64) <-> 183 + 256 * 45 = 11703, 64 * 256 = 16384
            // mmk ~ {kk[0], kk[1], kk[0], kk[1], ... kk[0], kk[1]}

            auto a0 = _mm_loadl_epi64((__m128i *) &lineIn0[x + xmin]);
            print_m128i(a0, "a0");
            // print_m128i a0: 0 1 2 255 3 4 5 255 0 0 0 0 0 0 0 0
            auto b0 = _mm256_castsi128_si256(a0);
            print_m256i(b0, "b0");
            // print_m256i b0: 0 1 2 255 3 4 5 255 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            auto a1 = _mm_loadl_epi64((__m128i *) &lineIn1[x + xmin]);
            print_m128i(a1, "a1");
            // print_m128i a1: 24 25 26 255 27 28 29 255 0 0 0 0 0 0 0 0
            pix = _mm256_inserti128_si256(b0, a1, 1);
            print_m256i(pix, "1 pix");
            // print_m256i 1 pix: 0 1 2 255 3 4 5 255 0 0 0 0 0 0 0 0 24 25 26 255 27 28 29 255 0 0 0 0 0 0 0 0
            auto m0 = _mm256_set_epi8(
                -1,7, -1,3, -1,6, -1,2, -1,5, -1,1, -1,4, -1,0,
                -1,7, -1,3, -1,6, -1,2, -1,5, -1,1, -1,4, -1,0);
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
                _mm_loadl_epi64((__m128i *) &lineIn2[x + xmin])),
                _mm_loadl_epi64((__m128i *) &lineIn3[x + xmin]), 1);
            pix = _mm256_shuffle_epi8(pix, _mm256_set_epi8(
                -1,7, -1,3, -1,6, -1,2, -1,5, -1,1, -1,4, -1,0,
                -1,7, -1,3, -1,6, -1,2, -1,5, -1,1, -1,4, -1,0));
            sss1 = _mm256_add_epi32(sss1, _mm256_madd_epi16(pix, mmk));
            print_m256i(sss1, "sss1");
        }

        for (; x < xmax; x ++) {
            std::cout << "- block 1, x: " << x << std::endl;
            __m256i pix, mmk;

            mmk = _mm256_set1_epi32(k[x]);
            // 123 34 0 0
            print_m256i(mmk, "mmk");
            // print_m256i mmk: 123 34 0 0 123 34 0 0 123 34 0 0 123 34 0 0 123 34 0 0 123 34 0 0 123 34 0 0

            auto a0 = mm_cvtepu8_epi32(&lineIn0[x + xmin]);
            print_m128i(a0, "a0");
            // print_m128i a0: 0 0 0 0 1 0 0 0 2 0 0 0 255 0 0 0
            auto b0 = _mm256_castsi128_si256(a0);
            print_m256i(b0, "b0");
            // print_m256i b0: 0 0 0 0 1 0 0 0 2 0 0 0 255 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            auto a1 = mm_cvtepu8_epi32(&lineIn1[x + xmin]);
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

            pix = _mm256_inserti128_si256(_mm256_castsi128_si256(
                mm_cvtepu8_epi32(&lineIn2[x + xmin])),
                mm_cvtepu8_epi32(&lineIn3[x + xmin]), 1);
            print_m256i(pix, "pix");
            auto tmp1 = _mm256_madd_epi16(pix, mmk);
            print_m256i(tmp1, "tmp1");
            sss1 = _mm256_add_epi32(sss1, tmp1);
            print_m256i(sss1, "sss1");
            // sss1 = initial + pix[0] * mmk[0] + pix[1] * mmk[1] + ...
            // for lines 2 and 3
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

        // _mm_cvtsi128_si32: Copy the lower 32-bit integer in a to dst.
        // _mm256_extracti128_si256: Extract 128 bits (composed of integer data) from a, selected with imm8, and store the result in dst.
        auto e0 = _mm256_extracti128_si256(sss0, 0);
        print_m128i(e0, "e0");
        // print_m128i e0: 15 16 17 255 0 0 0 0 0 0 0 0 0 0 0 0
        lineOut0[xx] = _mm_cvtsi128_si32(e0);
        auto e1 = _mm256_extracti128_si256(sss0, 1);
        print_m128i(e1, "e1");
        // print_m128i e1: 39 40 41 255 0 0 0 0 0 0 0 0 0 0 0 0

        lineOut1[xx] = _mm_cvtsi128_si32(e1);
        auto e2 = _mm256_extracti128_si256(sss1, 0);
        print_m128i(e2, "e2");
        // print_m128i e2: 63 64 65 255 0 0 0 0 0 0 0 0 0 0 0 0

        lineOut2[xx] = _mm_cvtsi128_si32(e2);
        auto e3 = _mm256_extracti128_si256(sss1, 1);
        print_m128i(e3, "e3");
        // print_m128i e3: 87 88 89 255 0 0 0 0 0 0 0 0 0 0 0 0
        lineOut3[xx] = _mm_cvtsi128_si32(e3);


        print_uint32(lineOut0[xx], "lineOut0");
        print_uint32(lineOut1[xx], "lineOut1");
        print_uint32(lineOut2[xx], "lineOut2");
        print_uint32(lineOut3[xx], "lineOut3");
    }

    print_data(output, 4 * 2 * 4, "output");
    // print_data output: 6 7 8 255 16 17 18 255 34 35 36 255 43 44 45 255 61 62 63 255 71 72 73 255 89 90 91 255 98 99 100 255

    return 0;
}

#endif
