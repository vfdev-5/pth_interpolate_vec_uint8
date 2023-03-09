#include <iostream>
#include <immintrin.h>
#include <stdlib.h>


template<typename scalar_t=uint8_t>
void print_data(scalar_t * data, int size, std::string tag="") {
    std::cout << "\nprint_data " << tag << ": ";
    for (int i=0; i < size; i++) {
        std::cout << (int64_t) data[i] << " ";
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


void test_1(int index = 0) {
    uint8_t input[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

    print_data(input, 16, "input");

    auto data1 = _mm_loadl_epi64((__m128i*) (input + index));
    print_m128i(data1, "data1");

    auto data2 = _mm_loadu_si64((input + index));
    print_m128i(data2, "data2");

    // long long * data_ll = (long long *) (input + index);
    // print_data((uint8_t *) data_ll, 8, "data_ll [8]");

}


void test_2() {

    std::cout << "---" << std::endl;

    constexpr int xsize = 32;
    constexpr int kmax = 27;

    double tensor[kmax * xsize];
    for (int i=0; i < kmax * xsize; i++) {
        tensor[i] = 1.0;
    }

    auto xx = 1;
    auto x = 0;

    int16_t* kk = (int16_t*)(tensor);
    int16_t* k;
    k = &kk[xx * kmax];

    auto p = *(int32_t*)&k[x];
    std::cout << (int) p << std::endl;
}


int main(int argc, char** argv) {


    // test_2();
    test_1((argc > 1) ? atoi(argv[1]) : 0);

    return 0;
}