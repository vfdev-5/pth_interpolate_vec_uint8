#include <iostream>
#include <ATen/ATen.h>

// AVX
#include <immintrin.h>


using namespace c10;
using namespace at;
using namespace at::native;
using namespace at::indexing;

int main() {

    std::cout << "--" << std::endl;

    auto xsize = 32;
    auto kmax = 27;

    auto p = at::CPU(at::kDouble);
    auto tensor = at::ones({xsize * kmax}, p);
    std::cout << tensor.sizes() << std::endl;

    auto xx = 1;
    auto x = 0;

    int16_t* kk = (int16_t*)(tensor.data_ptr<double>());

    int16_t* k;
    k = &kk[xx * kmax];

    auto mmk0 = _mm256_set1_epi32(*(int32_t*)&k[x]);

    return 0;
}
