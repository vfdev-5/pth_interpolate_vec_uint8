// With ASAN
//
// export LD_PRELOAD=/usr/lib/llvm-7/lib/clang/7.0.1/lib/linux/libclang_rt.asan-x86_64.so && ./check_alignment 3
//
// unset LD_PRELOAD && make
// CC=clang CXX=clang++ LDSHARED='clang --shared' cmake .. -DCMAKE_PREFIX_PATH="/pytorch/torch/"
//
// CMakeLists.txt
//
// cmake_minimum_required(VERSION 3.10)
// project(check)

// # As ~PyTorch
// # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mavx512f -mavx512dq -mavx512bw -mavx512vl -mavx -mfma -mavx2 -mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store -Wno-deprecated -fvisibility-inlines-hidden -fopenmp -DNDEBUG -O3 -fPIC -Wno-narrowing -Wall -Wextra -Werror=return-type -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-sign-compare -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function -Wno-unused-result -Wno-unused-local-typedefs -Wno-strict-overflow -Wno-strict-aliasing -Wno-error=deprecated-declarations -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=redundant-decls -Wno-error=old-style-cast -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Werror=cast-function-type -Wno-stringop-overflow")
// set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx -mfma -mavx2 -fsanitize=address -fsanitize=undefined -fno-sanitize-recover=all -fsanitize-address-use-after-scope -shared-libasan -g")

// find_package(Torch REQUIRED)

// add_executable(check_alignment check_alignment.cpp)
// target_link_libraries(check_alignment PUBLIC ${TORCH_LIBRARIES})
// set_property(TARGET check_alignment PROPERTY CXX_STANDARD 17)


#include <iostream>
#include <vector>
#include <ATen/ATen.h>
#include <stdlib.h>

// AVX
#include <immintrin.h>


using namespace c10;
using namespace at;
using namespace at::native;
using namespace at::indexing;

int main(int argc, char** argv) {

    std::vector<int> data;
    for (int i=1; i<64; i++) {
        data.push_back(i);
    }

    int16_t* kk = (int16_t*)(data.data());
    auto xx = 1;
    auto kmax = (argc > 1) ? atoi(argv[1]) : 27;
    std::cout << "kmax: " << kmax << std::endl;
    auto x = 0;
    int16_t* k;
    k = &kk[xx * kmax];

    int v1 = k[x];
    int v2 = k[x + 1];
    auto mmk0 = _mm256_set_epi16(
        v1, v2, v1, v2, v1, v2, v1, v2, v1, v2, v1, v2, v1, v2, v1, v2);

    // int32_t v = *(int32_t*)&k[x];  // <--- requires int32 mem alignment
    // auto mmk0 = _mm256_set1_epi32(v);

    std::cout << "--" << std::endl;

    // auto xsize = 32;
    // auto kmax = 27;

    // auto p = at::CPU(at::kDouble);
    // auto tensor = at::ones({xsize * kmax}, p);
    // std::cout << tensor.sizes() << std::endl;

    // auto xx = 1;
    // auto x = 0;

    // int16_t* kk = (int16_t*)(tensor.data_ptr<double>());

    // int16_t* k;
    // k = &kk[xx * kmax];

    // auto mmk0 = _mm256_set1_epi32(*(int32_t*)&k[x]);  // <--- requires int32 mem alignment

    // // auto tmp = _mm_loadu_si128((__m128i*)&k[x]);  // <--- does not require mem alignment

    // // __m128i tmp = _mm_loadl_epi64((__m128i*)&k[x]);  // <--- does not require mem alignment

    return 0;
}

// 256 bits = 32 bytes
// 128 bits = 16 bytes
// int32 = 4 bytes
// int16 = 2 bytes
