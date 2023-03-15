#include <stdlib.h>
#include <cstring>
#include <vector>

#include "benchmark/benchmark.h"


void setup_data(char * data, int size) {
    srand(1);
    for (int i=0; i<size; i++) {
        data[i] = rand() % 256;
    }

}

void copy_data_as_uint32(char * output, char * input, int size, int stride) {

    for (int i=0; i<size - 4; i+=stride) {
        uint32_t unaligned_data = *(uint32_t *)(input + i);
        *(uint32_t *)(output + i) = unaligned_data;
    }

}

void copy_data_with_memcpy_4(char * output, char * input, int size, int stride) {
    for (int i=0; i<size - 4; i+=stride) {
        memcpy(output + i, input + i, 4);
    }
}

void copy_data_with_memcpy_3(char * output, char * input, int size, int stride) {
    for (int i=0; i<size - 4; i+=stride) {
        memcpy(output + i, input + i, 3);
    }
}

static void BM_copy_data_as_uint32(benchmark::State& state) {
    const int size = state.range(0);

    std::vector<char> input_vec(size);
    std::vector<char> output_vec(size);

    char * input = input_vec.data();
    setup_data(input, size);

    char * output = output_vec.data();

    for (auto _ : state) {
        copy_data_as_uint32(output, input, size, 3);
        benchmark::ClobberMemory();
    }
}

static void BM_copy_data_with_memcpy_3(benchmark::State& state) {
    const int size = state.range(0);

    std::vector<char> input_vec(size);
    std::vector<char> output_vec(size);

    char * input = input_vec.data();
    setup_data(input, size);

    char * output = output_vec.data();

    for (auto _ : state) {
        copy_data_with_memcpy_3(output, input, size, 3);
        benchmark::ClobberMemory();
    }
}

static void BM_copy_data_with_memcpy_4(benchmark::State& state) {
    const int size = state.range(0);

    std::vector<char> input_vec(size);
    std::vector<char> output_vec(size);

    char * input = input_vec.data();
    setup_data(input, size);

    char * output = output_vec.data();

    for (auto _ : state) {
        copy_data_with_memcpy_4(output, input, size, 3);
        benchmark::ClobberMemory();
    }
}

// Register the function as a benchmark
constexpr int test_size = 10000;
BENCHMARK(BM_copy_data_as_uint32)->Arg(test_size);
BENCHMARK(BM_copy_data_with_memcpy_3)->Arg(test_size);
BENCHMARK(BM_copy_data_with_memcpy_4)->Arg(test_size);
// Run the benchmark
BENCHMARK_MAIN();