#include <iostream>
#include <chrono>

// Torch
#include <ATen/ATen.h>
#include <ATen/Parallel.h>


int main(int argc, char** argv) {

    int n = 1000;
    int m = 7;
    int num_threads = 1;

    at::manual_seed(10);
    at::set_num_threads(num_threads);

    std::cout << "Torch config: " << at::show_config() << std::endl;
    std::cout << "Num threads: " << at::get_num_threads() << std::endl;

    auto size = 256;
    auto t_input = at::randint(0, 256, {1, 4, size, size}, at::CPU(at::kByte)).contiguous(at::MemoryFormat::ChannelsLast);

    std::cout << "\nInput tensor: " << t_input.sizes() << std::endl;
    std::cout << "Input is_contiguous memory_format torch.channels_last: "
              << (t_input.is_contiguous(at::MemoryFormat::ChannelsLast) ? "true" : "false") << std::endl;
    std::cout << "Input is_contiguous : "
              << (t_input.is_contiguous() ? "true" : "false") << std::endl;

    int64_t osizes[2] = {
        ((argc >= 2) ? std::atoi(argv[1]) : 224),
        ((argc >= 3) ? std::atoi(argv[2]) : 224)
    };
    at::IntArrayRef output_size(osizes);

    std::cout << "\n- Bench _upsample_bilinear2d_aa (" << n << " rounds) - downsampling 256 -> (" << osizes[0] << ", " << osizes[1] << ")" << std::endl;
    // warm-up
    for (int i=0; i<int(n / 10); i++) {
        auto out = at::native::_upsample_bilinear2d_aa(t_input, output_size, false, c10::nullopt);
    }
    // measure
    double avg_output = 0.0;
    for (int j=0; j < m; j++) {
        auto start = std::chrono::steady_clock::now();
        for (int i=0; i<n; i++) {
            auto out = at::native::_upsample_bilinear2d_aa(t_input, output_size, false, c10::nullopt);
        }
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        avg_output += elapsed_seconds.count() / n;
    }
    avg_output /= m;

    std::cout << "Elapsed time (us): " << avg_output * 1e6 << std::endl;

    return 0;
}