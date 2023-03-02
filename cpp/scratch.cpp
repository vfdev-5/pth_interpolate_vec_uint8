
#include <iostream>
#include <ATen/ATen.h>



int main(int argc, char ** argv) {

    at::manual_seed(1);

    auto p = at::CPU(at::kFloat);
    auto t = at::rand({1, 3, 32, 32}, p);

    std::cout << t.numel() << " vs " << 1 * 3 * 32 * 32 << std::endl;

    // std::cout << t.index({0, 0, 0, 0}) << std::endl;

    return 0;
}