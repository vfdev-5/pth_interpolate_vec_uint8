#include <iostream>

#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>



int main() {

    auto input = at::arange(1 * 3 * 4 * 5, at::CPU(at::kFloat)).reshape({1, 3, 4, 5});
    input = input.contiguous(at::MemoryFormat::ChannelsLast);
    std::cout << "1 input mem format is CL: " << (input.is_contiguous(at::MemoryFormat::ChannelsLast) ? "true" : "false") << std::endl;

    at::Tensor output = at::zeros({1, 3, 4, 2});
    output = output.contiguous(at::MemoryFormat::ChannelsLast);

    std::cout << "2 output mem format is CL: " << (output.is_contiguous(at::MemoryFormat::ChannelsLast) ? "true" : "false") << std::endl;

    // restride the input
    auto shape = input.sizes().vec();
    auto strides = input.strides().vec();
    auto oshape = output.sizes();

    shape[3] = oshape[3];
    strides[3] = 0;
    auto restrided_input = input.as_strided(shape, strides);

    std::cout << "input shape: " << input.sizes() << std::endl;
    std::cout << "restrided_input strides: " << restrided_input.strides() << std::endl;
    std::cout << "restrided_input shape: " << restrided_input.sizes() << std::endl;
    std::cout << "output shape: " << output.sizes() << std::endl;

    auto iter = at::TensorIteratorConfig()
        .add_output(output)
        .add_input(restrided_input)
        .check_all_same_dtype(false)
        .resize_outputs(false)
        .build();

    auto loop2d = [](char** data, const int64_t* strides, int64_t size0, int64_t size1) {

      std::cout << "TI_SHOW: size0=" << size0 << std::endl;
      std::cout << "TI_SHOW: size1=" << size1 << std::endl;
      std::cout << "TI_SHOW_STRIDES: "
        << strides[0] << " "
        << strides[1] << " | ";

    //   constexpr int m = 2;
    //   int ndims = 1;
    //   for (int i=0; i<ndims; i++) {
    //     for (int j=0; j<2 * m; j++) {
    //       std::cout << strides[2 * m * i + j + 2] << " ";
    //     }
    //     std::cout << "| ";
    //   }
      std::cout << std::endl;
    };

    iter.for_each(loop2d);

    return 0;
}