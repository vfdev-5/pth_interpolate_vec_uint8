#include <iostream>

#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>



int main() {

  auto packed_tensor = at::arange(2 * 2 * 5 * 6, at::CPU(at::kByte)).reshape({2, 2, 5, 6});
  packed_tensor = packed_tensor[0];

  auto width = packed_tensor.size(2);
  auto height = packed_tensor.size(1);
  auto num_channels = packed_tensor.size(0);

  auto output_num_channels = (num_channels == 3 || num_channels == 4) ? num_channels : 4;

  auto unpacked_tensor = at::empty_strided(
      {output_num_channels, height, width}, {1, output_num_channels * width, output_num_channels}, at::CPU(at::kByte));
  unpacked_tensor.zero_();

  unpacked_tensor.index({at::indexing::Slice(0, num_channels)}).copy_(packed_tensor);

  std::cout << "-- Input: " << packed_tensor.sizes() << ", " << packed_tensor.strides()
            << " | contig: " << (packed_tensor.is_contiguous() ? "true" : "false")
            << " | contig CL-like: " << (packed_tensor.unsqueeze(0).is_contiguous(at::MemoryFormat::ChannelsLast) ? "true" : "false") << std::endl;

  std::cout << "-- Output: " << unpacked_tensor.sizes() << ", " << unpacked_tensor.strides()
            << " | contig CF: " << (unpacked_tensor.is_contiguous() ? "true" : "false")
            << " | contig CL-like: " << (unpacked_tensor.unsqueeze(0).is_contiguous(at::MemoryFormat::ChannelsLast) ? "true" : "false") << std::endl;

  std::cout << packed_tensor.sum().item<long>() << " vs " << unpacked_tensor.sum().item<long>() << std::endl;
  return 0;




  // // restride the input
  auto shape = packed_tensor.sizes().vec();
  auto strides = packed_tensor.strides().vec();
  auto oshape = unpacked_tensor.sizes();

  shape[0] = oshape[0];
  strides[0] = 0;
  auto restrided_packed_tensor = packed_tensor.as_strided(shape, strides);

  std::cout << "restrided_packed_tensor shape/strides: " << restrided_packed_tensor.sizes() << ", " << restrided_packed_tensor.strides() << std::endl;
  std::cout << "unpacked_tensor shape/strides: " << unpacked_tensor.sizes() << ", " << unpacked_tensor.strides() << std::endl;

  auto iter = at::TensorIteratorConfig()
      .add_output(unpacked_tensor)
      .add_input(restrided_packed_tensor)
      .resize_outputs(false)
      .build();

  // auto loop2d = [](char** data, const int64_t* strides, int64_t size0, int64_t size1) {

  //   std::cout << "TI_SHOW: size0=" << size0 << std::endl;
  //   std::cout << "TI_SHOW: size1=" << size1 << std::endl;
  //   std::cout << "TI_SHOW_STRIDES: "
  //     << strides[0] << " "
  //     << strides[1] << " | ";

  // //   constexpr int m = 2;
  // //   int ndims = 1;
  // //   for (int i=0; i<ndims; i++) {
  // //     for (int j=0; j<2 * m; j++) {
  // //       std::cout << strides[2 * m * i + j + 2] << " ";
  // //     }
  // //     std::cout << "| ";
  // //   }
  //   std::cout << std::endl;
  // };

  // iter.for_each(loop2d);

  auto test_loop = [&](char **data, const int64_t* strides, int64_t n) {
    std::cout << "n : " << n << std::endl;
    std::cout << "Input stride: " << strides[1] << std::endl;
    std::cout << "Output stride: " << strides[0] << std::endl;

    auto * out_data_bytes = data[0];
    auto * in_data_bytes = data[1];

    // assume float data type for this example.
    std::cout << " - input data: " << std::endl;
    for (int i = 0; i < n; i++) {
        // *reinterpret_cast<float*>(out_data_bytes) = *reinterpret_cast<float*>(in_data_bytes);
        std::cout << (int) *reinterpret_cast<uint8_t*>(in_data_bytes) << " ";
        // out_data_bytes += strides[0];
        in_data_bytes += strides[1];
    }
    std::cout << std::endl;
  };

  iter.for_each(test_loop);


  return 0;
}