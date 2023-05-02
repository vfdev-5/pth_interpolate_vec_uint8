#include <iostream>

#include <ATen/ATen.h>


int main() {

    // auto input = at::arange(1 * 3 * 4 * 5, at::CPU(at::kFloat)).reshape({1, 3, 4, 5});
    // input = input.contiguous(at::MemoryFormat::ChannelsLast);

    // std::cout << "1 input mem format is CL: " << (input.is_contiguous(at::MemoryFormat::ChannelsLast) ? "true" : "false") << std::endl;
    // std::cout << "input shape: " << input.sizes() << std::endl;
    // std::cout << "input strides: " << input.strides() << std::endl;

  auto t = at::rand({1, 3, 32, 32}).contiguous(at::MemoryFormat::ChannelsLast);
  std::cout << t.sizes() << std::endl;
  // > [1, 3, 32, 32]
  std::cout << t.strides() << std::endl;
  // > [3072, 1, 96, 3]
  std::cout << t.suggest_memory_format() << std::endl;
  // > ChannelsLast
  std::cout << (t.is_contiguous(at::MemoryFormat::ChannelsLast) ? "true" : "false") << std::endl;
  // > true


  auto t0 = t[0];

  // 3D tensor, is_channels_last_ and is_channels_last_contiguous_ are set to false
  // https://github.com/pytorch/pytorch/blob/25040893294a0656f50f545f649fc2719e5d15d4/c10/core/TensorImpl.h#L2734-L2747
  // (gdb) p is_channels_last_
  // $7 = false
  // (gdb) p is_channels_last_contiguous_
  // $8 = false

  std::cout << t0.suggest_memory_format() << std::endl;

  auto t1 = t0.unsqueeze(0);

  // 4D tensor, is_channels_last_ and is_channels_last_contiguous_ are computed
  // https://github.com/pytorch/pytorch/blob/25040893294a0656f50f545f649fc2719e5d15d4/c10/core/TensorImpl.h#L2709-L2719

  // bool _compute_contiguous(ArrayRef<T> sizes, ArrayRef<T> strides, T numel) {
  // (gdb) p {sizes[0], sizes[1], sizes[2], sizes[3]}
  // $12 = {@0x555558625360: 1, @0x555558625368: 3, @0x555558625370: 32, @0x555558625378: 32}
  // (gdb) p {strides[0], strides[1], strides[2], strides[3]}
  // $13 = {@0x555558625388: 3, @0x555558625390: 1, @0x555558625398: 96, @0x5555586253a0: 3}
  // (gdb) p is_contiguous_
  // $15 = false

  // Compute is_channels_last_contiguous_ -> true
  // bool _compute_channels_last_contiguous_2d, https://github.com/pytorch/pytorch/blob/55b661137f0c162968592bead66598fe4b091ce8/c10/core/TensorImpl.cpp#L307
  // (gdb) p {sizes[0], sizes[1], sizes[2], sizes[3]}
  // $17 = {@0x555558625360: 1, @0x555558625368: 3, @0x555558625370: 32, @0x555558625378: 32}
  // (gdb) p {strides[0], strides[1], strides[2], strides[3]}
  // $18 = {@0x555558625388: 3, @0x555558625390: 1, @0x555558625398: 96, @0x5555586253a0: 3}

  // Compute is_channels_last_ -> false
  // inline bool is_channels_last_strides_2d
  // inline bool is_channels_last_strides_2d_s4, https://github.com/pytorch/pytorch/blob/55b661137f0c162968592bead66598fe4b091ce8/c10/core/MemoryFormat.h#L128
  // (gdb) p {sizes[0], sizes[1], sizes[2], sizes[3]}
  // $24 = {@0x555558625360: 1, @0x555558625368: 3, @0x555558625370: 32, @0x555558625378: 32}
  // (gdb) p {strides[0], strides[1], strides[2], strides[3]}
  // $25 = {@0x555558625388: 3, @0x555558625390: 1, @0x555558625398: 96, @0x5555586253a0: 3}

  // Offending code: https://github.com/pytorch/pytorch/blob/55b661137f0c162968592bead66598fe4b091ce8/c10/core/MemoryFormat.h#L141-L143
  // When d=0, min = 96 * 32 and but strides[d] = 3

  auto shape = t1.sizes();
  auto strides = t1.strides().vec();

  // strides[0] = shape[1] * shape[2] * shape[3];
  // t1 = t1.as_strided(shape, strides);

  std::cout << t0.sizes() << std::endl;
  // > [3, 32, 32]
  std::cout << t0.strides() << std::endl;
  // > [1, 96, 3]
  std::cout << t1.sizes() << std::endl;
  // > [1, 3, 32, 32]
  std::cout << t1.strides() << std::endl;
  // > [3, 1, 96, 3]
  std::cout << (t1.is_contiguous(at::MemoryFormat::ChannelsLast) ? "true" : "false") << std::endl;
  // > true
  std::cout << t1.suggest_memory_format() << std::endl;
  // > Contiguous   <------ ????

  return 0;
}