# Issue with CL/CF memory format for 3D CL inputs

```
cd cpp/build

make
```

## Debug pytorch code
```
gdb --args ./check_mem_format_3d_input

(gdb) b 15
(gdb) b /pytorch/torch/include/ATen/core/TensorBase.h:257
(gdb) b /pytorch/torch/include/c10/core/TensorImpl.h:832
(gdb) b aten/src/ATen/native/TensorShape.cpp:3276

(gdb) b /pytorch/build/aten/src/ATen/Operators_3.cpp:4199
(gdb) b /pytorch/torch/csrc/autograd/generated/VariableType_3.cpp:12604
(gdb) b /pytorch/build/aten/src/ATen/RedispatchFunctions.h:6221
(gdb) b /pytorch/aten/src/ATen/native/TensorShape.cpp:1766
(gdb) b /pytorch/aten/src/ATen/native/TensorShape.cpp:1143
(gdb) b /pytorch/aten/src/ATen/native/Resize.h:153

(gdb) b /pytorch/c10/core/TensorImpl.h:1814
(gdb) b /pytorch/c10/core/TensorImpl.h:2708
```

```
return is_contiguous_default(memory_format);
```


```
(gdb) bt

#0  c10::TensorImpl::is_strides_like_default (this=0x555558620310, memory_format=c10::MemoryFormat::Preserve) at /pytorch/torch/include/c10/core/TensorImpl.h:858
#1  0x000055555555a190 in c10::TensorImpl::is_strides_like (this=0x555558620310, memory_format=c10::MemoryFormat::ChannelsLast) at /pytorch/torch/include/c10/core/TensorImpl.h:2395
#2  0x000055555555a1b5 in c10::TensorImpl::is_strides_like_channels_last (this=0x555558620310) at /pytorch/torch/include/c10/core/TensorImpl.h:2399
#3  0x000055555555a3d8 in at::TensorBase::suggest_memory_format (this=0x7fffffffdb30, channels_last_strides_exact_match=false) at /pytorch/torch/include/ATen/core/TensorBase.h:269
#4  0x0000555555557b33 in main () at /tmp/pth/fix-62396-interp-outsize/cpp/check_mem_format_3d_input.cpp:45
```