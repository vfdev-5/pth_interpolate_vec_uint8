## WIP on uint8 support for bicubic mode

### Description:

Main issue is that temporary buffer dtype is uint8 between Horizontal (HP) and Vertical (VP) passes.

In this case negative values produced in HP ...


### Reproduce value error

```
python -u debug_interp2_repro_bicubic.py
```



### Debug

```
gdb --args python -u debug_interp2_repro_bicubic.py

(gdb) source /pytorch/tools/gdb/pytorch-gdb.py

(gdb) b aten/src/ATen/native/cpu/UpSampleKernel.cpp:1931
(gdb) b aten/src/ATen/native/cpu/UpSampleKernel.cpp:1619
(gdb) b aten/src/ATen/native/cpu/UpSampleKernel.cpp:1532
(gdb) b aten/src/ATen/native/cpu/UpSampleKernel.cpp:1680
(gdb) b aten/src/ATen/native/cpu/UpSampleKernel.cpp:1010
(gdb) b aten/src/ATen/native/cpu/UpSampleKernel.cpp:981
(gdb) b aten/src/ATen/native/cpu/UpSampleKernel.cpp:433
(gdb) b aten/src/ATen/native/cpu/UpSampleKernel.cpp:353

(gdb) torch-tensor-repr temp_output
```

```
(gdb) p {(int32_t) t, (int32_t) wts, (int32_t) (output >> 15)}
(gdb) torch-tensor-repr temp_input

tensor([[[[ 12,  15,  17,  19],
          [ 60,  63,  65,  67],
          [108, 111, 113, 115],
          [156, 159, 161, 163],
          [204, 207, 209, 211],
          [252, 271, -15,   3],
          [ 44,  47,  49,  51],
          [ 92,  95,  97,  99]]]], dtype=torch.int32)


But should match
tensor([[[[ 12.4062,  14.5000,  16.5000,  18.5938],
          [ 60.4062,  62.5000,  64.5000,  66.5938],
          [108.4062, 110.5000, 112.5000, 114.5938],
          [156.4062, 158.5000, 160.5000, 162.5938],
          [204.4062, 206.5000, 208.5000, 210.5938],
          [252.4062, 278.5000, -23.5000,   2.5938],
          [ 44.4062,  46.5000,  48.5000,  50.5938],
          [ 92.4062,  94.5000,  96.5000,  98.5938]]]])
```

```
Compute `278.5000` value:

(gdb) p {(int32_t) t, (int32_t) wts, (int32_t) (output >> 15)}
$22 = {253, -2048, -16}
(gdb) p {(int32_t) t, (int32_t) wts, (int32_t) (output >> 15)}
$24 = {254, 18432, 127}
(gdb) p {(int32_t) t, (int32_t) wts, (int32_t) (output >> 15)}
$25 = {255, 18432, 271}
(gdb) p {(int32_t) t, (int32_t) wts, (int32_t) (output >> 15)}
$26 = {0, -2048, 271}


Compute `-23.5000` value:

(gdb) p {(int32_t) t, (int32_t) wts, (int32_t) (output >> 15)}
$28 = {255, -2048, -16}
(gdb) p {(int32_t) t, (int32_t) wts, (int32_t) (output >> 15)}
$29 = {0, 18432, -16}
(gdb) p {(int32_t) t, (int32_t) wts, (int32_t) (output >> 15)}
$30 = {1, 18432, -15}
(gdb) p {(int32_t) t, (int32_t) wts, (int32_t) (output >> 15)}
$31 = {2, -2048, -15}

Expected values:
dot([-0.09375, 0.59375, 0.59375, -0.09375], [255, 0, 1, 2]) -> -23.5

p (-0.09375 * (1 << 15)) -> -24
p (0.59375 * (1 << 15)) -> 19456
```

### xmin, xsize computation

```
Bicubic:

center = 2 * (i + 0.5)
support = 2

xmin_ = int(center - support + 0.5)
     = int(2 * i + 1 - 2 + 0.5)
     = int(2 * (i - 1) + 1.5)

xmin = max(xmin_, 0)

xsize = int(center + support + 0.5) - xmin_
      = int(2 * i + 1 + 2 + 0.5) - xmin_
      = 2 * (i + 1) + 1 - xmin_
      = 2 * (i + 1) + 1 - 2 * (i - 1) - 1
      = 4


Not AA case and xsize < max_interp_size -> set xsize = max_interp_size
j, x, w: 0 -0.5 0.59375
j, x, w: 1 0.5 0.59375
j, x, w: 2 1.5 -0.09375
j, x, w: 3 2.5 0
- Weights: xsize=4 0.542857 0.542857 -0.0857143 0
Not AA case
j, x, w: 0 -1.5 -0.09375
j, x, w: 1 -0.5 0.59375
j, x, w: 2 0.5 0.59375
j, x, w: 3 1.5 -0.09375
```

```
FLOAT32 PATH

i= 0 real_input_index= 0.5 input_index= 0 lambda= 0.5 weights= 0, -0.09375 | 0, 0.59375 | 1, 0.59375 | 2, -0.09375 |
i= 1 real_input_index= 2.5 input_index= 2 lambda= 0.5 weights= 1, -0.09375 | 2, 0.59375 | 3, 0.59375 | 4, -0.09375 |
i= 2 real_input_index= 4.5 input_index= 4 lambda= 0.5 weights= 3, -0.09375 | 4, 0.59375 | 5, 0.59375 | 6, -0.09375 |
i= 3 real_input_index= 6.5 input_index= 6 lambda= 0.5 weights= 5, -0.09375 | 6, 0.59375 | 7, 0.59375 | 7, -0.09375 |
i= 0 real_input_index= 0.5 input_index= 0 lambda= 0.5 weights= 0, -0.09375 | 0, 0.59375 | 1, 0.59375 | 2, -0.09375 |
i= 1 real_input_index= 2.5 input_index= 2 lambda= 0.5 weights= 1, -0.09375 | 2, 0.59375 | 3, 0.59375 | 4, -0.09375 |
i= 2 real_input_index= 4.5 input_index= 4 lambda= 0.5 weights= 3, -0.09375 | 4, 0.59375 | 5, 0.59375 | 6, -0.09375 |
i= 3 real_input_index= 6.5 input_index= 6 lambda= 0.5 weights= 5, -0.09375 | 6, 0.59375 | 7, 0.59375 | 7, -0.09375 |
```


```
center = 2 * (i + 0.5)
support = 2
xmin_unbounded = int(center - support + 0.5)


i = 0
xmin_unbounded = int(2 * (i + 0.5) - support + 0.5)
               = int(2 * i + 1 - 2 + 0.5)
               = int(2 * (i - 1) + 1.5)

xmin = max(xmin_unbounded, 0) -> max(-1, 0) = 0
center = 2 * (i + 0.5) -> 1
x = (j + xmin - center + 0.5 - align_corners_delta)
  = j + 0 - 1 + 0.5 - 0
  = j - 0.5 -> [-0.5, 0.5, 1.5, 2.5]

xxmin = -1
xx = (j + xxmin - center + 0.5 - align_corners_delta)
  = j - 1 - 1 + 0.5 - 0
  = j - 1.5 -> [-1.5, -0.5, 0.5, 1.5]


i = 1
xmin = max(2 * (i - 1) + 1, 0) -> max(1, 0) = 1
center = 2 * (i + 0.5) -> 3
x = (j + xmin - center + 0.5 - align_corners_delta)
  = j + 1 - 3 + 0.5 - 0
  = j - 1.5 -> [-1.5, -0.5, 0.5, 1.5]
```