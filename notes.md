# WIP on Vectorized bilinear interpolation uint8, channels last

- Install Pillow-SIMD
```
pip uninstall -y pillow && CC="cc -mavx2" pip install --no-cache-dir --force-reinstall pillow-simd
```

## TODO:

- [x] Understand better the problem with `xmax - 1` boundary
```
    for (; x < xmax - 1; x += 2) {
      auto source = _mm_loadl_epi64((__m128i*) (lineIn + stride * (x + xmin)));
```
I have an impression that here xmax-1 is wrong as we read 16 bytes when casting to `m128i` even if `_mm_loadl_epi64` loads only 8 bytes. IMO, correct boundare is `xmax - 3`.
Memory boundary is `lineIn + stride * (xmax + xmin)`.

https://github.com/uploadcare/pillow-simd/blob/668aa48d12305b8f093958792a5e4f690c2583d6/src/libImaging/ResampleSIMDHorizontalConv.c#L153


  - [x] Check if statement "Memory boundary is `lineIn + stride * (xmax + xmin)`." is true ?
=> xmax is computed as following:
```c++
        // Round the value
        xmax = (int)(center + support + 0.5);
        if (xmax > inSize) {
            xmax = inSize;
        }
        xmax -= xmin;
```
==> so `xmin + xmax = inSize`

  - [x] Run ASAN on a simple example to reproduce the issue

Notes: https://github.com/pytorch/pytorch/blob/master/CONTRIBUTING.md#building-pytorch-with-asan

```
apt-get install clang llvm
```

```
cd cpp/asan

make check_mem_boundaries_with_asan && ./check_mem_boundaries_with_asan
```

- [ ] Figure out how to optimize horizontal passes.

PR (git78eda38) perfs are slower than Pillow-SIMD. For ch=4, there is an overhead of 30us for horizontal vs nightly.

```
      3 torch.uint8 channels_last bilinear 256 -> (224, 224) aa=True   |         129.9          |              249.3
      3 torch.uint8 channels_last bilinear 256 -> (256, 224) aa=True   |          91.9          |              207.4
      3 torch.uint8 channels_last bilinear 256 -> (224, 256) aa=True   |          51.9          |               52.0

      4 torch.uint8 channels_last bilinear 256 -> (224, 224) aa=True   |                        |              170.9

      4 torch.uint8 channels_last bilinear 256 -> (256, 224) aa=True   |                        |              127.2
      4 torch.uint8 channels_last bilinear 256 -> (256, 224) aa=False  |                        |              118.5

      4 torch.uint8 channels_last bilinear 256 -> (224, 256) aa=True   |                        |               62.0
      4 torch.uint8 channels_last bilinear 256 -> (225, 256) aa=False  |                        |               51.6
```

Nightly version horizontal only and vertical only give same perfs as Pillow-SIMD:
```
      3 torch.uint8 channels_last bilinear 256 -> (256, 224) aa=True   |          91.1          |              262.9
      3 torch.uint8 channels_last bilinear 256 -> (224, 256) aa=True   |          51.5          |              218.3

      4 torch.uint8 channels_last bilinear 256 -> (256, 224) aa=True   |                        |               99.3
      4 torch.uint8 channels_last bilinear 256 -> (224, 256) aa=True   |                        |               56.9
```

but Nightly horizontal + vertical are a bit slower than Pillow by ~10us:
```
      3 torch.uint8 channels_last bilinear 256 -> (224, 224) aa=True   |         128.8          |              292.9
      4 torch.uint8 channels_last bilinear 256 -> (224, 224) aa=True   |                        |              141.3
```

Some debuging logs:
```
channels_last 3 torch.uint8 256 (224, 224) True bilinear
- Compute horizontal weights
- Compute vertical weights
- is_rgb_or_rgba: true
- Allocatate horizontal buffer
- Apply ImagingResampleHorizontal
- Apply ImagingResampleVertical

channels_last 3 torch.uint8 256 (256, 224) True bilinear
- Compute horizontal weights
- is_rgb_or_rgba: true
- Apply ImagingResampleHorizontal

channels_last 3 torch.uint8 256 (224, 256) True bilinear
- Compute vertical weights
- is_rgb_or_rgba: true
- Apply ImagingResampleVertical

channels_last 4 torch.uint8 256 (224, 224) True bilinear
- Compute horizontal weights
- Compute vertical weights
- is_rgb_or_rgba: true
- Allocatate horizontal buffer
- Apply ImagingResampleHorizontal
- Apply ImagingResampleVertical

channels_last 4 torch.uint8 256 (256, 224) True bilinear
- Compute horizontal weights
- is_rgb_or_rgba: true
- Apply ImagingResampleHorizontal

channels_last 4 torch.uint8 256 (224, 256) True bilinear
- Compute vertical weights
- is_rgb_or_rgba: true
- Apply ImagingResampleVertical
```


- Profiling `channels_last 4 torch.uint8 256 (224, 224) True bilinear` = profile_vect_interp.cpp with VTune (30000 iters)
```
Source Function / Function / Call Stack	CPU Time	Clockticks	Instructions Retired	CPI Rate	Retiring	Front-End Bound	Bad Speculation	Memory Bound	Divider	Port Utilization	Average CPU Frequency	Module	Function (Full)	Source File	Start Address

test::ImagingResampleHorizontalConvolution8u4x	1.120s	4,038,700,000	4,201,500,000	0.961	8.6%	2.6%	0.7%	53.1%	0.5%	6.0%	3.6 GHz	[Unknown]

test::ImagingResampleVerticalConvolution8u	0.338s	1,213,000,000	1,366,000,000	0.888	8.2%	7.8%	1.5%	70.0%	17.2%	1.1%	3.6 GHz	[Unknown]

test::HelperInterpBase::_compute_indices_weights_aa<double, double (*)(double), (int)2>	0.065s	236,000,000	186,400,000	1.266	7.4%	10.0%	0.0%	100.0%	59.0%	0.0%	3.7 GHz	[Unknown]

test::HelperInterpBase::_compute_indices_int16_weights_aa<double (*)(double)>	0.043s	156,100,000	190,200,000	0.821	10.8%	8.4%	1.2%	72.3%	0.0%	2.7%	3.7 GHz	[Unknown]

test::HelperInterpLinear::aa_filter<double>	0.016s	53,700,000	61,600,000	0.872	10.0%	4.4%	0.0%	44.7%	0.0%	18.6%	3.5 GHz	[Unknown]

test::upsample_avx_bilinear_uint8<std::vector<c10::optional<double>, std::allocator<c10::optional<double>>>, test::HelperInterpLinear>	0.005s	17,400,000	3,300,000	5.273	4.1%	36.1%	0.6%	63.4%	0.0%	0.0%	3.8 GHz	[Unknown]
```

- Computing `b4_xmax`-like values is expensive in the loop:
```
// lineIn0 + stride * (x + xmin) + 16 <= lineIn0 + stride * (xmax + xmin)
// --> x <= xmax - 16.0 / stride
// --> x < xmax + 1 - int(16.0 / stride)
```


- Using templated ImagingResampleHorizontal and ImagingResampleVertical and fixing
"Computing `b4_xmax`-like values is expensive in the loop"
```
Num threads: 1

PIL version:  9.0.0.post1
[----------------------------------------------------------- Resize ----------------------------------------------------------]
                                                                      |  Pillow (9.0.0.post1)  |  torch (2.1.0a0+git78eda38) PR
1 threads: --------------------------------------------------------------------------------------------------------------------
      3 torch.uint8 channels_last bilinear 256 -> (224, 224) aa=True  |         127.0          |              238.9
      4 torch.uint8 channels_last bilinear 256 -> (224, 224) aa=True  |                        |              139.9

      3 torch.uint8 channels_last bilinear 256 -> (32, 32) aa=True    |          38.8          |               56.8
      4 torch.uint8 channels_last bilinear 256 -> (32, 32) aa=True    |                        |               50.6

      # Horizontal pass only
      3 torch.uint8 channels_last bilinear 256 -> (256, 224) aa=True  |          91.4          |              196.7
      4 torch.uint8 channels_last bilinear 256 -> (256, 224) aa=True  |                        |               94.0
      3 torch.uint8 channels_last bilinear 256 -> (256, 225) aa=True  |          93.0          |              193.7
      4 torch.uint8 channels_last bilinear 256 -> (256, 225) aa=True  |                        |               97.8

      # Vertical pass only
      3 torch.uint8 channels_last bilinear 256 -> (224, 256) aa=True  |          51.8          |               54.3
      3 torch.uint8 channels_last bilinear 256 -> (225, 256) aa=True  |          51.9          |               54.9
      4 torch.uint8 channels_last bilinear 256 -> (224, 256) aa=True  |                        |               58.2
      4 torch.uint8 channels_last bilinear 256 -> (225, 256) aa=True  |                        |               58.2

Times are in microseconds (us).
```
vs Nightly
```
      3 torch.uint8 channels_last bilinear 256 -> (224, 224) aa=True   |         128.8          |              292.9
      4 torch.uint8 channels_last bilinear 256 -> (224, 224) aa=True   |                        |              141.3
```

-
```
Num threads: 1

PIL version:  9.0.0.post1
[----------------------------------------------------------- Resize -----------------------------------------------------------]
                                                                       |  Pillow (9.0.0.post1)  |  torch (2.1.0a0+gitb30e839) PR
1 threads: ---------------------------------------------------------------------------------------------------------------------
      3 torch.uint8 channels_last bilinear 256 -> (224, 224) aa=True  |         127.8          |              164.0
      4 torch.uint8 channels_last bilinear 256 -> (224, 224) aa=True  |                        |              139.2

      3 torch.uint8 channels_last bilinear 256 -> (32, 32) aa=True    |          38.8          |               56.5
      4 torch.uint8 channels_last bilinear 256 -> (32, 32) aa=True    |                        |               51.6

      # Horizontal pass only
      3 torch.uint8 channels_last bilinear 256 -> (256, 224) aa=True  |          91.6          |              122.8
      3 torch.uint8 channels_last bilinear 256 -> (256, 227) aa=True  |          92.9          |              125.2
      4 torch.uint8 channels_last bilinear 256 -> (256, 224) aa=True  |                        |               94.0
      4 torch.uint8 channels_last bilinear 256 -> (256, 227) aa=True  |                        |               99.5

      # Vertical pass only
      3 torch.uint8 channels_last bilinear 256 -> (224, 256) aa=True  |          53.4          |               54.0
      3 torch.uint8 channels_last bilinear 256 -> (227, 256) aa=True  |          56.9          |               54.3
      4 torch.uint8 channels_last bilinear 256 -> (224, 256) aa=True  |                        |               57.5
      4 torch.uint8 channels_last bilinear 256 -> (227, 256) aa=True  |                        |               57.6

Times are in microseconds (us).
```


- try branchless code

```
Num threads: 1

PIL version:  9.0.0.post1
[----------------------------------------------------------- Resize ----------------------------------------------------------]
                                                                      |  Pillow (9.0.0.post1)  |  torch (2.1.0a0+gitb30e839) PR
1 threads: --------------------------------------------------------------------------------------------------------------------
      3 torch.uint8 channels_last bilinear 256 -> (224, 224) aa=True  |         128.1          |              166.7
      3 torch.uint8 channels_last bilinear 256 -> (256, 224) aa=True  |          91.0          |              124.3
      3 torch.uint8 channels_last bilinear 256 -> (256, 227) aa=True  |          93.1          |              128.2

Times are in microseconds (us).
```


- Results for RGB vs RGBA runtime vs PIL
```

Num threads: 1

PIL version:  9.0.0.post1
[----------------------------------------------------------- Resize ----------------------------------------------------------]
                                                                      |  Pillow (9.0.0.post1)  |  torch (2.1.0a0+gitb30e839) PR
1 threads: --------------------------------------------------------------------------------------------------------------------
      3 torch.uint8 channels_last bilinear 700 -> (700, 612) aa=True  |         602.3          |              765.5
      4 torch.uint8 channels_last bilinear 700 -> (700, 612) aa=True  |                        |              637.9

      3 torch.uint8 channels_last bilinear 520 -> (520, 455) aa=True  |         339.6          |              436.9
      4 torch.uint8 channels_last bilinear 520 -> (520, 455) aa=True  |                        |              363.7

      3 torch.uint8 channels_last bilinear 256 -> (256, 224) aa=True  |          90.8          |              123.9
      4 torch.uint8 channels_last bilinear 256 -> (256, 224) aa=True  |                        |               96.4

      3 torch.uint8 channels_last bilinear 172 -> (172, 150) aa=True  |          45.9          |               66.3
      4 torch.uint8 channels_last bilinear 172 -> (172, 150) aa=True  |                        |               54.1

      3 torch.uint8 channels_last bilinear 128 -> (128, 112) aa=True  |          27.5          |               41.8
      4 torch.uint8 channels_last bilinear 128 -> (128, 112) aa=True  |                        |               36.0

      3 torch.uint8 channels_last bilinear 96 -> (96, 84) aa=True     |          17.3          |               29.2
      4 torch.uint8 channels_last bilinear 96 -> (96, 84) aa=True     |                        |               25.8

      3 torch.uint8 channels_last bilinear 74 -> (74, 64) aa=True     |          12.3          |               22.8
      4 torch.uint8 channels_last bilinear 74 -> (74, 64) aa=True     |                        |               20.6

      3 torch.uint8 channels_last bilinear 64 -> (64, 56) aa=True     |           9.9          |               19.7
      4 torch.uint8 channels_last bilinear 64 -> (64, 56) aa=True     |                        |               18.4

      3 torch.uint8 channels_last bilinear 32 -> (32, 28) aa=True     |           5.0          |               13.6
      4 torch.uint8 channels_last bilinear 32 -> (32, 28) aa=True     |                        |               13.2

Times are in microseconds (us).
```

- No more `mask_ch` and write as int32 in horizontal pass for the last element in the line except the last line
```
Num threads: 1

PIL version:  9.0.0.post1
[----------------------------------------------------------- Resize ----------------------------------------------------------]
                                                                      |  Pillow (9.0.0.post1)  |  torch (2.1.0a0+gitb30e839) PR
1 threads: --------------------------------------------------------------------------------------------------------------------
      3 torch.uint8 channels_last bilinear 256 -> (224, 224) aa=True  |         127.8          |              145.8
      4 torch.uint8 channels_last bilinear 256 -> (224, 224) aa=True  |                        |              144.5

      3 torch.uint8 channels_last bilinear 256 -> (256, 224) aa=True  |          91.3          |              104.1
      4 torch.uint8 channels_last bilinear 256 -> (256, 224) aa=True  |                        |               97.9
      3 torch.uint8 channels_last bilinear 256 -> (256, 227) aa=True  |          93.2          |              107.6
      4 torch.uint8 channels_last bilinear 256 -> (256, 227) aa=True  |                        |              102.8

      3 torch.uint8 channels_last bilinear 256 -> (224, 256) aa=True  |          52.3          |               55.9
      4 torch.uint8 channels_last bilinear 256 -> (224, 256) aa=True  |                        |               59.5
      3 torch.uint8 channels_last bilinear 256 -> (227, 256) aa=True  |          52.5          |               56.3
      4 torch.uint8 channels_last bilinear 256 -> (227, 256) aa=True  |                        |               59.5

Times are in microseconds (us).
```

- Few other tweaks
```
Num threads: 1

PIL version:  9.0.0.post1
[----------------------------------------------------------- Resize ----------------------------------------------------------]
                                                                      |  Pillow (9.0.0.post1)  |  torch (2.1.0a0+git1d3a939) PR
1 threads: --------------------------------------------------------------------------------------------------------------------
      3 torch.uint8 channels_last bilinear 256 -> (224, 224) aa=True  |         127.3          |              145.3
      4 torch.uint8 channels_last bilinear 256 -> (224, 224) aa=True  |                        |              140.6

      3 torch.uint8 channels_last bilinear 256 -> (256, 224) aa=True  |          92.3          |              102.0
      4 torch.uint8 channels_last bilinear 256 -> (256, 224) aa=True  |                        |               94.7
      3 torch.uint8 channels_last bilinear 256 -> (256, 227) aa=True  |          93.4          |              105.5
      4 torch.uint8 channels_last bilinear 256 -> (256, 227) aa=True  |                        |              100.7

      3 torch.uint8 channels_last bilinear 256 -> (224, 256) aa=True  |          51.9          |               54.2
      4 torch.uint8 channels_last bilinear 256 -> (224, 256) aa=True  |                        |               57.5
      3 torch.uint8 channels_last bilinear 256 -> (227, 256) aa=True  |          53.9          |               55.0
      4 torch.uint8 channels_last bilinear 256 -> (227, 256) aa=True  |                        |               57.6

Times are in microseconds (us).
```


- Fixed unsafe memory reads
```
Num threads: 1

PIL version:  9.0.0.post1
[----------------------------------------------------------- Resize ----------------------------------------------------------]
                                                                      |  Pillow (9.0.0.post1)  |  torch (2.1.0a0+git1d3a939) PR
1 threads: --------------------------------------------------------------------------------------------------------------------
      3 torch.uint8 channels_last bilinear 256 -> (224, 224) aa=True  |         128.5          |              149.5
      4 torch.uint8 channels_last bilinear 256 -> (224, 224) aa=True  |                        |              142.9

      3 torch.uint8 channels_last bilinear 256 -> (256, 224) aa=True  |          91.2          |              104.3
      4 torch.uint8 channels_last bilinear 256 -> (256, 224) aa=True  |                        |               95.3
      3 torch.uint8 channels_last bilinear 256 -> (256, 227) aa=True  |          93.3          |              110.7
      4 torch.uint8 channels_last bilinear 256 -> (256, 227) aa=True  |                        |              100.1

      3 torch.uint8 channels_last bilinear 256 -> (224, 256) aa=True  |          52.0          |               56.3
      4 torch.uint8 channels_last bilinear 256 -> (224, 256) aa=True  |                        |               59.8
      3 torch.uint8 channels_last bilinear 256 -> (227, 256) aa=True  |          51.8          |               55.8
      4 torch.uint8 channels_last bilinear 256 -> (227, 256) aa=True  |                        |               59.9

Times are in microseconds (us).
```


- Check potential memory corruptions with ASAN

```
cd /tmp/pth/interpolate_vec_uint8/
run_with_asan python verif_interp2.py verif_expected --is_ref=False
```


- Check `wt_max` computation overhead. `wt_max` is computed in `_compute_weights_aa`

Before (`wt_max` is NOT computed inside `_compute_weights_aa`):
```
cd /tmp/pth/interpolate_vec_uint8/ && python -u check_interp2.py

Num threads: 1

PIL version:  9.0.0.post1
[------------------------------------------------ Resize ------------------------------------------------]
                                                                          |  torch (2.1.0a0+git0c58e8a) PR
1 threads: -----------------------------------------------------------------------------------------------
      3 torch.float32 channels_first bilinear 700 -> (224, 224) aa=True   |              2586.9
      4 torch.float32 channels_first bilinear 700 -> (224, 224) aa=True   |              3437.1
      3 torch.float32 channels_first bilinear 700 -> (224, 224) aa=False  |              1832.3
      4 torch.float32 channels_first bilinear 700 -> (224, 224) aa=False  |               996.8
      3 torch.float32 channels_first bilinear 520 -> (224, 224) aa=True   |              1657.1
      4 torch.float32 channels_first bilinear 520 -> (224, 224) aa=True   |              2195.9
      3 torch.float32 channels_first bilinear 520 -> (224, 224) aa=False  |              1416.8
      4 torch.float32 channels_first bilinear 520 -> (224, 224) aa=False  |               988.3
      3 torch.float32 channels_first bilinear 256 -> (224, 224) aa=True   |               621.0
      4 torch.float32 channels_first bilinear 256 -> (224, 224) aa=True   |               816.9
      3 torch.float32 channels_first bilinear 256 -> (224, 224) aa=False  |              1049.6
      4 torch.float32 channels_first bilinear 256 -> (224, 224) aa=False  |               967.8

Times are in microseconds (us).
```

Before (`wt_max` is computed inside `_compute_weights_aa`):
```
Num threads: 1

PIL version:  9.0.0.post1
[------------------------------------------------ Resize ------------------------------------------------]
                                                                          |  torch (2.1.0a0+git0c58e8a) PR
1 threads: -----------------------------------------------------------------------------------------------
      3 torch.float32 channels_first bilinear 700 -> (224, 224) aa=True   |              2586.5
      4 torch.float32 channels_first bilinear 700 -> (224, 224) aa=True   |              3460.0
      3 torch.float32 channels_first bilinear 700 -> (224, 224) aa=False  |              1823.1
      4 torch.float32 channels_first bilinear 700 -> (224, 224) aa=False  |               986.2
      3 torch.float32 channels_first bilinear 520 -> (224, 224) aa=True   |              1668.1
      4 torch.float32 channels_first bilinear 520 -> (224, 224) aa=True   |              2207.0
      3 torch.float32 channels_first bilinear 520 -> (224, 224) aa=False  |              1415.0
      4 torch.float32 channels_first bilinear 520 -> (224, 224) aa=False  |               975.7
      3 torch.float32 channels_first bilinear 256 -> (224, 224) aa=True   |               624.4
      4 torch.float32 channels_first bilinear 256 -> (224, 224) aa=True   |               820.1
      3 torch.float32 channels_first bilinear 256 -> (224, 224) aa=False  |              1048.9
      4 torch.float32 channels_first bilinear 256 -> (224, 224) aa=False  |               955.9

Times are in microseconds (us).
```

- Code cosmetics:
before:
```
PIL version:  9.0.0.post1
[----------------------------------------------------------- Resize ----------------------------------------------------------]
                                                                      |  Pillow (9.0.0.post1)  |  torch (2.1.0a0+git5309c44) PR
1 threads: --------------------------------------------------------------------------------------------------------------------
      3 torch.uint8 channels_last bilinear 256 -> (224, 224) aa=True  |         127.9          |              292.4
      4 torch.uint8 channels_last bilinear 256 -> (224, 224) aa=True  |                        |              140.7
      3 torch.uint8 channels_last bilinear 256 -> (256, 224) aa=True  |          91.8          |              263.2
      4 torch.uint8 channels_last bilinear 256 -> (256, 224) aa=True  |                        |               99.2
      3 torch.uint8 channels_last bilinear 256 -> (224, 256) aa=True  |          51.6          |              218.7
      4 torch.uint8 channels_last bilinear 256 -> (224, 256) aa=True  |                        |               55.3

Times are in microseconds (us).
```
after:
```
PIL version:  9.0.0.post1
[----------------------------------------------------------- Resize ----------------------------------------------------------]
                                                                      |  Pillow (9.0.0.post1)  |  torch (2.1.0a0+git0c58e8a) PR
1 threads: --------------------------------------------------------------------------------------------------------------------
      3 torch.uint8 channels_last bilinear 256 -> (224, 224) aa=True  |         127.8          |              296.7
      4 torch.uint8 channels_last bilinear 256 -> (224, 224) aa=True  |                        |              144.9
      3 torch.uint8 channels_last bilinear 256 -> (256, 224) aa=True  |          92.7          |              269.1
      4 torch.uint8 channels_last bilinear 256 -> (256, 224) aa=True  |                        |              105.2
      3 torch.uint8 channels_last bilinear 256 -> (224, 256) aa=True  |          51.5          |              215.4
      4 torch.uint8 channels_last bilinear 256 -> (224, 256) aa=True  |                        |               52.6

Times are in microseconds (us).
```

- Check "// There is no much perf gain using more detailed condition // if (num_channels == 3) {" (ImagingResampleVerticalConvolution8u)

Condition `if (num_channels == 3 && ids_min + j + data_size * i + 4 >= in_max_size) {`
```
PIL version:  9.0.0.post1
[------------------------------------------------------------ Resize -----------------------------------------------------------]
                                                                        |  Pillow (9.0.0.post1)  |  torch (2.1.0a0+git0c58e8a) PR
1 threads: ----------------------------------------------------------------------------------------------------------------------
      3 torch.uint8 channels_last bilinear 256 -> (224, 256) aa=True    |          51.9          |                53.7
      4 torch.uint8 channels_last bilinear 256 -> (224, 256) aa=True    |                        |                58.1
      3 torch.uint8 channels_last bilinear 256 -> (227, 256) aa=True    |          51.7          |                54.0
      4 torch.uint8 channels_last bilinear 256 -> (227, 256) aa=True    |                        |                57.5
      3 torch.uint8 channels_last bilinear 256 -> (224, 256) aa=False   |                        |                49.1
      4 torch.uint8 channels_last bilinear 256 -> (224, 256) aa=False   |                        |                52.5
      3 torch.uint8 channels_last bilinear 256 -> (32, 256) aa=False    |                        |                18.7
      4 torch.uint8 channels_last bilinear 256 -> (32, 256) aa=False    |                        |                19.2
      3 torch.uint8 channels_last bilinear 520 -> (455, 520) aa=True    |         179.5          |               154.2
      4 torch.uint8 channels_last bilinear 520 -> (455, 520) aa=True    |                        |               167.1
      3 torch.uint8 channels_last bilinear 520 -> (227, 520) aa=True    |         159.6          |               132.5
      4 torch.uint8 channels_last bilinear 520 -> (227, 520) aa=True    |                        |               135.8
      3 torch.uint8 channels_last bilinear 520 -> (224, 520) aa=False   |                        |                77.2
      4 torch.uint8 channels_last bilinear 520 -> (224, 520) aa=False   |                        |                83.8
      3 torch.uint8 channels_last bilinear 520 -> (32, 520) aa=False    |                        |                23.2
      4 torch.uint8 channels_last bilinear 520 -> (32, 520) aa=False    |                        |                24.3
      3 torch.uint8 channels_last bilinear 712 -> (623, 712) aa=True    |         320.7          |               256.0
      4 torch.uint8 channels_last bilinear 712 -> (623, 712) aa=True    |                        |               287.9
      3 torch.uint8 channels_last bilinear 712 -> (227, 712) aa=True    |         260.7          |               190.2
      4 torch.uint8 channels_last bilinear 712 -> (227, 712) aa=True    |                        |               208.3
      3 torch.uint8 channels_last bilinear 712 -> (224, 712) aa=False   |                        |                96.0
      4 torch.uint8 channels_last bilinear 712 -> (224, 712) aa=False   |                        |               104.8
      3 torch.uint8 channels_last bilinear 712 -> (32, 712) aa=False    |                        |                25.8
      4 torch.uint8 channels_last bilinear 712 -> (32, 712) aa=False    |                        |                27.3
      3 torch.uint8 channels_first bilinear 256 -> (224, 256) aa=True   |          51.7          |               228.2
      4 torch.uint8 channels_first bilinear 256 -> (224, 256) aa=True   |                        |               200.5
      3 torch.uint8 channels_first bilinear 256 -> (227, 256) aa=True   |          51.7          |               226.7
      4 torch.uint8 channels_first bilinear 256 -> (227, 256) aa=True   |                        |               197.1
      3 torch.uint8 channels_first bilinear 256 -> (224, 256) aa=False  |                        |               219.1
      4 torch.uint8 channels_first bilinear 256 -> (224, 256) aa=False  |                        |               210.0
      3 torch.uint8 channels_first bilinear 256 -> (32, 256) aa=False   |                        |               109.5
      4 torch.uint8 channels_first bilinear 256 -> (32, 256) aa=False   |                        |                57.1
      3 torch.uint8 channels_first bilinear 520 -> (455, 520) aa=True   |         179.2          |               840.9
      4 torch.uint8 channels_first bilinear 520 -> (455, 520) aa=True   |                        |               736.1
      3 torch.uint8 channels_first bilinear 520 -> (227, 520) aa=True   |         164.1          |               629.5
      4 torch.uint8 channels_first bilinear 520 -> (227, 520) aa=True   |                        |               446.9
      3 torch.uint8 channels_first bilinear 520 -> (224, 520) aa=False  |                        |               565.5
      4 torch.uint8 channels_first bilinear 520 -> (224, 520) aa=False  |                        |               405.1
      3 torch.uint8 channels_first bilinear 520 -> (32, 520) aa=False   |                        |               358.7
      4 torch.uint8 channels_first bilinear 520 -> (32, 520) aa=False   |                        |               130.1
      3 torch.uint8 channels_first bilinear 712 -> (623, 712) aa=True   |         318.9          |              1560.8
      4 torch.uint8 channels_first bilinear 712 -> (623, 712) aa=True   |                        |              1343.6
      3 torch.uint8 channels_first bilinear 712 -> (227, 712) aa=True   |         261.4          |              1037.8
      4 torch.uint8 channels_first bilinear 712 -> (227, 712) aa=True   |                        |               693.0
      3 torch.uint8 channels_first bilinear 712 -> (224, 712) aa=False  |                        |               928.3
      4 torch.uint8 channels_first bilinear 712 -> (224, 712) aa=False  |                        |               577.9
      3 torch.uint8 channels_first bilinear 712 -> (32, 712) aa=False   |                        |               636.2
      4 torch.uint8 channels_first bilinear 712 -> (32, 712) aa=False   |                        |               207.8

Times are in microseconds (us).
```

Condition `if (num_channels == 3) {`
```
PIL version:  9.0.0.post1
[------------------------------------------------------------ Resize -----------------------------------------------------------]
                                                                        |  Pillow (9.0.0.post1)  |  torch (2.1.0a0+git0c58e8a) PR
1 threads: ----------------------------------------------------------------------------------------------------------------------
      3 torch.uint8 channels_last bilinear 256 -> (224, 256) aa=True    |          52.2          |                55.3
      4 torch.uint8 channels_last bilinear 256 -> (224, 256) aa=True    |                        |                57.7
      3 torch.uint8 channels_last bilinear 256 -> (227, 256) aa=True    |          54.2          |                55.0
      4 torch.uint8 channels_last bilinear 256 -> (227, 256) aa=True    |                        |                57.9
      3 torch.uint8 channels_last bilinear 256 -> (224, 256) aa=False   |                        |                50.0
      4 torch.uint8 channels_last bilinear 256 -> (224, 256) aa=False   |                        |                52.1
      3 torch.uint8 channels_last bilinear 256 -> (32, 256) aa=False    |                        |                18.7
      4 torch.uint8 channels_last bilinear 256 -> (32, 256) aa=False    |                        |                19.2
      3 torch.uint8 channels_last bilinear 520 -> (455, 520) aa=True    |         179.5          |               153.6
      4 torch.uint8 channels_last bilinear 520 -> (455, 520) aa=True    |                        |               166.5
      3 torch.uint8 channels_last bilinear 520 -> (227, 520) aa=True    |         161.9          |               132.1
      4 torch.uint8 channels_last bilinear 520 -> (227, 520) aa=True    |                        |               136.7
      3 torch.uint8 channels_last bilinear 520 -> (224, 520) aa=False   |                        |                77.2
      4 torch.uint8 channels_last bilinear 520 -> (224, 520) aa=False   |                        |                83.8
      3 torch.uint8 channels_last bilinear 520 -> (32, 520) aa=False    |                        |                23.2
      4 torch.uint8 channels_last bilinear 520 -> (32, 520) aa=False    |                        |                24.1
      3 torch.uint8 channels_last bilinear 712 -> (623, 712) aa=True    |         317.9          |               257.4
      4 torch.uint8 channels_last bilinear 712 -> (623, 712) aa=True    |                        |               286.0
      3 torch.uint8 channels_last bilinear 712 -> (227, 712) aa=True    |         261.6          |               192.8
      4 torch.uint8 channels_last bilinear 712 -> (227, 712) aa=True    |                        |               209.8
      3 torch.uint8 channels_last bilinear 712 -> (224, 712) aa=False   |                        |                96.4
      4 torch.uint8 channels_last bilinear 712 -> (224, 712) aa=False   |                        |               104.6
      3 torch.uint8 channels_last bilinear 712 -> (32, 712) aa=False    |                        |                25.9
      4 torch.uint8 channels_last bilinear 712 -> (32, 712) aa=False    |                        |                27.2
      3 torch.uint8 channels_first bilinear 256 -> (224, 256) aa=True   |          52.0          |               223.4
      4 torch.uint8 channels_first bilinear 256 -> (224, 256) aa=True   |                        |               188.1
      3 torch.uint8 channels_first bilinear 256 -> (227, 256) aa=True   |          52.0          |               223.1
      4 torch.uint8 channels_first bilinear 256 -> (227, 256) aa=True   |                        |               179.4
      3 torch.uint8 channels_first bilinear 256 -> (224, 256) aa=False  |                        |               216.8
      4 torch.uint8 channels_first bilinear 256 -> (224, 256) aa=False  |                        |               176.8
      3 torch.uint8 channels_first bilinear 256 -> (32, 256) aa=False   |                        |               109.9
      4 torch.uint8 channels_first bilinear 256 -> (32, 256) aa=False   |                        |                53.1
      3 torch.uint8 channels_first bilinear 520 -> (455, 520) aa=True   |         179.0          |               837.4
      4 torch.uint8 channels_first bilinear 520 -> (455, 520) aa=True   |                        |               652.0
      3 torch.uint8 channels_first bilinear 520 -> (227, 520) aa=True   |         164.1          |               620.4
      4 torch.uint8 channels_first bilinear 520 -> (227, 520) aa=True   |                        |               415.1
      3 torch.uint8 channels_first bilinear 520 -> (224, 520) aa=False  |                        |               565.5
      4 torch.uint8 channels_first bilinear 520 -> (224, 520) aa=False  |                        |               358.9
      3 torch.uint8 channels_first bilinear 520 -> (32, 520) aa=False   |                        |               358.0
      4 torch.uint8 channels_first bilinear 520 -> (32, 520) aa=False   |                        |               121.7
      3 torch.uint8 channels_first bilinear 712 -> (623, 712) aa=True   |         320.3          |              1546.3
      4 torch.uint8 channels_first bilinear 712 -> (623, 712) aa=True   |                        |              1209.9
      3 torch.uint8 channels_first bilinear 712 -> (227, 712) aa=True   |         278.5          |              1027.3
      4 torch.uint8 channels_first bilinear 712 -> (227, 712) aa=True   |                        |               619.5
      3 torch.uint8 channels_first bilinear 712 -> (224, 712) aa=False  |                        |               922.6
      4 torch.uint8 channels_first bilinear 712 -> (224, 712) aa=False  |                        |               510.9
      3 torch.uint8 channels_first bilinear 712 -> (32, 712) aa=False   |                        |               635.3
      4 torch.uint8 channels_first bilinear 712 -> (32, 712) aa=False   |                        |               193.8

Times are in microseconds (us).
```


- Check if we really need template for vertical pass. Godbolt assembly code is the same with constexpr or just const num_channels: see godbolt_source_vertpass.md

With templated num_channels:
```
Num threads: 1

PIL version:  9.0.0.post1
[------------------------------------------------------------ Resize -----------------------------------------------------------]
                                                                        |  Pillow (9.0.0.post1)  |  torch (2.1.0a0+gitcc42a3f) PR
1 threads: ----------------------------------------------------------------------------------------------------------------------
      3 torch.uint8 channels_last bilinear 256 -> (224, 224) aa=True    |         127.6          |              159.4
      4 torch.uint8 channels_last bilinear 256 -> (224, 224) aa=True    |                        |              150.0
      3 torch.uint8 channels_last bilinear 256 -> (224, 224) aa=False   |                        |              148.3
      4 torch.uint8 channels_last bilinear 256 -> (224, 224) aa=False   |                        |              135.6
      3 torch.uint8 channels_last bilinear 256 -> (256, 224) aa=True    |          91.2          |              116.5
      4 torch.uint8 channels_last bilinear 256 -> (256, 224) aa=True    |                        |              103.3
      3 torch.uint8 channels_last bilinear 256 -> (256, 224) aa=False   |                        |              110.5
      4 torch.uint8 channels_last bilinear 256 -> (256, 224) aa=False   |                        |               96.4
      3 torch.uint8 channels_last bilinear 256 -> (224, 256) aa=True    |          52.3          |               55.9
      4 torch.uint8 channels_last bilinear 256 -> (224, 256) aa=True    |                        |               58.8
      3 torch.uint8 channels_last bilinear 256 -> (224, 256) aa=False   |                        |               50.6
      4 torch.uint8 channels_last bilinear 256 -> (224, 256) aa=False   |                        |               53.5
      3 torch.uint8 channels_first bilinear 256 -> (224, 224) aa=True   |         129.2          |              302.4
      4 torch.uint8 channels_first bilinear 256 -> (224, 224) aa=True   |                        |              276.7
      3 torch.uint8 channels_first bilinear 256 -> (224, 224) aa=False  |                        |              289.3
      4 torch.uint8 channels_first bilinear 256 -> (224, 224) aa=False  |                        |              268.1
      3 torch.uint8 channels_first bilinear 256 -> (256, 224) aa=True   |          90.7          |              270.3
      4 torch.uint8 channels_first bilinear 256 -> (256, 224) aa=True   |                        |              260.0
      3 torch.uint8 channels_first bilinear 256 -> (256, 224) aa=False  |                        |              262.7
      4 torch.uint8 channels_first bilinear 256 -> (256, 224) aa=False  |                        |              258.9
      3 torch.uint8 channels_first bilinear 256 -> (224, 256) aa=True   |          53.9          |              225.4
      4 torch.uint8 channels_first bilinear 256 -> (224, 256) aa=True   |                        |              214.8
      3 torch.uint8 channels_first bilinear 256 -> (224, 256) aa=False  |                        |              220.4
      4 torch.uint8 channels_first bilinear 256 -> (224, 256) aa=False  |                        |              205.1

Times are in microseconds (us).
```

Without templated num_channels:
```

Num threads: 1

PIL version:  9.0.0.post1
[------------------------------------------------------------ Resize -----------------------------------------------------------]
                                                                        |  Pillow (9.0.0.post1)  |  torch (2.1.0a0+gitcc42a3f) PR
1 threads: ----------------------------------------------------------------------------------------------------------------------
      3 torch.uint8 channels_last bilinear 256 -> (224, 224) aa=True    |         128.9          |              153.4
      4 torch.uint8 channels_last bilinear 256 -> (224, 224) aa=True    |                        |              144.3
      3 torch.uint8 channels_last bilinear 256 -> (224, 224) aa=False   |                        |              141.8
      4 torch.uint8 channels_last bilinear 256 -> (224, 224) aa=False   |                        |              131.4
      3 torch.uint8 channels_last bilinear 256 -> (256, 224) aa=True    |          91.1          |              113.9
      4 torch.uint8 channels_last bilinear 256 -> (256, 224) aa=True    |                        |              102.9
      3 torch.uint8 channels_last bilinear 256 -> (256, 224) aa=False   |                        |              106.1
      4 torch.uint8 channels_last bilinear 256 -> (256, 224) aa=False   |                        |               94.0
      3 torch.uint8 channels_last bilinear 256 -> (224, 256) aa=True    |          52.4          |               52.8
      4 torch.uint8 channels_last bilinear 256 -> (224, 256) aa=True    |                        |               55.4
      3 torch.uint8 channels_last bilinear 256 -> (224, 256) aa=False   |                        |               48.5
      4 torch.uint8 channels_last bilinear 256 -> (224, 256) aa=False   |                        |               49.8
      3 torch.uint8 channels_first bilinear 256 -> (224, 224) aa=True   |         127.8          |              296.8
      4 torch.uint8 channels_first bilinear 256 -> (224, 224) aa=True   |                        |              251.4
      3 torch.uint8 channels_first bilinear 256 -> (224, 224) aa=False  |                        |              285.5
      4 torch.uint8 channels_first bilinear 256 -> (224, 224) aa=False  |                        |              239.6
      3 torch.uint8 channels_first bilinear 256 -> (256, 224) aa=True   |          91.8          |              269.8
      4 torch.uint8 channels_first bilinear 256 -> (256, 224) aa=True   |                        |              228.4
      3 torch.uint8 channels_first bilinear 256 -> (256, 224) aa=False  |                        |              261.1
      4 torch.uint8 channels_first bilinear 256 -> (256, 224) aa=False  |                        |              222.6
      3 torch.uint8 channels_first bilinear 256 -> (224, 256) aa=True   |          52.6          |              220.5
      4 torch.uint8 channels_first bilinear 256 -> (224, 256) aa=True   |                        |              183.5
      3 torch.uint8 channels_first bilinear 256 -> (224, 256) aa=False  |                        |              216.1
      4 torch.uint8 channels_first bilinear 256 -> (224, 256) aa=False  |                        |              178.7

Times are in microseconds (us).
```

- Check templated num_channels in horizontal pass

Without templated num_channels
```
Num threads: 1

PIL version:  9.0.0.post1
[------------------------------------------------------------ Resize -----------------------------------------------------------]
                                                                        |  Pillow (9.0.0.post1)  |  torch (2.1.0a0+gitcc42a3f) PR
1 threads: ----------------------------------------------------------------------------------------------------------------------
      3 torch.uint8 channels_last bilinear 256 -> (224, 224) aa=True    |         128.8          |              161.7
      4 torch.uint8 channels_last bilinear 256 -> (224, 224) aa=True    |                        |              158.6
      3 torch.uint8 channels_last bilinear 256 -> (224, 224) aa=False   |                        |              150.6
      4 torch.uint8 channels_last bilinear 256 -> (224, 224) aa=False   |                        |              145.8
      3 torch.uint8 channels_last bilinear 256 -> (256, 224) aa=True    |          91.7          |              122.1
      4 torch.uint8 channels_last bilinear 256 -> (256, 224) aa=True    |                        |              116.5
      3 torch.uint8 channels_last bilinear 256 -> (256, 224) aa=False   |                        |              113.6
      4 torch.uint8 channels_last bilinear 256 -> (256, 224) aa=False   |                        |              107.7
      3 torch.uint8 channels_last bilinear 256 -> (224, 256) aa=True    |          52.7          |               53.7
      4 torch.uint8 channels_last bilinear 256 -> (224, 256) aa=True    |                        |               56.1
      3 torch.uint8 channels_last bilinear 256 -> (224, 256) aa=False   |                        |               49.8
      4 torch.uint8 channels_last bilinear 256 -> (224, 256) aa=False   |                        |               52.1
      3 torch.uint8 channels_first bilinear 256 -> (224, 224) aa=True   |         128.8          |              312.7
      4 torch.uint8 channels_first bilinear 256 -> (224, 224) aa=True   |                        |              267.0
      3 torch.uint8 channels_first bilinear 256 -> (224, 224) aa=False  |                        |              300.2
      4 torch.uint8 channels_first bilinear 256 -> (224, 224) aa=False  |                        |              254.9
      3 torch.uint8 channels_first bilinear 256 -> (256, 224) aa=True   |          91.8          |              283.8
      4 torch.uint8 channels_first bilinear 256 -> (256, 224) aa=True   |                        |              246.5
      3 torch.uint8 channels_first bilinear 256 -> (256, 224) aa=False  |                        |              277.1
      4 torch.uint8 channels_first bilinear 256 -> (256, 224) aa=False  |                        |              238.0
      3 torch.uint8 channels_first bilinear 256 -> (224, 256) aa=True   |          53.3          |              225.8
      4 torch.uint8 channels_first bilinear 256 -> (224, 256) aa=True   |                        |              187.4
      3 torch.uint8 channels_first bilinear 256 -> (224, 256) aa=False  |                        |              219.3
      4 torch.uint8 channels_first bilinear 256 -> (224, 256) aa=False  |                        |              181.6

Times are in microseconds (us).
```

- Fix ASAN issue: https://hud.pytorch.org/pr/96848#12024446137
```
2023-03-17T11:57:05.2469361Z test_nn.py::TestNNDeviceTypeCPU::test_upsamplingBiLinear2d_consistency_memory_format_torch_channels_last_antialias_False_align_corners_False_num_channels_3_output_size_32_cpu /var/lib/jenkins/workspace/aten/src/ATen/native/cpu/UpSampleKernelAVXAntialias.h:39:46: runtime error: load of misaligned address 0x7f2cd5a2584f for type 'int32_t' (aka 'int'), which requires 4 byte alignment
2023-03-17T11:57:05.2469468Z 0x7f2cd5a2584f: note: pointer points here
2023-03-17T11:57:05.2469608Z  cb 04 3f a9 3e  b0 f8 d7 7f c0 5e 5c 9a  ab 51 94 19 77 fa 51 eb  7d 3c b0 b6 2e e1 27 c5  0b 6d c8
```

```
run_with_asan pytest -vvv test/test_nn.py -k test_upsamplingBiLinear2d_consistency
```



## Run benchmarks: nightly vs PR

### On nightly
```
python -u run_bench_interp.py "output/$(date "+%Y%m%d-%H%M%S")-nightly.pkl" --tag=nightly
```


### On PR
```
python -u run_bench_interp.py "output/$(date "+%Y%m%d-%H%M%S")-pr.pkl" --tag=PR

python -u make_results_table_from_pickles.py output/$(date "+%Y%m%d-%H%M%S")-pr_vs_nightly.md output/XYZ-pr.pkl output/ABC-nightly.pkl

python -u perf_results_compute_speedup.py output/20230320-160044-pr_vs_nightly-speedup.md "['output/XYZ-pr.pkl', 'output/ABC-nightly.pkl']" --col1="torch (2.1.0a0+gitc005105) PR" --col2="torch (2.1.0a0+git5309c44) nightly" --description="Speed-up: PR vs nightly"
```

```
python -u run_bench_interp.py "output/$(date "+%Y%m%d-%H%M%S")-pr.pkl" --tag=PR --with-torchvision
```



## Output consistency with master pytorch

```
pip install fire

# On pytorch-nightly or 1.13.1
python verif_interp2.py verif_expected --is_ref=True

# On PR
python verif_interp2.py verif_expected --is_ref=False
```

## Some results

### 22/03/2023

Compare `_write_endline_rgb_as_uint32` 3 memcpy vs 1 uint8 set + 1 memcpy

- 3 memcpy
```
Num threads: 1

PIL version:  9.0.0.post1
[----------------------------------------------------------- Resize -----------------------------------------------------------]
                                                                       |  Pillow (9.0.0.post1)  |  torch (2.1.0a0+git8d955df) PR
1 threads: ---------------------------------------------------------------------------------------------------------------------
      3 torch.uint8 channels_last bilinear 256 -> (224, 224) aa=True   |         128.4          |              153.8
      3 torch.uint8 channels_last bilinear 256 -> (32, 32) aa=True     |          38.5          |               58.2
      3 torch.uint8 channels_last bilinear 256 -> (256, 224) aa=True   |          91.2          |              114.5
      3 torch.uint8 channels_last bilinear 256 -> (256, 227) aa=True   |          93.7          |              123.8
      3 torch.uint8 channels_last bilinear 256 -> (224, 256) aa=True   |          52.2          |               52.0
      3 torch.uint8 channels_last bilinear 256 -> (227, 256) aa=True   |          57.2          |               52.6
      3 torch.uint8 channels_first bilinear 256 -> (224, 224) aa=True  |         128.4          |              311.7
      3 torch.uint8 channels_first bilinear 256 -> (32, 32) aa=True    |          38.6          |              131.7
      3 torch.uint8 channels_first bilinear 256 -> (256, 224) aa=True  |          90.8          |              283.3
      3 torch.uint8 channels_first bilinear 256 -> (256, 227) aa=True  |          92.2          |              286.7
      3 torch.uint8 channels_first bilinear 256 -> (224, 256) aa=True  |          52.3          |              233.3
      3 torch.uint8 channels_first bilinear 256 -> (227, 256) aa=True  |          52.1          |              235.4

Times are in microseconds (us).
```


- 1 memcpy + 1 uint8 set
```
Num threads: 1

PIL version:  9.0.0.post1
[----------------------------------------------------------- Resize -----------------------------------------------------------]
                                                                       |  Pillow (9.0.0.post1)  |  torch (2.1.0a0+git8d955df) PR
1 threads: ---------------------------------------------------------------------------------------------------------------------
      3 torch.uint8 channels_last bilinear 256 -> (224, 224) aa=True   |         128.1          |              154.0
      3 torch.uint8 channels_last bilinear 256 -> (32, 32) aa=True     |          38.5          |               55.3
      3 torch.uint8 channels_last bilinear 256 -> (256, 224) aa=True   |          91.5          |              115.6
      3 torch.uint8 channels_last bilinear 256 -> (256, 227) aa=True   |          93.1          |              125.4
      3 torch.uint8 channels_last bilinear 256 -> (224, 256) aa=True   |          52.2          |               52.0
      3 torch.uint8 channels_last bilinear 256 -> (227, 256) aa=True   |          52.9          |               52.4
      3 torch.uint8 channels_first bilinear 256 -> (224, 224) aa=True  |         129.6          |              300.6
      3 torch.uint8 channels_first bilinear 256 -> (32, 32) aa=True    |          38.3          |              130.8
      3 torch.uint8 channels_first bilinear 256 -> (256, 224) aa=True  |          92.4          |              267.9
      3 torch.uint8 channels_first bilinear 256 -> (256, 227) aa=True  |          92.5          |              270.9
      3 torch.uint8 channels_first bilinear 256 -> (224, 256) aa=True  |          52.1          |              218.0
      3 torch.uint8 channels_first bilinear 256 -> (227, 256) aa=True  |          52.3          |              218.8

Times are in microseconds (us).
```

### 16/03/2023

#### Results on Nightly

```
export fileprefix=$(date "+%Y%m%d-%H%M%S") && python -u run_bench_interp2.py output/${fileprefix}-pt2.0.pkl --tag=PT2.0 --with_torchvision &> output/${fileprefix}-pt2.0.log

python perf_results_compute_speedup.py output/20230316-155841-pt2.0_vs_torchvision_speedup.md "['output/20230316-155841-pt2.0.pkl',]" --col1="torch (2.1.0a0+git5309c44) PT2.0" --col2="torchvision resize" --description="Speed-up: PTH vs TV"
```

```
Timestamp: 20230316-155843
Torch version: 2.1.0a0+git5309c44
Torch config: PyTorch built with:
  - GCC 9.4
  - C++ Version: 201703
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - CPU capability usage: AVX2
  - Build settings: BUILD_TYPE=Release, CXX_COMPILER=/usr/bin/c++, CXX_FLAGS= -D_GLIBCXX_USE_CXX11_ABI=1 -Wno-deprecated -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -DNDEBUG -DUSE_KINETO -DLIBKINETO_NOCUPTI -DLIBKINETO_NOROCTRACER -DUSE_PYTORCH_QNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -O2 -fPIC -Wall -Wextra -Werror=return-type -Werror=non-virtual-dtor -Werror=bool-operation -Wnarrowing -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-unused-parameter -Wno-unused-function -Wno-unused-result -Wno-strict-overflow -Wno-strict-aliasing -Wno-error=deprecated-declarations -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=old-style-cast -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Werror=cast-function-type -Wno-stringop-overflow, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, TORCH_DISABLE_GPU_ASSERTS=ON, TORCH_VERSION=2.1.0, USE_CUDA=0, USE_CUDNN=OFF, USE_EIGEN_FOR_BLAS=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=OFF, USE_MKLDNN=0, USE_MPI=OFF, USE_NCCL=OFF, USE_NNPACK=0, USE_OPENMP=ON, USE_ROCM=OFF,

Num threads: 1

[----------------------------------------------------------------------------------------- Resize -----------------------------------------------------------------------------------------]
                                                                                 |  Pillow (9.0.0.post1)  |  torch (2.1.0a0+git5309c44) PT2.0  |  torchvision resize  |  Speed-up: PTH vs TV
1 threads: ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      3 torch.uint8 channels_last bilinear (64, 64) -> (224, 224) aa=True        |          61.2          |                166.6               |         823.3        |          4.9
      3 torch.uint8 channels_last bilinear (224, 224) -> (270, 268) aa=True      |         133.0          |                333.9               |        1455.2        |          4.4
      3 torch.uint8 channels_last bilinear (256, 256) -> (1024, 1024) aa=True    |         945.2          |               2769.7               |       24002.0        |          8.7
      3 torch.uint8 channels_last bilinear (224, 224) -> (64, 64) aa=True        |          52.3          |                137.5               |         374.7        |          2.7
      3 torch.uint8 channels_last bilinear (270, 268) -> (224, 224) aa=True      |         138.7          |                324.5               |        1248.8        |          3.8
      3 torch.uint8 channels_last bilinear (1024, 1024) -> (256, 256) aa=True    |         696.7          |               2042.6               |        9597.2        |          4.7
      3 torch.uint8 channels_last bilinear (64, 64) -> (224, 224) aa=False       |                        |                167.4               |         873.2        |          5.2
      3 torch.uint8 channels_last bilinear (224, 224) -> (270, 268) aa=False     |                        |                331.2               |        1270.2        |          3.8
      3 torch.uint8 channels_last bilinear (256, 256) -> (1024, 1024) aa=False   |                        |               2735.2               |       19251.7        |          7.0
      3 torch.uint8 channels_last bilinear (224, 224) -> (64, 64) aa=False       |                        |                112.4               |         119.2        |          1.1
      3 torch.uint8 channels_last bilinear (270, 268) -> (224, 224) aa=False     |                        |                299.0               |         908.5        |          3.0
      3 torch.uint8 channels_last bilinear (1024, 1024) -> (256, 256) aa=False   |                        |               1700.0               |        2132.7        |          1.3
      3 torch.uint8 channels_first bilinear (64, 64) -> (224, 224) aa=True       |          60.8          |                209.9               |         405.0        |          1.9
      3 torch.uint8 channels_first bilinear (224, 224) -> (270, 268) aa=True     |         132.6          |                388.2               |         846.1        |          2.2
      3 torch.uint8 channels_first bilinear (256, 256) -> (1024, 1024) aa=True   |         944.1          |               3141.2               |        8627.0        |          2.7
      3 torch.uint8 channels_first bilinear (224, 224) -> (64, 64) aa=True       |          52.5          |                138.9               |         337.2        |          2.4
      3 torch.uint8 channels_first bilinear (270, 268) -> (224, 224) aa=True     |         139.2          |                335.8               |         823.6        |          2.5
      3 torch.uint8 channels_first bilinear (1024, 1024) -> (256, 256) aa=True   |         691.2          |               2090.5               |        6692.3        |          3.2
      3 torch.uint8 channels_first bilinear (64, 64) -> (224, 224) aa=False      |                        |                208.0               |        1053.6        |          5.1
      3 torch.uint8 channels_first bilinear (224, 224) -> (270, 268) aa=False    |                        |                388.0               |        1606.3        |          4.1
      3 torch.uint8 channels_first bilinear (256, 256) -> (1024, 1024) aa=False  |                        |               3091.6               |       24532.5        |          7.9
      3 torch.uint8 channels_first bilinear (224, 224) -> (64, 64) aa=False      |                        |                115.1               |         231.9        |          2.0
      3 torch.uint8 channels_first bilinear (270, 268) -> (224, 224) aa=False    |                        |                342.4               |        1214.5        |          3.5
      3 torch.uint8 channels_first bilinear (1024, 1024) -> (256, 256) aa=False  |                        |               1753.7               |        4988.4        |          2.8

Times are in microseconds (us).

```

```
export fileprefix=$(date "+%Y%m%d-%H%M%S") && ATEN_CPU_CAPABILITY=default python -u run_bench_interp2.py output/${fileprefix}-default-cpu_cap-pt2.0.pkl --tag=PT2.0 --with_torchvision &> output/${fileprefix}-pt2.0.log

python perf_results_compute_speedup.py output/20230316-162354-pt2.0_vs_torchvision_speedup.md "['output/20230316-162354-default-cpu-cap-pt2.0.pkl',]" --col1="torch (2.1.0a0+git5309c44) PT2.0" --col2="torchvision resize" --description="Speed-up: PTH vs TV"
```

```
Timestamp: 20230316-162355
Torch version: 2.1.0a0+git5309c44
Torch config: PyTorch built with:
  - GCC 9.4
  - C++ Version: 201703
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - CPU capability usage: NO AVX
  - Build settings: BUILD_TYPE=Release, CXX_COMPILER=/usr/bin/c++, CXX_FLAGS= -D_GLIBCXX_USE_CXX11_ABI=1 -Wno-deprecated -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -DNDEBUG -DUSE_KINETO -DLIBKINETO_NOCUPTI -DLIBKINETO_NOROCTRACER -DUSE_PYTORCH_QNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -O2 -fPIC -Wall -Wextra -Werror=return-type -Werror=non-virtual-dtor -Werror=bool-operation -Wnarrowing -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-unused-parameter -Wno-unused-function -Wno-unused-result -Wno-strict-overflow -Wno-strict-aliasing -Wno-error=deprecated-declarations -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=old-style-cast -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Werror=cast-function-type -Wno-stringop-overflow, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, TORCH_DISABLE_GPU_ASSERTS=ON, TORCH_VERSION=2.1.0, USE_CUDA=0, USE_CUDNN=OFF, USE_EIGEN_FOR_BLAS=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=OFF, USE_MKLDNN=0, USE_MPI=OFF, USE_NCCL=OFF, USE_NNPACK=0, USE_OPENMP=ON, USE_ROCM=OFF,

Num threads: 1

PIL version:  9.0.0.post1

[----------------------------------------------------------------------------------------- Resize -----------------------------------------------------------------------------------------]
                                                                                 |  Pillow (9.0.0.post1)  |  torch (2.1.0a0+git5309c44) PT2.0  |  torchvision resize  |  Speed-up: PTH vs TV
1 threads: ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      3 torch.uint8 channels_last bilinear (64, 64) -> (224, 224) aa=True        |           60.8         |                985.0               |        1143.2        |          1.2
      3 torch.uint8 channels_last bilinear (224, 224) -> (270, 268) aa=True      |          133.5         |               1865.2               |        1962.5        |          1.1
      3 torch.uint8 channels_last bilinear (256, 256) -> (1024, 1024) aa=True    |         1031.8         |              19498.8               |       31670.5        |          1.6
      3 torch.uint8 channels_last bilinear (224, 224) -> (64, 64) aa=True        |           52.4         |                451.5               |         472.2        |          1.0
      3 torch.uint8 channels_last bilinear (270, 268) -> (224, 224) aa=True      |          138.9         |               1617.4               |        1649.1        |          1.0
      3 torch.uint8 channels_last bilinear (1024, 1024) -> (256, 256) aa=True    |          690.2         |               7913.4               |        8451.1        |          1.1
      3 torch.uint8 channels_last bilinear (64, 64) -> (224, 224) aa=False       |                        |                982.3               |        1011.7        |          1.0
      3 torch.uint8 channels_last bilinear (224, 224) -> (270, 268) aa=False     |                        |               1861.8               |        1513.1        |          0.8
      3 torch.uint8 channels_last bilinear (256, 256) -> (1024, 1024) aa=False   |                        |              19482.3               |       20998.5        |          1.1
      3 torch.uint8 channels_last bilinear (224, 224) -> (64, 64) aa=False       |                        |                270.4               |         182.5        |          0.7
      3 torch.uint8 channels_last bilinear (270, 268) -> (224, 224) aa=False     |                        |               1527.6               |        1115.5        |          0.7
      3 torch.uint8 channels_last bilinear (1024, 1024) -> (256, 256) aa=False   |                        |               4131.4               |        2967.0        |          0.7
      3 torch.uint8 channels_first bilinear (64, 64) -> (224, 224) aa=True       |           61.0         |                629.9               |         682.8        |          1.1
      3 torch.uint8 channels_first bilinear (224, 224) -> (270, 268) aa=True     |          133.9         |               1356.5               |        1298.8        |          1.0
      3 torch.uint8 channels_first bilinear (256, 256) -> (1024, 1024) aa=True   |          945.5         |              11945.5               |       13136.1        |          1.1
      3 torch.uint8 channels_first bilinear (224, 224) -> (64, 64) aa=True       |           52.7         |                420.7               |         434.9        |          1.0
      3 torch.uint8 channels_first bilinear (270, 268) -> (224, 224) aa=True     |          138.9         |               1263.9               |        1187.5        |          0.9
      3 torch.uint8 channels_first bilinear (1024, 1024) -> (256, 256) aa=True   |          691.9         |               7423.7               |        7661.4        |          1.0
      3 torch.uint8 channels_first bilinear (64, 64) -> (224, 224) aa=False      |                        |                628.9               |        1190.9        |          1.9
      3 torch.uint8 channels_first bilinear (224, 224) -> (270, 268) aa=False    |                        |               1354.9               |        1846.4        |          1.4
      3 torch.uint8 channels_first bilinear (256, 256) -> (1024, 1024) aa=False  |                        |              11961.0               |       27052.5        |          2.3
      3 torch.uint8 channels_first bilinear (224, 224) -> (64, 64) aa=False      |                        |                239.1               |         295.3        |          1.2
      3 torch.uint8 channels_first bilinear (270, 268) -> (224, 224) aa=False    |                        |               1174.4               |        1416.4        |          1.2
      3 torch.uint8 channels_first bilinear (1024, 1024) -> (256, 256) aa=False  |                        |               3659.3               |       13981.4        |          3.8

Times are in microseconds (us).

```


#### Results on PR

```
export fileprefix=$(date "+%Y%m%d-%H%M%S") && ATEN_CPU_CAPABILITY=default python -u run_bench_interp2.py output/${fileprefix}-default-cpu_cap-pr.pkl --tag=PR --with_torchvision &> output/${fileprefix}-pr.log

python perf_results_compute_speedup.py output/20230316-164209-pr_vs_torchvision_speedup.md "['output/20230316-164209-default-cpu_cap-pr.pkl',]" --col1="torch (2.1.0a0+git0968a5d) PR" --col2="torchvision resize" --description="Speed-up: PTH vs TV"
```

```
...
```

### 02/03/2023

- AVX code update to handle RGB without copying to RGBA

```
PIL version:  9.0.0.post1
[-------------------------------------------------------------------- Resize --------------------------------------------------------------------]
                                                                 |  Pillow (9.0.0.post1)  |  torch (2.0.0a0+git8d22fc6) PR  |   torchvision resize
1 threads: ---------------------------------------------------------------------------------------------------------------------------------------
      3 torch.uint8 channels_last bilinear 256 -> 32 aa=True     |    38.494 (+-0.147)    |         92.690 (+-0.469)        |   365.949 (+-0.733)
      3 torch.uint8 channels_last bilinear 256 -> 32 aa=False    |                        |         69.978 (+-0.149)        |    74.216 (+-0.254)
      3 torch.uint8 channels_last bilinear 520 -> 32 aa=True     |   112.797 (+-0.664)    |        199.637 (+-1.008)        |   1562.155 (+-5.665)
      3 torch.uint8 channels_last bilinear 520 -> 32 aa=False    |                        |        120.906 (+-0.916)        |   187.384 (+-1.056)
      3 torch.uint8 channels_last bilinear 712 -> 32 aa=True     |   186.492 (+-4.095)    |        298.649 (+-1.025)        |  2969.693 (+-20.888)
      3 torch.uint8 channels_last bilinear 712 -> 32 aa=False    |                        |        157.991 (+-0.577)        |   317.267 (+-0.414)
      3 torch.uint8 channels_last bilinear 270 -> 224 aa=True    |   139.807 (+-0.900)    |        428.483 (+-2.282)        |   1243.928 (+-9.034)
      3 torch.uint8 channels_last bilinear 270 -> 224 aa=False   |                        |        413.357 (+-2.621)        |   920.346 (+-6.303)
      4 torch.uint8 channels_last bilinear 256 -> 32 aa=True     |                        |         71.099 (+-0.621)        |   469.975 (+-0.832)
      4 torch.uint8 channels_last bilinear 256 -> 32 aa=False    |                        |         71.985 (+-0.352)        |    87.489 (+-0.232)
      4 torch.uint8 channels_last bilinear 520 -> 32 aa=True     |                        |        178.459 (+-0.538)        |   2077.379 (+-6.035)
      4 torch.uint8 channels_last bilinear 520 -> 32 aa=False    |                        |        125.413 (+-0.595)        |   238.725 (+-0.642)
      4 torch.uint8 channels_last bilinear 712 -> 32 aa=True     |                        |        273.980 (+-6.355)        |  3928.220 (+-30.252)
      4 torch.uint8 channels_last bilinear 712 -> 32 aa=False    |                        |        165.420 (+-0.632)        |   422.717 (+-1.076)
      4 torch.uint8 channels_last bilinear 270 -> 224 aa=True    |                        |        444.087 (+-5.410)        |  1493.987 (+-46.857)
      4 torch.uint8 channels_last bilinear 270 -> 224 aa=False   |                        |        425.220 (+-2.867)        |   999.073 (+-10.833)
      3 torch.uint8 channels_first bilinear 256 -> 32 aa=True    |    38.582 (+-0.115)    |        147.970 (+-0.303)        |   353.628 (+-1.016)
      3 torch.uint8 channels_first bilinear 256 -> 32 aa=False   |                        |        149.710 (+-3.397)        |   203.966 (+-0.399)
      3 torch.uint8 channels_first bilinear 520 -> 32 aa=True    |   112.984 (+-0.439)    |        486.716 (+-1.509)        |   1545.843 (+-2.481)
      3 torch.uint8 channels_first bilinear 520 -> 32 aa=False   |                        |        433.658 (+-2.452)        |   683.053 (+-10.833)
      3 torch.uint8 channels_first bilinear 712 -> 32 aa=True    |   186.904 (+-0.621)    |        848.435 (+-2.495)        |  2909.753 (+-17.134)
      3 torch.uint8 channels_first bilinear 712 -> 32 aa=False   |                        |        739.664 (+-2.719)        |   1415.403 (+-4.448)
      3 torch.uint8 channels_first bilinear 270 -> 224 aa=True   |   140.154 (+-0.353)    |        604.671 (+-1.417)        |   814.513 (+-1.964)
      3 torch.uint8 channels_first bilinear 270 -> 224 aa=False  |                        |        586.507 (+-1.497)        |   1221.633 (+-5.803)
      4 torch.uint8 channels_first bilinear 256 -> 32 aa=True    |                        |         90.029 (+-0.464)        |   457.867 (+-1.191)
      4 torch.uint8 channels_first bilinear 256 -> 32 aa=False   |                        |         91.796 (+-0.552)        |   251.914 (+-0.726)
      4 torch.uint8 channels_first bilinear 520 -> 32 aa=True    |                        |        246.137 (+-0.816)        |   2049.130 (+-6.283)
      4 torch.uint8 channels_first bilinear 520 -> 32 aa=False   |                        |        193.739 (+-0.453)        |   936.896 (+-4.979)
      4 torch.uint8 channels_first bilinear 712 -> 32 aa=True    |                        |        398.765 (+-1.025)        |  3893.657 (+-22.681)
      4 torch.uint8 channels_first bilinear 712 -> 32 aa=False   |                        |        289.966 (+-0.439)        |  1863.072 (+-13.275)
      4 torch.uint8 channels_first bilinear 270 -> 224 aa=True   |                        |        564.773 (+-11.819)       |   1070.920 (+-3.889)
      4 torch.uint8 channels_first bilinear 270 -> 224 aa=False  |                        |        546.738 (+-1.310)        |   1161.150 (+-4.633)

Times are in microseconds (us).
```

- Nightly
```
Num threads: 1

PIL version:  9.0.0.post1
[----------------------------------------------------------- Resize ----------------------------------------------------------]
                                                                 |  Pillow (9.0.0.post1)  |  torch (2.1.0a0+git5309c44) nightly
1 threads: --------------------------------------------------------------------------------------------------------------------
      3 torch.uint8 channels_last bilinear 256 -> 32 aa=True     |    38.698 (+-0.253)    |          132.323 (+-0.976)
      3 torch.uint8 channels_last bilinear 256 -> 32 aa=False    |                        |          111.236 (+-1.079)
      3 torch.uint8 channels_last bilinear 520 -> 32 aa=True     |   112.887 (+-0.896)    |          442.878 (+-2.618)
      3 torch.uint8 channels_last bilinear 520 -> 32 aa=False    |                        |          363.400 (+-1.934)
      3 torch.uint8 channels_last bilinear 712 -> 32 aa=True     |   185.901 (+-1.760)    |          784.421 (+-3.588)
      3 torch.uint8 channels_last bilinear 712 -> 32 aa=False    |                        |          647.022 (+-3.552)
      3 torch.uint8 channels_last bilinear 270 -> 224 aa=True    |   138.670 (+-0.821)    |          312.605 (+-9.297)
      3 torch.uint8 channels_last bilinear 270 -> 224 aa=False   |                        |          290.585 (+-4.413)
      4 torch.uint8 channels_last bilinear 256 -> 32 aa=True     |                        |           55.064 (+-0.520)
      4 torch.uint8 channels_last bilinear 256 -> 32 aa=False    |                        |           34.352 (+-0.172)
      4 torch.uint8 channels_last bilinear 520 -> 32 aa=True     |                        |          133.909 (+-1.064)
      4 torch.uint8 channels_last bilinear 520 -> 32 aa=False    |                        |           54.917 (+-0.755)
      4 torch.uint8 channels_last bilinear 712 -> 32 aa=True     |                        |          210.390 (+-3.139)
      4 torch.uint8 channels_last bilinear 712 -> 32 aa=False    |                        |           71.756 (+-0.731)
      4 torch.uint8 channels_last bilinear 270 -> 224 aa=True    |                        |          152.521 (+-1.761)
      4 torch.uint8 channels_last bilinear 270 -> 224 aa=False   |                        |          130.649 (+-1.364)
      3 torch.uint8 channels_first bilinear 256 -> 32 aa=True    |    38.841 (+-0.239)    |          132.497 (+-1.526)
      3 torch.uint8 channels_first bilinear 256 -> 32 aa=False   |                        |          112.043 (+-1.116)
      3 torch.uint8 channels_first bilinear 520 -> 32 aa=True    |   112.429 (+-1.484)    |          442.762 (+-9.324)
      3 torch.uint8 channels_first bilinear 520 -> 32 aa=False   |                        |          363.378 (+-3.789)
      3 torch.uint8 channels_first bilinear 712 -> 32 aa=True    |   187.229 (+-2.536)    |          786.058 (+-8.815)
      3 torch.uint8 channels_first bilinear 712 -> 32 aa=False   |                        |          648.510 (+-4.419)
      3 torch.uint8 channels_first bilinear 270 -> 224 aa=True   |   138.505 (+-1.135)    |          336.899 (+-4.264)
      3 torch.uint8 channels_first bilinear 270 -> 224 aa=False  |                        |          310.388 (+-7.450)
      4 torch.uint8 channels_first bilinear 256 -> 32 aa=True    |                        |           73.812 (+-0.255)
      4 torch.uint8 channels_first bilinear 256 -> 32 aa=False   |                        |           53.240 (+-0.193)
      4 torch.uint8 channels_first bilinear 520 -> 32 aa=True    |                        |          201.602 (+-1.954)
      4 torch.uint8 channels_first bilinear 520 -> 32 aa=False   |                        |          122.700 (+-0.608)
      4 torch.uint8 channels_first bilinear 712 -> 32 aa=True    |                        |          335.420 (+-1.667)
      4 torch.uint8 channels_first bilinear 712 -> 32 aa=False   |                        |          197.773 (+-0.729)
      4 torch.uint8 channels_first bilinear 270 -> 224 aa=True   |                        |          295.821 (+-2.399)
      4 torch.uint8 channels_first bilinear 270 -> 224 aa=False  |                        |          261.934 (+-4.472)

Times are in microseconds (us).
```


### 08/02/2023

```

PIL version:  9.0.0.post1
[--------------------------------------------------------------------- Resize --------------------------------------------------------------------]
                                                                 |  Pillow (9.0.0.post1)  |  torch (2.0.0a0+gite6bdca1) PR  |   torchvision resize
1 threads: ----------------------------------------------------------------------------------------------------------------------------------------
      3 torch.uint8 channels_last bilinear 256 -> 32 aa=True     |    38.074 (+-0.541)    |        145.393 (+-1.645)        |   368.952 (+-1.475)
      3 torch.uint8 channels_last bilinear 256 -> 32 aa=False    |                        |        112.422 (+-0.668)        |    74.104 (+-0.135)
      3 torch.uint8 channels_last bilinear 520 -> 32 aa=True     |   112.459 (+-0.690)    |        496.323 (+-3.041)        |   1560.163 (+-5.492)
      3 torch.uint8 channels_last bilinear 520 -> 32 aa=False    |                        |        363.923 (+-1.618)        |   186.801 (+-0.461)
      3 torch.uint8 channels_last bilinear 712 -> 32 aa=True     |   184.901 (+-1.414)    |        890.993 (+-2.717)        |  2949.707 (+-95.723)
      3 torch.uint8 channels_last bilinear 712 -> 32 aa=False    |                        |        647.951 (+-4.457)        |   318.293 (+-0.674)
      3 torch.uint8 channels_last bilinear 270 -> 224 aa=True    |   139.299 (+-0.729)    |        329.859 (+-1.688)        |   1242.137 (+-4.737)
      3 torch.uint8 channels_last bilinear 270 -> 224 aa=False   |                        |        307.541 (+-2.314)        |   908.356 (+-1.292)
      4 torch.uint8 channels_last bilinear 256 -> 32 aa=True     |                        |         67.892 (+-0.188)        |   473.954 (+-0.875)
      4 torch.uint8 channels_last bilinear 256 -> 32 aa=False    |                        |         34.854 (+-0.124)        |    87.333 (+-0.212)
      4 torch.uint8 channels_last bilinear 520 -> 32 aa=True     |                        |        188.218 (+-1.294)        |   2064.724 (+-7.161)
      4 torch.uint8 channels_last bilinear 520 -> 32 aa=False    |                        |         55.389 (+-0.175)        |   238.161 (+-0.517)
      4 torch.uint8 channels_last bilinear 712 -> 32 aa=True     |                        |        316.895 (+-1.609)        |  3929.540 (+-11.386)
      4 torch.uint8 channels_last bilinear 712 -> 32 aa=False    |                        |         73.027 (+-0.227)        |   424.204 (+-1.261)
      4 torch.uint8 channels_last bilinear 270 -> 224 aa=True    |                        |        166.030 (+-0.901)        |   1489.629 (+-5.092)
      4 torch.uint8 channels_last bilinear 270 -> 224 aa=False   |                        |        143.489 (+-0.757)        |   992.293 (+-1.604)
      3 torch.uint8 channels_first bilinear 256 -> 32 aa=True    |    37.804 (+-0.178)    |        145.445 (+-0.874)        |   355.802 (+-1.438)
      3 torch.uint8 channels_first bilinear 256 -> 32 aa=False   |                        |        112.183 (+-0.704)        |   203.691 (+-0.742)
      3 torch.uint8 channels_first bilinear 520 -> 32 aa=True    |   112.137 (+-0.763)    |        496.563 (+-11.028)       |  1549.939 (+-10.290)
      3 torch.uint8 channels_first bilinear 520 -> 32 aa=False   |                        |        364.179 (+-2.418)        |   678.691 (+-2.422)
      3 torch.uint8 channels_first bilinear 712 -> 32 aa=True    |   184.557 (+-1.122)    |        891.174 (+-3.050)        |  2930.927 (+-11.987)
      3 torch.uint8 channels_first bilinear 712 -> 32 aa=False   |                        |        647.634 (+-1.752)        |  1287.492 (+-877.768)
      3 torch.uint8 channels_first bilinear 270 -> 224 aa=True   |   139.091 (+-1.009)    |        329.487 (+-1.858)        |   818.238 (+-1.593)
      3 torch.uint8 channels_first bilinear 270 -> 224 aa=False  |                        |        308.697 (+-2.485)        |   1209.505 (+-4.367)
      4 torch.uint8 channels_first bilinear 256 -> 32 aa=True    |                        |         87.350 (+-0.238)        |   460.749 (+-1.384)
      4 torch.uint8 channels_first bilinear 256 -> 32 aa=False   |                        |         53.891 (+-0.200)        |   252.033 (+-0.734)
      4 torch.uint8 channels_first bilinear 520 -> 32 aa=True    |                        |        257.106 (+-1.468)        |   2052.175 (+-6.119)
      4 torch.uint8 channels_first bilinear 520 -> 32 aa=False   |                        |        124.054 (+-0.658)        |  909.929 (+-272.601)
      4 torch.uint8 channels_first bilinear 712 -> 32 aa=True    |                        |        442.343 (+-2.452)        |  3904.617 (+-11.139)
      4 torch.uint8 channels_first bilinear 712 -> 32 aa=False   |                        |        199.340 (+-1.232)        |  4785.501 (+-106.702)
      4 torch.uint8 channels_first bilinear 270 -> 224 aa=True   |                        |        285.454 (+-3.306)        |   1073.443 (+-6.514)
      4 torch.uint8 channels_first bilinear 270 -> 224 aa=False  |                        |        264.808 (+-3.390)        |   1157.429 (+-4.480)

Times are in microseconds (us).
```

### 07/02/2023

```
Num threads: 1

PIL version:  9.0.0.post1
[-------------------------------------------------------------------- Resize --------------------------------------------------------------------]
                                                                 |  Pillow (9.0.0.post1)  |  torch (2.0.0a0+gite6bdca1) PR  |   torchvision resize
1 threads: ---------------------------------------------------------------------------------------------------------------------------------------
      3 torch.uint8 channels_last bilinear 256 -> 32 aa=True     |    38.945 (+-0.222)    |        130.363 (+-0.612)        |   364.956 (+-2.782)
      3 torch.uint8 channels_last bilinear 256 -> 32 aa=False    |                        |        108.715 (+-0.305)        |    72.821 (+-0.245)
      3 torch.uint8 channels_last bilinear 520 -> 32 aa=True     |   112.800 (+-0.394)    |        439.170 (+-0.637)        |   1596.292 (+-2.404)
      3 torch.uint8 channels_last bilinear 520 -> 32 aa=False    |                        |        360.557 (+-0.442)        |   185.144 (+-0.231)
      3 torch.uint8 channels_last bilinear 712 -> 32 aa=True     |   186.025 (+-0.873)    |        781.784 (+-4.723)        |   2941.418 (+-3.263)
      3 torch.uint8 channels_last bilinear 712 -> 32 aa=False    |                        |        643.826 (+-2.035)        |   316.546 (+-0.371)
      3 torch.uint8 channels_last bilinear 270 -> 224 aa=True    |   139.784 (+-0.302)    |        319.836 (+-1.226)        |   1238.219 (+-1.816)
      3 torch.uint8 channels_last bilinear 270 -> 224 aa=False   |                        |        297.607 (+-3.890)        |   908.446 (+-1.849)
      4 torch.uint8 channels_last bilinear 256 -> 32 aa=True     |                        |         52.814 (+-0.490)        |   470.149 (+-7.399)
      4 torch.uint8 channels_last bilinear 256 -> 32 aa=False    |                        |         31.900 (+-0.115)        |    86.144 (+-0.203)
      4 torch.uint8 channels_last bilinear 520 -> 32 aa=True     |                        |        131.809 (+-1.086)        |   2099.700 (+-3.938)
      4 torch.uint8 channels_last bilinear 520 -> 32 aa=False    |                        |         52.489 (+-0.074)        |   236.924 (+-0.330)
      4 torch.uint8 channels_last bilinear 712 -> 32 aa=True     |                        |        207.632 (+-1.031)        |   3934.327 (+-4.734)
      4 torch.uint8 channels_last bilinear 712 -> 32 aa=False    |                        |         69.291 (+-0.169)        |   422.172 (+-0.420)
      4 torch.uint8 channels_last bilinear 270 -> 224 aa=True    |                        |        149.362 (+-0.545)        |   1484.460 (+-3.488)
      4 torch.uint8 channels_last bilinear 270 -> 224 aa=False   |                        |        127.503 (+-0.296)        |   992.280 (+-1.658)
      3 torch.uint8 channels_first bilinear 256 -> 32 aa=True    |    38.934 (+-0.066)    |        130.096 (+-0.442)        |   352.259 (+-0.315)
      3 torch.uint8 channels_first bilinear 256 -> 32 aa=False   |                        |        108.973 (+-0.755)        |   201.381 (+-0.268)
      3 torch.uint8 channels_first bilinear 520 -> 32 aa=True    |   112.429 (+-0.337)    |        439.524 (+-2.111)        |   1582.000 (+-2.005)
      3 torch.uint8 channels_first bilinear 520 -> 32 aa=False   |                        |        360.747 (+-0.596)        |   707.031 (+-4.563)
      3 torch.uint8 channels_first bilinear 712 -> 32 aa=True    |   186.654 (+-0.514)    |        781.276 (+-2.859)        |   2929.456 (+-7.766)
      3 torch.uint8 channels_first bilinear 712 -> 32 aa=False   |                        |        643.654 (+-2.010)        |  1436.077 (+-45.418)
      3 torch.uint8 channels_first bilinear 270 -> 224 aa=True   |   140.697 (+-0.760)    |        318.995 (+-1.630)        |   814.211 (+-2.972)
      3 torch.uint8 channels_first bilinear 270 -> 224 aa=False  |                        |        295.188 (+-1.540)        |   1208.328 (+-1.880)
      4 torch.uint8 channels_first bilinear 256 -> 32 aa=True    |                        |         71.006 (+-0.246)        |   456.845 (+-1.242)
      4 torch.uint8 channels_first bilinear 256 -> 32 aa=False   |                        |         50.860 (+-0.104)        |   249.063 (+-0.477)
      4 torch.uint8 channels_first bilinear 520 -> 32 aa=True    |                        |        199.646 (+-0.859)        |   2091.296 (+-2.892)
      4 torch.uint8 channels_first bilinear 520 -> 32 aa=False   |                        |        120.245 (+-0.589)        |   950.757 (+-17.017)
      4 torch.uint8 channels_first bilinear 712 -> 32 aa=True    |                        |        330.196 (+-1.010)        |   3908.203 (+-4.160)
      4 torch.uint8 channels_first bilinear 712 -> 32 aa=False   |                        |        194.640 (+-0.230)        |   4760.218 (+-9.555)
      4 torch.uint8 channels_first bilinear 270 -> 224 aa=True   |                        |        266.951 (+-1.631)        |   1069.409 (+-4.428)
      4 torch.uint8 channels_first bilinear 270 -> 224 aa=False  |                        |        243.640 (+-0.885)        |   1163.805 (+-1.581)

Times are in microseconds (us).
```

### 02/02/2023

- Removed pointer from upsample_avx_bilinear
- Avoid copy if num_channels=4 and channels_last

```
Num threads: 1

PIL version:  9.0.0.post1
[----------------------------------------------------------------------------- Resize ----------------------------------------------------------------------------]
                                                                |  Pillow (9.0.0.post1)  |  torch (2.0.0a0+git7f72623) PR  |  torch (2.0.0a0+git7f72623) PR (float)
1 threads: --------------------------------------------------------------------------------------------------------------------------------------------------------
      3 torch.uint8 channels_last bilinear 256 -> 32 aa=True    |          38.6          |               395.9             |                   360.7
      3 torch.uint8 channels_last bilinear 256 -> 32 aa=False   |                        |               363.4             |                    68.2
      3 torch.uint8 channels_last bilinear 520 -> 32 aa=True    |         112.5          |              1530.6             |                  1555.5
      3 torch.uint8 channels_last bilinear 520 -> 32 aa=False   |                        |              1369.7             |                   179.9
      3 torch.uint8 channels_last bilinear 712 -> 32 aa=True    |         186.0          |              2652.6             |                  2935.8
      3 torch.uint8 channels_last bilinear 712 -> 32 aa=False   |                        |              2507.9             |                   309.7
      4 torch.uint8 channels_last bilinear 256 -> 32 aa=True    |                        |                57.1             |                   466.0
      4 torch.uint8 channels_last bilinear 256 -> 32 aa=False   |                        |                37.6             |                    81.1
      4 torch.uint8 channels_last bilinear 520 -> 32 aa=True    |                        |               131.6             |                  2093.5
      4 torch.uint8 channels_last bilinear 520 -> 32 aa=False   |                        |                58.0             |                   231.4
      4 torch.uint8 channels_last bilinear 712 -> 32 aa=True    |                        |               204.7             |                  3926.6
      4 torch.uint8 channels_last bilinear 712 -> 32 aa=False   |                        |                74.7             |                   418.0
      3 torch.uint8 channels_first bilinear 256 -> 32 aa=True   |          38.7          |               397.9             |                   348.7
      3 torch.uint8 channels_first bilinear 256 -> 32 aa=False  |                        |               361.9             |                   197.9
      3 torch.uint8 channels_first bilinear 520 -> 32 aa=True   |         112.2          |              1448.7             |                  1540.7
      3 torch.uint8 channels_first bilinear 520 -> 32 aa=False  |                        |              1388.4             |                  1493.0
      3 torch.uint8 channels_first bilinear 712 -> 32 aa=True   |         186.0          |              2633.9             |                  2923.4
      3 torch.uint8 channels_first bilinear 712 -> 32 aa=False  |                        |              2585.5             |                  1271.8
      4 torch.uint8 channels_first bilinear 256 -> 32 aa=True   |                        |               208.1             |                   453.3
      4 torch.uint8 channels_first bilinear 256 -> 32 aa=False  |                        |               188.8             |                   245.1
      4 torch.uint8 channels_first bilinear 520 -> 32 aa=True   |                        |               748.0             |                  2043.2
      4 torch.uint8 channels_first bilinear 520 -> 32 aa=False  |                        |               673.7             |                   864.4
      4 torch.uint8 channels_first bilinear 712 -> 32 aa=True   |                        |              1362.0             |                  3897.1
      4 torch.uint8 channels_first bilinear 712 -> 32 aa=False  |                        |              1230.4             |                  1837.6

Times are in microseconds (us).
```

- recoded unpack rgb method

```
Num threads: 1

PIL version:  9.0.0.post1
[----------------------------------------------------------------------------- Resize ----------------------------------------------------------------------------]
                                                                |  Pillow (9.0.0.post1)  |  torch (2.0.0a0+git7f72623) PR  |  torch (2.0.0a0+git7f72623) PR (float)
1 threads: --------------------------------------------------------------------------------------------------------------------------------------------------------
      3 torch.uint8 channels_last bilinear 256 -> 32 aa=True    |          38.9          |              132.6              |                   360.5
      3 torch.uint8 channels_last bilinear 256 -> 32 aa=False   |                        |              111.6              |                    68.2
      3 torch.uint8 channels_last bilinear 520 -> 32 aa=True    |         113.7          |              443.0              |                  1554.3
      3 torch.uint8 channels_last bilinear 520 -> 32 aa=False   |                        |              362.6              |                   180.1
      3 torch.uint8 channels_last bilinear 712 -> 32 aa=True    |         187.5          |              784.7              |                  2904.1
      3 torch.uint8 channels_last bilinear 712 -> 32 aa=False   |                        |              645.3              |                   309.0
      4 torch.uint8 channels_last bilinear 256 -> 32 aa=True    |                        |               55.1              |                   464.7
      4 torch.uint8 channels_last bilinear 256 -> 32 aa=False   |                        |               33.5              |                    80.9
      4 torch.uint8 channels_last bilinear 520 -> 32 aa=True    |                        |              135.8              |                  2065.8
      4 torch.uint8 channels_last bilinear 520 -> 32 aa=False   |                        |               55.1              |                   231.5
      4 torch.uint8 channels_last bilinear 712 -> 32 aa=True    |                        |              209.8              |                  3873.7
      4 torch.uint8 channels_last bilinear 712 -> 32 aa=False   |                        |               71.3              |                   411.1
      3 torch.uint8 channels_first bilinear 256 -> 32 aa=True   |          39.2          |              132.5              |                   348.6
      3 torch.uint8 channels_first bilinear 256 -> 32 aa=False  |                        |              111.9              |                   199.4
      3 torch.uint8 channels_first bilinear 520 -> 32 aa=True   |         112.7          |              439.8              |                  1542.1
      3 torch.uint8 channels_first bilinear 520 -> 32 aa=False  |                        |              362.2              |                  1569.3
      3 torch.uint8 channels_first bilinear 712 -> 32 aa=True   |         185.4          |              779.8              |                  2888.7
      3 torch.uint8 channels_first bilinear 712 -> 32 aa=False  |                        |              645.0              |                  1440.4
      4 torch.uint8 channels_first bilinear 256 -> 32 aa=True   |                        |               73.9              |                   453.4
      4 torch.uint8 channels_first bilinear 256 -> 32 aa=False  |                        |               53.5              |                   245.7
      4 torch.uint8 channels_first bilinear 520 -> 32 aa=True   |                        |              200.3              |                  2041.5
      4 torch.uint8 channels_first bilinear 520 -> 32 aa=False  |                        |              122.8              |                   933.2
      4 torch.uint8 channels_first bilinear 712 -> 32 aa=True   |                        |              331.4              |                  3852.1
      4 torch.uint8 channels_first bilinear 712 -> 32 aa=False  |                        |              197.1              |                  2046.7

Times are in microseconds (us).
```

### 01/02/2023

- Use tensor to allocate memory
- Compute weights once

```
cd /tmp/pth/interpolate_vec_uint8/ && python -u check_interp.py
```

```
Torch version: 2.0.0a0+git7f72623Torch config: PyTorch built with:  - GCC 9.4
  - C++ Version: 201703
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - CPU capability usage: AVX2  - Build settings: BUILD_TYPE=Release, CXX_COMPILER=/usr/bin/c++, CXX_FLAGS= -Wno-deprecated -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -fopenmp -DNDEBUG -DUSE_KINETO -DLIBKINETO_NOCUPTI -DLIBKINETO_NOROCTRACER -DUSE_PYTORCH_QNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -O2 -fPIC -Wall -Wextra -Werror=return-type -Werror=non-virtual-dtor -Werror=bool-operation -Wnarrowing -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wunused-local-typedefs -Wno-unused-parameter -Wno-unused-function -Wno-unused-result -Wno-strict-overflow -Wno-strict-aliasing -Wno-error=deprecated-declarations -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=redundant-decls -Wno-error=old-style-cast -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Werror=cast-function-type -Wno-stringop-overflow, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, TORCH_DISABLE_GPU_ASSERTS=ON, TORCH_VERSION=2.0.0, USE_CUDA=0, USE_CUDNN=OFF, USE_EIGEN_FOR_BLAS=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=OFF, USE_MKLDNN=0, USE_MPI=OFF, USE_NCCL=OFF, USE_NNPACK=0, USE_OPENMP=ON, USE_ROCM=OFF,

Num threads: 1

PIL version:  9.0.0.post1
[----------------------------------------------------------------------------- Resize ----------------------------------------------------------------------------]
                                                                |  Pillow (9.0.0.post1)  |  torch (2.0.0a0+git7f72623) PR  |  torch (2.0.0a0+git7f72623) PR (float)
1 threads: --------------------------------------------------------------------------------------------------------------------------------------------------------
      3 torch.uint8 channels_last bilinear 256 -> 32 aa=True    |          38.6          |               346.7             |                   361.0
      3 torch.uint8 channels_last bilinear 256 -> 32 aa=False   |                        |               327.8             |                    67.5
      3 torch.uint8 channels_last bilinear 520 -> 32 aa=True    |         112.1          |              1321.2             |                  1553.2
      3 torch.uint8 channels_last bilinear 520 -> 32 aa=False   |                        |              1248.8             |                   179.6
      3 torch.uint8 channels_last bilinear 712 -> 32 aa=True    |         184.9          |              2429.3             |                  2910.2
      3 torch.uint8 channels_last bilinear 712 -> 32 aa=False   |                        |              2306.4             |                   309.6
      4 torch.uint8 channels_last bilinear 256 -> 32 aa=True    |                        |               208.1             |                   466.6
      4 torch.uint8 channels_last bilinear 256 -> 32 aa=False   |                        |               189.6             |                    80.7
      4 torch.uint8 channels_last bilinear 520 -> 32 aa=True    |                        |               744.2             |                  2053.8
      4 torch.uint8 channels_last bilinear 520 -> 32 aa=False   |                        |               674.7             |                   231.1
      4 torch.uint8 channels_last bilinear 712 -> 32 aa=True    |                        |              1359.0             |                  3886.2
      4 torch.uint8 channels_last bilinear 712 -> 32 aa=False   |                        |              1230.8             |                   412.0
      3 torch.uint8 channels_first bilinear 256 -> 32 aa=True   |          38.3          |               346.6             |                   349.3
      3 torch.uint8 channels_first bilinear 256 -> 32 aa=False  |                        |               328.0             |                   196.9
      3 torch.uint8 channels_first bilinear 520 -> 32 aa=True   |         112.3          |              1321.6             |                  1538.2
      3 torch.uint8 channels_first bilinear 520 -> 32 aa=False  |                        |              1249.3             |                  1515.5
      3 torch.uint8 channels_first bilinear 712 -> 32 aa=True   |         185.0          |              2435.4             |                  2887.2
      3 torch.uint8 channels_first bilinear 712 -> 32 aa=False  |                        |              2312.7             |                  2556.5
      4 torch.uint8 channels_first bilinear 256 -> 32 aa=True   |                        |               209.3             |                   453.2
      4 torch.uint8 channels_first bilinear 256 -> 32 aa=False  |                        |               190.6             |                   244.8
      4 torch.uint8 channels_first bilinear 520 -> 32 aa=True   |                        |               745.8             |                  2278.6
      4 torch.uint8 channels_first bilinear 520 -> 32 aa=False  |                        |               730.4             |                  1360.3
      4 torch.uint8 channels_first bilinear 712 -> 32 aa=True   |                        |              1480.8             |                  4110.3
      4 torch.uint8 channels_first bilinear 712 -> 32 aa=False  |                        |              1311.5             |                  3714.8

Times are in microseconds (us).
```

### 30/01/2023 (Repro current results)

```
cd /tmp/pth/interpolate_vec_uint8/ && python -u check_interp.py
```


```
Torch version: 2.0.0a0+git7f72623Torch config: PyTorch built with:  - GCC 9.4  - C++ Version: 201703  - OpenMP 201511 (a.k.a. OpenMP 4.5)  - CPU capability usage: AVX2  - Build settings: BUILD_TYPE=Release, CXX_COMPILER=/usr/bin/c++, CXX_FLAGS= -Wno-deprecated -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -fopenmp -DNDEBUG -DUSE_KINETO -DLIBKINETO_NOCUPTI -DLIBKINETO_NOROCTRACER -DUSE_PYTORCH_QNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -DEDGE_PROFILER_USE_KINETO -O2 -fPIC -Wall -Wextra -Werror=return-type -Werror=non-virtual-dtor -Werror=bool-operation -Wnarrowing -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wunused-local-typedefs -Wno-unused-parameter -Wno-unused-function -Wno-unused-result -Wno-strict-overflow -Wno-strict-aliasing -Wno-error=deprecated-declarations -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=redundant-decls -Wno-error=old-style-cast -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Werror=cast-function-type -Wno-stringop-overflow, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, TORCH_DISABLE_GPU_ASSERTS=ON, TORCH_VERSION=2.0.0, USE_CUDA=0, USE_CUDNN=OFF, USE_EIGEN_FOR_BLAS=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=OFF, USE_MKLDNN=0, USE_MPI=OFF, USE_NCCL=OFF, USE_NNPACK=0, USE_OPENMP=ON, USE_ROCM=OFF,

Num threads: 1

PIL version:  9.0.0.post1
[----------------------------------------------------------------------------- Resize ----------------------------------------------------------------------------]
                                                                |  Pillow (9.0.0.post1)  |  torch (2.0.0a0+git7f72623) PR  |  torch (2.0.0a0+git7f72623) PR (float)
1 threads: --------------------------------------------------------------------------------------------------------------------------------------------------------
      3 torch.uint8 channels_last bilinear 256 -> 32 aa=True    |          41.3          |               395.4             |                   377.4
      3 torch.uint8 channels_last bilinear 256 -> 32 aa=False   |                        |               368.4             |                    67.8
      3 torch.uint8 channels_last bilinear 520 -> 32 aa=True    |         112.2          |              1456.1             |                  1557.2
      3 torch.uint8 channels_last bilinear 520 -> 32 aa=False   |                        |              1372.7             |                   180.1
      3 torch.uint8 channels_first bilinear 256 -> 32 aa=True   |          38.5          |               372.0             |                   349.0
      3 torch.uint8 channels_first bilinear 256 -> 32 aa=False  |                        |               356.8             |                   196.9
      3 torch.uint8 channels_first bilinear 520 -> 32 aa=True   |         112.8          |              1449.1             |                  1543.7
      3 torch.uint8 channels_first bilinear 520 -> 32 aa=False  |                        |              1379.2             |                  1306.1

Times are in microseconds (us).
```


```
Num threads: 1

PIL version:  9.0.0.post1
[----------------------------------------------------------------------------- Resize -----------------------------------------------------------------------------]
                                                                 |  Pillow (9.0.0.post1)  |  torch (2.0.0a0+git7f72623) PR  |  torch (2.0.0a0+git7f72623) PR (float)
1 threads: ---------------------------------------------------------------------------------------------------------------------------------------------------------
      3 torch.uint8 channels_last bilinear 270 -> 224 aa=True    |         148.4          |              628.7              |                  1269.7
      3 torch.uint8 channels_last bilinear 270 -> 224 aa=False   |                        |              608.8              |                   917.8
      3 torch.uint8 channels_first bilinear 270 -> 224 aa=True   |         149.7          |              598.4              |                   772.5
      3 torch.uint8 channels_first bilinear 270 -> 224 aa=False  |                        |              569.8              |                  1300.6

Times are in microseconds (us).
```

```
Num threads: 1

PIL version:  9.4.0
check_interp.py:93: UserWarning: The given NumPy array is not writable, and PyTorch does not support non-writable tensors. This means writing to this tensor will result in undefined behavior. You may want to copy the array to protect its data or make it writable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at ../torch/csrc/utils/tensor_numpy.cpp:206.)
  expected_pil = torch.from_numpy(np.asarray(output_pil_img)).clone().permute(2, 0, 1).contiguous()
[------------------------------------------------------------------------- Resize ------------------------------------------------------------------------]
                                                              |  Pillow (9.4.0)  |  torch (2.0.0a0+git7f72623) PR  |  torch (2.0.0a0+git7f72623) PR (float)
1 threads: ------------------------------------------------------------------------------------------------------------------------------------------------
      3 torch.uint8 channels_last bilinear 256 -> 32 aa=True  |      223.3       |              382.2              |                  361.8

Times are in microseconds (us).
```

### uint8 -> float -> resize -> uint8

```
Num threads: 1
[--------- Downsampling: torch.Size([3, 438, 906]) -> (320, 196) ---------]
                                 |  PIL 9.0.0.post1  |  1.14.0a0+git943acd4
1 threads: ----------------------------------------------------------------
      channels_first contiguous  |       345.0       |         2530.7

Times are in microseconds (us).

[--------- Downsampling: torch.Size([3, 438, 906]) -> (460, 220) ---------]
                                 |  PIL 9.0.0.post1  |  1.14.0a0+git943acd4
1 threads: ----------------------------------------------------------------
      channels_first contiguous  |       412.8       |         2947.4

Times are in microseconds (us).

[---------- Downsampling: torch.Size([3, 438, 906]) -> (120, 96) ---------]
                                 |  PIL 9.0.0.post1  |  1.14.0a0+git943acd4
1 threads: ----------------------------------------------------------------
      channels_first contiguous  |       214.0       |         2124.6

Times are in microseconds (us).

[--------- Downsampling: torch.Size([3, 438, 906]) -> (1200, 196) --------]
                                 |  PIL 9.0.0.post1  |  1.14.0a0+git943acd4
1 threads: ----------------------------------------------------------------
      channels_first contiguous  |       911.3       |         7560.4

Times are in microseconds (us).

[--------- Downsampling: torch.Size([3, 438, 906]) -> (120, 1200) --------]
                                 |  PIL 9.0.0.post1  |  1.14.0a0+git943acd4
1 threads: ----------------------------------------------------------------
      channels_first contiguous  |       291.0       |         2700.7

Times are in microseconds (us).
```


### 30/11/2022 - fallback uint8 implementation

```
Num threads: 1
[---------------------------------- Downsampling: torch.Size([3, 438, 906]) -> (320, 196) ----------------------------------]
                                 |  PIL 9.0.0.post1  |  1.14.0a0+git7a3055e, using uint8  |  1.14.0a0+git7a3055e, using float
1 threads: ------------------------------------------------------------------------------------------------------------------
      channels_first contiguous  |       348.8       |               3315.0               |               2578.3

Times are in microseconds (us).

[---------------------------------- Downsampling: torch.Size([3, 438, 906]) -> (460, 220) ----------------------------------]
                                 |  PIL 9.0.0.post1  |  1.14.0a0+git7a3055e, using uint8  |  1.14.0a0+git7a3055e, using float
1 threads: ------------------------------------------------------------------------------------------------------------------
      channels_first contiguous  |       412.5       |               4231.5               |               3004.9

Times are in microseconds (us).

[----------------------------------- Downsampling: torch.Size([3, 438, 906]) -> (120, 96) ----------------------------------]
                                 |  PIL 9.0.0.post1  |  1.14.0a0+git7a3055e, using uint8  |  1.14.0a0+git7a3055e, using float
1 threads: ------------------------------------------------------------------------------------------------------------------
      channels_first contiguous  |       216.4       |               1818.1               |               2286.3

Times are in microseconds (us).

[---------------------------------- Downsampling: torch.Size([3, 438, 906]) -> (1200, 196) ---------------------------------]
                                 |  PIL 9.0.0.post1  |  1.14.0a0+git7a3055e, using uint8  |  1.14.0a0+git7a3055e, using float
1 threads: ------------------------------------------------------------------------------------------------------------------
      channels_first contiguous  |       907.3       |               9095.5               |               5861.1

Times are in microseconds (us).

[---------------------------------- Downsampling: torch.Size([3, 438, 906]) -> (120, 1200) ---------------------------------]
                                 |  PIL 9.0.0.post1  |  1.14.0a0+git7a3055e, using uint8  |  1.14.0a0+git7a3055e, using float
1 threads: ------------------------------------------------------------------------------------------------------------------
      channels_first contiguous  |       298.1       |               2753.7               |               2865.6

Times are in microseconds (us).
```

```
PIL version:  9.0.0.post1
[-------------------- Resize measurements ---------------------]
                                |  Pillow image  |  torch tensor
1 threads: -----------------------------------------------------
      1200 -> 256, torch.uint8  |      57.0      |     295.1

Times are in microseconds (us).
```


## Dev results

- On nightly
```
cd /tmp/pth/interpolate_vec_uint8/ && python -u check_interp.py

Torch version: 2.1.0a0+git5309c44
Torch config: PyTorch built with:
  - GCC 9.4
  - C++ Version: 201703
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - CPU capability usage: AVX2
  - Build settings: BUILD_TYPE=Release, CXX_COMPILER=/usr/bin/c++, CXX_FLAGS= -D_GLIBCXX_USE_CXX11_ABI=1 -Wno-deprecated -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -DNDEBUG -DUSE_KINETO -DLIBKINETO_NOCUPTI -DLIBKINETO_NOROCTRACER -DUSE_PYTORCH_QNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -O2 -fPIC -Wall -Wextra -Werror=return-type -Werror=non-virtual-dtor -Werror=bool-operation -Wnarrowing -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-unused-parameter -Wno-unused-function -Wno-unused-result -Wno-strict-overflow -Wno-strict-aliasing -Wno-error=deprecated-declarations -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=old-style-cast -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Werror=cast-function-type -Wno-stringop-overflow, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, TORCH_DISABLE_GPU_ASSERTS=ON, TORCH_VERSION=2.1.0, USE_CUDA=0, USE_CUDNN=OFF, USE_EIGEN_FOR_BLAS=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=OFF, USE_MKLDNN=0, USE_MPI=OFF, USE_NCCL=OFF, USE_NNPACK=0, USE_OPENMP=ON, USE_ROCM=OFF,

Num threads: 1


PIL version:  9.0.0.post1
[----------------------------------------------------------- Resize -----------------------------------------------------------]
                                                                       |  Pillow (9.0.0.post1)  |  torch (2.1.0a0+git5309c44) PR
1 threads: ---------------------------------------------------------------------------------------------------------------------
      3 torch.uint8 channels_last bilinear 256 -> (224, 224) aa=True   |         128.8          |              293.1
      3 torch.uint8 channels_last bilinear 256 -> (224, 224) aa=False  |                        |              278.6

      3 torch.uint8 channels_last bilinear 256 -> (256, 224) aa=True   |          91.9          |              263.3
      3 torch.uint8 channels_last bilinear 256 -> (256, 224) aa=False  |                        |              253.1

      3 torch.uint8 channels_last bilinear 256 -> (224, 256) aa=True   |          52.4          |              220.1
      3 torch.uint8 channels_last bilinear 256 -> (225, 256) aa=False  |                        |              214.0

      4 torch.uint8 channels_last bilinear 256 -> (224, 224) aa=True   |                        |              141.3
      4 torch.uint8 channels_last bilinear 256 -> (224, 224) aa=False  |                        |              126.3

      4 torch.uint8 channels_last bilinear 256 -> (256, 224) aa=True   |                        |               99.3
      4 torch.uint8 channels_last bilinear 256 -> (256, 224) aa=False  |                        |               89.1

      4 torch.uint8 channels_last bilinear 256 -> (224, 256) aa=True   |                        |               56.9
      4 torch.uint8 channels_last bilinear 256 -> (225, 256) aa=False  |                        |               49.2

Times are in microseconds (us).
```

- PR
```
Num threads: 1

PIL version:  9.0.0.post1
[-------------------------------------------------------- Resize -------------------------------------------------------]
                                                                |  Pillow (9.0.0.post1)  |  torch (2.0.0a0+git8d22fc6) PR
1 threads: --------------------------------------------------------------------------------------------------------------
      3 torch.uint8 channels_last bilinear 256 -> 224 aa=True   |         128.5          |              496.1
      3 torch.uint8 channels_last bilinear 256 -> 224 aa=False  |                        |              483.5
      4 torch.uint8 channels_last bilinear 256 -> 224 aa=True   |                        |              194.7
      4 torch.uint8 channels_last bilinear 256 -> 224 aa=False  |                        |              178.5

Times are in microseconds (us).
```

- PR, vertical only (as template)
```
Num threads: 1

PIL version:  9.0.0.post1
[-------------------------------------------------------- Resize -------------------------------------------------------]
                                                                |  Pillow (9.0.0.post1)  |  torch (2.0.0a0+git8d22fc6) PR
1 threads: --------------------------------------------------------------------------------------------------------------
      3 torch.uint8 channels_last bilinear 256 -> 224 aa=True   |          52.7          |               52.6
      3 torch.uint8 channels_last bilinear 256 -> 224 aa=False  |                        |               47.9
      4 torch.uint8 channels_last bilinear 256 -> 224 aa=True   |                        |               56.3
      4 torch.uint8 channels_last bilinear 256 -> 224 aa=False  |                        |               51.1

Times are in microseconds (us).
```

- PR, horizontal only (as template)
```
Num threads: 1

PIL version:  9.0.0.post1
[-------------------------------------------------------- Resize -------------------------------------------------------]
                                                                |  Pillow (9.0.0.post1)  |  torch (2.0.0a0+git8d22fc6) PR
1 threads: --------------------------------------------------------------------------------------------------------------
      3 torch.uint8 channels_last bilinear 256 -> 224 aa=True   |          91.5          |              164.3
      3 torch.uint8 channels_last bilinear 256 -> 224 aa=False  |                        |              151.4
      4 torch.uint8 channels_last bilinear 256 -> 224 aa=True   |                        |              148.6
      4 torch.uint8 channels_last bilinear 256 -> 224 aa=False  |                        |              138.1

Times are in microseconds (us).
```

- PR, fixing `xmax - 3` -> `xmax - 1`

```
Num threads: 1

PIL version:  9.0.0.post1
[----------------------------------------------------------- Resize -----------------------------------------------------------]
                                                                       |  Pillow (9.0.0.post1)  |  torch (2.1.0a0+git78eda38) PR
1 threads: ---------------------------------------------------------------------------------------------------------------------
      3 torch.uint8 channels_last bilinear 256 -> (224, 224) aa=True   |         127.8          |              251.4
      3 torch.uint8 channels_last bilinear 256 -> (224, 224) aa=False  |                        |              164.6

      3 torch.uint8 channels_last bilinear 256 -> (256, 224) aa=True   |          92.5          |              211.9
      3 torch.uint8 channels_last bilinear 256 -> (256, 224) aa=False  |                        |              129.5

      3 torch.uint8 channels_last bilinear 256 -> (224, 256) aa=True   |          52.5          |               52.6
      3 torch.uint8 channels_last bilinear 256 -> (225, 256) aa=False  |                        |               48.0

      4 torch.uint8 channels_last bilinear 256 -> (224, 224) aa=True   |                        |              170.9
      4 torch.uint8 channels_last bilinear 256 -> (224, 224) aa=False  |                        |              157.5

      4 torch.uint8 channels_last bilinear 256 -> (256, 224) aa=True   |                        |              127.2
      4 torch.uint8 channels_last bilinear 256 -> (256, 224) aa=False  |                        |              118.5

      4 torch.uint8 channels_last bilinear 256 -> (224, 256) aa=True   |                        |               62.0
      4 torch.uint8 channels_last bilinear 256 -> (225, 256) aa=False  |                        |               51.6

Times are in microseconds (us).
```




## Some tracebacks

- Channels last
```
(1, 3, 9, 8) --> (1, 3, 9, 2)

TI_SHOW: size0=3  <--- this is channels
TI_SHOW: size1=2  <--- this is output width
AA TI_SHOW_STRIDES: 1 1 | 0 0 0 0 0 3 0 8 8 8 |
AA TI_BASIC_LOOP -> NONZERO STRIDES   <-> horizontal pass

... (9 times)

(1, 3, 9, 2) --> (1, 3, 4, 2)

TI_SHOW: size0=6   <--- this is channels x output width = 3 * 2
TI_SHOW: size1=4   <--- this is output height
AA TI_SHOW_STRIDES: 1 1 | 0 0 0 0 0 6 0 8 8 8 |
AA TI_BASIC_LOOP -> ZERO STRIDES   <-> vertical pass
```


### Debug

```
OMP_NUM_THREADS=1 gdb --args python -m pytest nhug_script.py::test_lol[True-False-bilinear-1-1-271-268-224-224] -vvv


(gdb) b aten/src/ATen/native/cpu/UpSampleKernel.cpp:1774
(gdb) b aten/src/ATen/native/cpu/UpSampleKernelAVXAntialias.h:564
(gdb) b aten/src/ATen/native/cpu/UpSampleKernelAVXAntialias.h:410
(gdb) b aten/src/ATen/native/cpu/UpSampleKernelAVXAntialias.h:281
```



```
OMP_NUM_THREADS=1 ATEN_CPU_CAPABILITY=default gdb --args python -u debug_interp2.py



(gdb) b aten/src/ATen/native/cpu/UpSampleKernel.cpp:1870
(gdb) b aten/src/ATen/native/cpu/UpSampleKernel.cpp:1591
(gdb) b aten/src/ATen/native/cpu/UpSampleKernel.cpp:1524
(gdb) b aten/src/ATen/native/cpu/UpSampleKernel.cpp:1490

(gdb) b aten/src/ATen/native/cpu/UpSampleKernel.cpp:1774
(gdb) b aten/src/ATen/native/cpu/UpSampleKernel.cpp:1365
(gdb) b aten/src/ATen/native/cpu/UpSampleKernel.cpp:293
```


```
tensor_uint8 = torch.tensor([
torch.Size([1, 8, 8])

    tensor_uint8 = torch.tensor([
        [ 12.,  13.,  14.,  15.,  16.,  17.,  18.,  19.],
        [ 60.,  61.,  62.,  63.,  64.,  65.,  66.,  67.],
        [108., 109., 110., 111., 112., 113., 114., 115.],
        [156., 157., 158., 159., 160., 161., 162., 163.],
        [204., 205., 206., 207., 208., 209., 210., 211.],
        [252., 253., 254., 255.,   0.,   1.,   2.,   3.],
        [ 44.,  45.,  46.,  47.,  48.,  49.,  50.,  51.],
        [ 92.,  93.,  94.,  95.,  96.,  97.,  98.,  99.]
    ], dtype=torch.uint8)[None, ...]


Memory format: channels_first
output_uint8:
 tensor([[ 30,  33,  35,  37],
        [132, 135, 137, 139],
        [252, 255, 104, 107],
        [ 52,  55,  79,  81]], dtype=torch.uint8)

output_value: 268.5
output_float32:
 tensor([[ 32.,  34.,  36.,  38.],
        [132., 134., 136., 139.],
        [252., 255.,  90., 107.],
        [ 49.,  49.,  79.,  79.]])
PyTorch uint8 vs PyTorch float: Mean Absolute Error: 2.0
PyTorch uint8 vs PyTorch float: Max Absolute Error: 14.0

Dim: 0
index0: 24
w0: -0.09375
index1: 32
w1: 0.59375
index2: 40
w2: 0.59375
index3: 48
w3: 48
input[i0]: 156
input[i1]: 204
input[i2]: 252
input[i3]: 44
output_value: 252
Dim: 1
index0: 3
w0: -0.09375
index1: 4
w1: 0.59375
index2: 5
w2: 0.59375
index3: 6
w3: 6
input[i0]: 15
input[i1]: 16
input[i2]: 17
input[i3]: 18
```



```
96              static inline scalar_t eval(char* src, char** data, const int64_t* strides, int64_t i) {

(gdb) p i
$40 = 2


(gdb) p {(float)ids, wts, (float)t, output}
$113 = {12, -0.09375, 159, -14.90625}

(gdb) p {(float)ids, wts, (float)t, output}
$113 = {16, 0.59375, 160, 80.09375}

(gdb) p {(float)ids, wts, (float)t, output}
$113 = {20, 0.59375, 161, 175.6875}

(gdb) p {(float)ids, wts, (float)t, output}
$113 = {24, -0.09375, 162, 160.5}


82                scalar_t t = Interpolate<n - 1, scalar_t, index_t, interp_size>::eval(src + ids, &data[2 * interp_size], &strides[2 * interp_size], i)
83                scalar_t output = t * wts

(gdb) p output
$58 = -15.046875
(gdb) p t
$59 = 160.5


96              static inline scalar_t eval(char* src, char** data, const int64_t* strides, int64_t i) {


(gdb) p {(float)ids, wts, (float)t, output}
$113 = {12, -0.09375, 207, -19.40625}

(gdb) p {(float)ids, wts, (float)t, output}
$113 = {16, 0.59375, 208, 104.09375}

(gdb) p {(float)ids, wts, (float)t, output}
$113 = {20, 0.59375, 209, 228.1875}

(gdb) p {(float)ids, wts, (float)t, output}
$113 = {24, -0.09375, 210, 208.5}


87                  t = Interpolate<n - 1, scalar_t, index_t, interp_size>::eval(src + ids, &data[2 * interp_size], &strides[2 * interp_size], i);

(gdb) p {(float)ids, wts, (float)t, output}
$113 = {128, 0.59375, 208.5, 108.75}


96              static inline scalar_t eval(char* src, char** data, const int64_t* strides, int64_t i) {

(gdb) p {(float)ids, wts, (float)t, output}
$113 = {12, -0.09375, 255, -23.90625}

(gdb) p {(float)ids, wts, (float)t, output}
$113 = {16, 0.59375, 0, -23.90625}

(gdb) p {(float)ids, wts, (float)t, output}
$113 = {20, 0.59375, 1, -23.3125}

(gdb) p {(float)ids, wts, (float)t, output}
$113 = {24, -0.09375, 2, -23.5}


87                  t = Interpolate<n - 1, scalar_t, index_t, interp_size>::eval(src + ids, &data[2 * interp_size], &strides[2 * interp_size], i);

(gdb) p {wts, (float)t, output}
$113 = {0.59375, -23.5, 94.796875}


96              static inline scalar_t eval(char* src, char** data, const int64_t* strides, int64_t i) {

(gdb) p {(float)ids, wts, (float)t, output}
$113 = {12, -0.09375, 47, -4.40625}

(gdb) p {(float)ids, wts, (float)t, output}
$113 = {16, 0.59375, 48, 24.09375}

(gdb) p {(float)ids, wts, (float)t, output}
$113 = {20, 0.59375, 49, 53.1875}$109 = 20

(gdb) p {(float)ids, wts, (float)t, output}
$114 = {24, -0.09375, 50, 48.5}


87                  t = Interpolate<n - 1, scalar_t, index_t, interp_size>::eval(src + ids, &data[2 * interp_size], &strides[2 * interp_size], i);

(gdb) p {(float)ids, wts, (float)t, output}
$115 = {192, -0.09375, 48.5, 90.25}

(gdb) p output
$116 = 90.25
```

=> Equivalent horizontal pass produces the following vertical line:
```
[160.5, 208.5, -23.5, 48.5]
```
But in separable version as intermediate temp tensor is uint8 this vertical line is
```
[161, 209, 0, 49]
```

! Major difference is that negative value -23.5 is saturated to 0


==> float version produces the value:
dot([-0.09375, 0.59375, 0.59375, -0.09375], [160.5, 208.5, -23.5, 48.5]) -> 90.25


==> uint8 version produces the value:
dot([-0.09375, 0.59375, 0.59375, -0.09375], [161, 209, 0, 49]) -> 104.40625


```
│   1625            _separable_upsample_generic_Nd_kernel_impl_single_dim<
│   1626                out_ndims,
│   1627                scale_t,
│   1628                F,
│   1629                true>(
│   1630                temp_output, temp_input, interp_dim, align_corners, scales, antialias);
│   1631            temp_input = temp_output;

(gdb) torch-tensor-repr temp_input
Python-level repr of temp_input:
tensor([[[[ 12,  15,  17,  19],
          [ 60,  63,  65,  67],
          [108, 111, 113, 115],
          [156, 159, 161, 163],
          [204, 207, 209, 211],
          [252, 255,   0,   3],
          [ 44,  47,  49,  51],
          [ 92,  95,  97,  99]]]], dtype=torch.uint8)
```


## Various past issues

PT113 vs PT nightly (2.1.0a0+git5309c44)
```
python verif_interp2.py verif_expected --is_ref=False

mf/size/dtype/c/osize/aa/mode/ac :  channels_last [256, 256] torch.uint8 3 (320, 321) False bilinear True  -> get expected from file
Traceback (most recent call last):
  File "verif_interp2.py", line 191, in <module>
    fire.Fire(main)
  File "/usr/local/lib/python3.8/dist-packages/fire/core.py", line 141, in Fire
    component_trace = _Fire(component, args, parsed_flag_args, context, name)
  File "/usr/local/lib/python3.8/dist-packages/fire/core.py", line 475, in _Fire
    component, remaining_args = _CallAndUpdateTrace(
  File "/usr/local/lib/python3.8/dist-packages/fire/core.py", line 691, in _CallAndUpdateTrace
    component = fn(*varargs, **kwargs)
  File "verif_interp2.py", line 170, in main
    assert max_abs_err.item() < 1.0 + 1e-5, (max_abs_err.item(), expected_ten.float()[m], output.float()[m])
AssertionError: (2.0, tensor([159.]), tensor([161.]))
```




# RGBA Image resizing with a vectorized algorithm

## Horizontal pass vectorized algorithm on RGBA data

Input data is stored as
```
input = [r[0], g[0], b[0], a[0], r[1], g[1], b[1], a[1], r[2], g[2], b[2], a[2], ...]
```

Weights are float values computed for each output pixel and rescaled to uint16:
```
weights[i] = [w[i, 0], w[i, 1], ..., w[i, K - 1]]
```

We want to compute the output as following:
```
output = [oR[0], oG[0], oB[0], oA[0], oR[1], oG[1], oB[1], oA[1], ...]
```
where
```
oR[i] = r[xmin[i]] * w[i, 0] + r[xmin[i] + 1] * w[i, 1] + ... + r[xmin[i] + K - 1] * w[i, K - 1]
oG[i] = g[xmin[i]] * w[i, 0] + g[xmin[i] + 1] * w[i, 1] + ... + g[xmin[i] + K - 1] * w[i, K - 1]
oB[i] = b[xmin[i]] * w[i, 0] + b[xmin[i] + 1] * w[i, 1] + ... + b[xmin[i] + K - 1] * w[i, K - 1]
```

### Output computation with integers

```
oR[i] = r[xmin[i]] * w[i, 0] + r[xmin[i] + 1] * w[i, 1] + ... + r[xmin[i] + K - 1] * w[i, K - 1]
```
where `r` is uint8 and `w` is float.

Here is a way to perform computations in integer with a minimal precision loss.
1) Rescale float weights into int16
- find max float weight to estimate `weights_precision`
```c++
unsigned int weights_precision = 0;
for (weights_precision = 0; weights_precision < 22; weights_precision += 1) {
      int next_value = (int) (0.5 + w_max * (1 << (weights_precision + 1)));
      if (next_value >= (1 << 15))
            break;
}
```
- transform float value into int16 value:
```c++
w_i16[i] = (int16) (sign(w_f32) * 0.5 + w_f32 * (1 << weights_precision));
```

2) Compute output value using int dtype:
```c++
uint8 dst = ...
uint8 src = ...
int16 wts = ...
int output = 1 << (weights_precision - 1);

output += src[0] * wts[0];
output += src[1] * wts[1];
...
output += src[K-1] * wts[K-1];
output = (output >> weights_precision);

dst[o] = (uint8) clamp(output, 0, 255);
```

### Vectorized version

As data format is RGBA with R,G,B,A being uint8, we can encode 4 values as a single uint32 value.

Working register, avx2 = 32 uint8 places
```
reg = [0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 0 0]
```

We can split K (size of weight vector for a given output index) as a sum: `K = n * 4 + m * 2 + k`.
We load and process 4 weights values in a loop ("block 4") then we process 2 weights values in another loop ("block 2") and finally we process 1 weights value in the final loop ("block 1").

1) As we are doing computations in integer dtype, we add the offset (=`1 << (weights_precision - 1)`):
```
reg = [
      0 128 0 0 0 128 0 0 | 0 128 0 0 0 128 0 0 | 0 128 0 0 0 128 0 0 | 0 128 0 0 0 128 0 0
]
```
2) Load weights. For "block 4" we load 4 int16 values `(w0, w1)` and `(w2, w3)`. Each value then will be represented in the register with uint8 values `wl_0` and `wh_0`:
```
w01 = [
      wl_0 wh_0 wl_1 wh_1 wl_0 wh_0 wl_1 wh_1 | ... | ... | wl_0 wh_0 wl_1 wh_1 wl_0 wh_0 wl_1 wh_1
]
```
For example,
```
w01 = [
      183 45 0 64 183 45 0 64 | 183 45 0 64 183 45 0 64 | 183 45 0 64 183 45 0 64 | 183 45 0 64 183 45 0 64
]
```

```
w23 = [
      wl_2 wh_2 wl_3 wh_3 wl_2 wh_3 wl_2 wh_3 | ... | ... | wl_2 wh_2 wl_3 wh_3 wl_2 wh_2 wl_3 wh_3
]
```
On the next iteration we will load next pair of weights `(w4, w5)` as `w45` and `(w6, w7)` as `w67` in case of "block 4".

In case of "block 2" we will load 2 int16 values `(w0, w1)`:
```
w01 = [
      wl_0 wh_0 wl_1 wh_1 wl_0 wh_0 wl_1 wh_1 | ... | ... | wl_0 wh_0 wl_1 wh_1 wl_0 wh_0 wl_1 wh_1
]
```

And in case of "block 1" we will load only 1 int16 value `w0`:
```
w0 = [
      wl_0 wh_0 0 0 wl_0 wh_0 0 0 | ... | ... | wl_0 wh_0 0 0 wl_0 wh_0 0 0
]
```

3) Load source data. Each RGBA pixel has 4 uint8 size, so half of 256-bits register (=16 uint8 places) can be filled with 4 pixels. To fill 32 uint8 places (=256 bits) we can load 4 pixels from two lines, e.g. `r0`-`r3` and `rr0`-`rr3` where `ri` is a red value from line0 and `rri` is a red value from line1.

Thus, we can process in parallel 2 lines. The number of loaded pixels determines block option. For "block 4" we load pixels 0-3:
```
data = [
      r0 g0 b0 a0 r1 g1 b1 a1 | r2 g2 b2 a2 r3 g3 b3 a3 | rr0 gg0 bb0 aa0 rr1 gg1 bb1 aa1 | rr2 gg2 bb2 aa2 rr3 gg3 bb3 aa3
]
```
For example,
```
data = [
      0 1 2 255 3 4 5 255 | 6 7 8 255 9 10 11 255 | 27 28 29 255 30 31 32 255 | 33 34 35 255 36 37 38 255
]
```

In case of "block 2", we load
```
data = [
      r0 g0 b0 a0 r1 g1 b1 a1 | 0 0 0 0 0 0 0 0 | rr0 gg0 bb0 aa0 rr1 gg1 bb1 aa1 | 0 0 0 0 0 0 0 0
]
```
and in case of "block 1", we load
```
data = [
      r0 g0 b0 a0 0 0 0 0 | 0 0 0 0 0 0 0 0 | rr0 gg0 bb0 aa0 0 0 0 0 | 0 0 0 0 0 0 0 0
]
```

4) As we loaded weights only 2 values we have to split and shuffle the source data such we could correctly multiply `r0 * w0 + r1 * w1` and `r2 * w2 + r3 * w3`. For "block 4" we obtain:
```
data_01 = [
      r0 0 r1 0 g0 0 g1 0 | b0 0 b1 0 a0 0 a1 0 | rr0 0 rr1 0 gg0 0 gg1 0 | bb0 0 bb1 0 aa0 0 aa1 0
]
data_23 = [
      r2 0 r3 0 g2 0 g3 0 | b2 0 b3 0 a2 0 a3 0 | rr2 0 rr3 0 gg2 0 gg3 0 | bb2 0 bb3 0 aa2 0 aa3 0
]
```

For "block 2" we will have
```
data_01 = [
      r0 0 r1 0 g0 0 g1 0 | b0 0 b1 0 a0 0 a1 0 | rr0 0 rr1 0 gg0 0 gg1 0 | bb0 0 bb1 0 aa0 0 aa1 0
]
```
and for "block 1" we will have
```
data_0 = [
      r0 0 0 0 g0 0 0 0 | b0 0 0 0 a0 0 0 0 | rr0 0 0 0 gg0 0 0 0 | bb0 0 0 0 aa0 0 0 0
]
```

5) Multiply and add weights and source data using integer 32-bits precision. Integer 32-bits precision means the output will take 4 placeholders (a b c d).

```
# w01 = [
#       wl_0 wh_0 wl_1 wh_1 wl_0 wh_0 wl_1 wh_1 | ... | ... | wl_0 wh_0 wl_1 wh_1 wl_0 wh_0 wl_1 wh_1
# ]

out01 = data_01 * w01

out01 = [
      (r0 0) * (wl_0 wh_0) + (r1 0) * (wl_1 wh_1), (g0 0) * (wl_0 wh_0) + (g1 0) * (wl_1 wh_1)     |
      (b0 0) * (wl_0 wh_0) + (b1 0) * (wl_1 wh_1), (a0 0) * (wl_0 wh_0), (a1 0) * (wl_1 wh_1)      |
      (rr0 0) * (wl_0 wh_0) + (rr1 0) * (wl_1 wh_1), (gg0 0) * (wl_0 wh_0) + (gg1 0) * (wl_1 wh_1) |
      (bb0 0) * (wl_0 wh_0) + (bb1 0) * (wl_1 wh_1), (aa0 0) * (wl_0 wh_0) + (a1 0) * (wl_1 wh_1)
]
```
where `(pi 0) * (wl_j wh_j) + (pk 0) * (wl_n wh_n) = (out_0, out_1, out_2, out_3)`.

```
out23 = data_23 * w23

out23 = [
      (r2 0) * (wl_2 wh_2) + (r3 0) * (wl_3, wh_3), (g2 0) * (wl_2 wh_2) + (g3 0) * (wl_3 wh_3)    |
      (b2 0) * (wl_2 wh_2) + (b3 0) * (wl_3 wh_3), (a2 0) * (wl_2 wh_2) + (a3 0) * (wl_3 wh_3)     |
      (rr2 0) * (wl_2 wh_2) + (rr3 0) * (wl_3 wh_3), (gg2 0) * (wl_2 wh_2) + (gg3 0) * (wl_3 wh_3) |
      (bb2 0) * (wl_2 wh_2) + (bb3 0) * (wl_3 wh_3), (aa2 0) * (wl_2 wh_2) + (a3 0) * (wl_3 wh_3)]
```

For "block 1" we will have
```
out0 = [
      (r0 0) * (wl_0 wh_0), (g0 0) * (wl_0 wh_0)   |
      (b0 0) * (wl_0 wh_0), (a0 0) * (wl_0 wh_0)   |
      (rr0 0) * (wl_0 wh_0), (gg0 0) * (wl_0 wh_0) |
      (bb0 0) * (wl_0 wh_0), (aa0 0) * (wl_0 wh_0)
]
```
Here each element like `(r0 0) * (wl_0 wh_0)` represent int32 and takes 4 placeholders.


Output is accumulated with the results from previous iterations.

6) Add registers `out01` and `out23` together in case of "block 4"

```
out1234 = [

      (r0 0) * (wl_0 wh_0) + (r1 0) * (wl_1 wh_1) + (r2 0) * (wl_2 wh_2) + (r3 0) * (wl_3, wh_3),
      (g0 0) * (wl_0 wh_0) + (g1 0) * (wl_1 wh_1) + (g2 0) * (wl_2 wh_2) + (g3 0) * (wl_3 wh_3) |

      (b0 0) * (wl_0 wh_0) + (b1 0) * (wl_1 wh_1) + (b2 0) * (wl_2 wh_2) + (b3 0) * (wl_3 wh_3),
      (a0 0) * (wl_0 wh_0), (a1 0) * (wl_1 wh_1) + (a2 0) * (wl_2 wh_2) + (a3 0) * (wl_3 wh_3) |

      (rr0 0) * (wl_0 wh_0) + (rr1 0) * (wl_1 wh_1) + (rr2 0) * (wl_2 wh_2) + (rr3 0) * (wl_3 wh_3),
      (gg0 0) * (wl_0 wh_0) + (gg1 0) * (wl_1 wh_1) + (gg2 0) * (wl_2 wh_2) + (gg3 0) * (wl_3 wh_3) |

      (bb0 0) * (wl_0 wh_0) + (bb1 0) * (wl_1 wh_1) + (bb2 0) * (wl_2 wh_2) + (bb3 0) * (wl_3 wh_3),
      (aa0 0) * (wl_0 wh_0) + (aa1 0) * (wl_1 wh_1) + (aa2 0) * (wl_2 wh_2) + (aa3 0) * (wl_3 wh_3)
]
```

7) Shift back the output integer values (`output = (output >> weights_precision)`)

```
out12 = out12 >> weights_precision
# or
out1234 = out1234 >> weights_precision
```

8) Convert packed signed 32-bit integers to packed 16-bit integers using signed saturation

```
(a a a a b b b b | c c c c d d d d) -> (a a b b c c d d | 0 0 0 0 0 0 0 0)
```

9) Convert packed signed 16-bit integers to packed 8-bit integers using unsigned saturation

```
(a a b b c c d d) -> (a b c d 0 0 0 0)
```

10) Write the output into single uint32
```
(a b c d) -> x_uint32
```


### interpolate_vec_uint8/cpp/check_resize_avx.cpp logs

```
print_uint32 lineIn0: 0 1 2 255
print_uint32 lineIn1: 27 28 29 255
print_uint32 lineIn2: 54 55 56 255
print_uint32 lineIn3: 81 82 83 255
Weights as int16:
11703 16384 16384 11703 7022 2341 1235 0 0
2341 7022 11703 16384 16384 11703 1235 0 0
Weights as uint8:
183 45 0 64 0 64 183 45 110 27 37 9 211 4 0 0 0 0
27 37 9 211 4 0 0 0 0 37 9 110 27 183 45 0 64 0 64 183 45 211 4 0 0 0 0

-- xx=0
print_m256i sss0: 0 128 0 0 0 128 0 0 0 128 0 0 0 128 0 0 0 128 0 0 0 128 0 0 0 128 0 0 0 128 0 0
print_m256i sss1: 0 128 0 0 0 128 0 0 0 128 0 0 0 128 0 0 0 128 0 0 0 128 0 0 0 128 0 0 0 128 0 0

- block 4, x: 0
print_m256i mmk0: 183 45 0 64 183 45 0 64 183 45 0 64 183 45 0 64 183 45 0 64 183 45 0 64 183 45 0 64 183 45 0 64
print_m256i mmk1: 0 64 183 45 0 64 183 45 0 64 183 45 0 64 183 45 0 64 183 45 0 64 183 45 0 64 183 45 0 64 183 45
print_m256i source: 0 1 2 255 3 4 5 255 6 7 8 255 9 10 11 255 27 28 29 255 30 31 32 255 33 34 35 255 36 37 38 255
print_m256i pix: 0 0 3 0 1 0 4 0 2 0 5 0 255 0 255 0 27 0 30 0 28 0 31 0 29 0 32 0 255 0 255 0
print_m256i sss0: 0 64 1 0 183 173 1 0 110 27 2 0 73 201 109 0 77 210 12 0 4 64 13 0 187 173 13 0 73 201 109 0
print_m256i pix: 6 0 9 0 7 0 10 0 8 0 11 0 255 0 255 0 33 0 36 0 34 0 37 0 35 0 38 0 255 0 255 0
print_m256i sss0: 111 91 4 0 221 54 5 0 75 18 6 0 146 18 219 0 9 128 27 0 119 91 28 0 229 54 29 0 146 18 219 0
print_m256i source: 54 55 56 255 57 58 59 255 60 61 62 255 63 64 65 255 81 82 83 255 84 85 86 255 87 88 89 255 90 91 92 255
print_m256i pix: 54 0 57 0 55 0 58 0 56 0 59 0 255 0 255 0 81 0 84 0 82 0 85 0 83 0 86 0 255 0 255 0
print_m256i sss1: 154 100 24 0 81 210 24 0 8 64 25 0 73 201 109 0 231 246 35 0 158 100 36 0 85 210 36 0 73 201 109 0
print_m256i pix: 60 0 63 0 61 0 64 0 62 0 65 0 255 0 255 0 87 0 90 0 88 0 91 0 89 0 92 0 255 0 255 0
print_m256i sss1: 163 164 50 0 17 128 51 0 127 91 52 0 146 18 219 0 61 201 73 0 171 164 74 0 25 128 75 0 146 18 219 0
- block 2, x: 4
print_m256i mmk: 110 27 37 9 110 27 37 9 110 27 37 9 110 27 37 9 110 27 37 9 110 27 37 9 110 27 37 9 110 27 37 9
print_m128i a0: 12 13 14 255 15 16 17 255 0 0 0 0 0 0 0 0
print_m256i b0: 12 13 14 255 15 16 17 255 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
print_m128i a1: 39 40 41 255 42 43 44 255 0 0 0 0 0 0 0 0
print_m256i 1 pix: 12 13 14 255 15 16 17 255 0 0 0 0 0 0 0 0 39 40 41 255 42 43 44 255 0 0 0 0 0 0 0 0
print_m256i m0: 0 255 4 255 1 255 5 255 2 255 6 255 3 255 7 255 0 255 4 255 1 255 5 255 2 255 6 255 3 255 7 255
print_m256i 2 pix: 12 0 15 0 13 0 16 0 14 0 17 0 255 0 255 0 39 0 42 0 40 0 43 0 41 0 44 0 255 0 255 0
print_m256i tmp0: 83 210 1 0 230 246 1 0 121 27 2 0 109 110 36 0 212 173 5 0 103 210 5 0 250 246 5 0 109 110 36 0
print_m256i sss0: 194 45 6 0 195 45 7 0 196 45 8 0 255 128 255 0 221 45 33 0 222 45 34 0 223 45 35 0 255 128 255 0
print_m256i sss1: 248 45 60 0 249 45 61 0 250 45 62 0 255 128 255 0 19 46 87 0 20 46 88 0 21 46 89 0 255 128 255 0
- block 1, x: 6
print_m256i mmk: 211 4 0 0 211 4 0 0 211 4 0 0 211 4 0 0 211 4 0 0 211 4 0 0 211 4 0 0 211 4 0 0
print_m128i a0: 18 0 0 0 19 0 0 0 20 0 0 0 255 0 0 0
print_m256i b0: 18 0 0 0 19 0 0 0 20 0 0 0 255 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
print_m128i a1: 45 0 0 0 46 0 0 0 47 0 0 0 255 0 0 0
print_m256i pix: 18 0 0 0 19 0 0 0 20 0 0 0 255 0 0 0 45 0 0 0 46 0 0 0 47 0 0 0 255 0 0 0
print_m256i tmp0: 214 86 0 0 169 91 0 0 124 96 0 0 45 206 4 0 23 217 0 0 234 221 0 0 189 226 0 0 45 206 4 0
print_m256i sss0: 152 132 6 0 108 137 7 0 64 142 8 0 44 79 4 1 244 6 34 0 200 11 35 0 156 16 36 0 44 79 4 1
print_m256i pix: 72 0 0 0 73 0 0 0 74 0 0 0 255 0 0 0 99 0 0 0 100 0 0 0 101 0 0 0 255 0 0 0
print_m256i tmp1: 88 91 1 0 43 96 1 0 254 100 1 0 45 206 4 0 153 221 1 0 108 226 1 0 63 231 1 0 45 206 4 0
print_m256i sss1: 80 137 61 0 36 142 62 0 248 146 63 0 44 79 4 1 172 11 89 0 128 16 90 0 84 21 91 0 44 79 4 1
--
print_m256i sss0 ->: 152 132 6 0 108 137 7 0 64 142 8 0 44 79 4 1 244 6 34 0 200 11 35 0 156 16 36 0 44 79 4 1
print_m256i -> sss0: 6 0 0 0 7 0 0 0 8 0 0 0 4 1 0 0 34 0 0 0 35 0 0 0 36 0 0 0 4 1 0 0
print_m256i sss1 ->: 80 137 61 0 36 142 62 0 248 146 63 0 44 79 4 1 172 11 89 0 128 16 90 0 84 21 91 0 44 79 4 1
print_m256i -> sss1: 61 0 0 0 62 0 0 0 63 0 0 0 4 1 0 0 89 0 0 0 90 0 0 0 91 0 0 0 4 1 0 0
print_m256i sss0: 6 0 7 0 8 0 4 1 0 0 0 0 0 0 0 0 34 0 35 0 36 0 4 1 0 0 0 0 0 0 0 0
print_m256i sss1: 61 0 62 0 63 0 4 1 0 0 0 0 0 0 0 0 89 0 90 0 91 0 4 1 0 0 0 0 0 0 0 0
print_m256i sss0: 6 7 8 255 0 0 0 0 0 0 0 0 0 0 0 0 34 35 36 255 0 0 0 0 0 0 0 0 0 0 0 0
print_m256i sss1: 61 62 63 255 0 0 0 0 0 0 0 0 0 0 0 0 89 90 91 255 0 0 0 0 0 0 0 0 0 0 0 0
print_m128i e0: 6 7 8 255 0 0 0 0 0 0 0 0 0 0 0 0
print_m128i e1: 34 35 36 255 0 0 0 0 0 0 0 0 0 0 0 0
print_m128i e2: 61 62 63 255 0 0 0 0 0 0 0 0 0 0 0 0
print_m128i e3: 89 90 91 255 0 0 0 0 0 0 0 0 0 0 0 0
print_uint32 lineOut0: 6 7 8 255
print_uint32 lineOut1: 34 35 36 255
print_uint32 lineOut2: 61 62 63 255
print_uint32 lineOut3: 89 90 91 255
-- xx=1
print_m256i sss0: 0 128 0 0 0 128 0 0 0 128 0 0 0 128 0 0 0 128 0 0 0 128 0 0 0 128 0 0 0 128 0 0
print_m256i sss1: 0 128 0 0 0 128 0 0 0 128 0 0 0 128 0 0 0 128 0 0 0 128 0 0 0 128 0 0 0 128 0 0
- block 4, x: 0
print_m256i mmk0: 37 9 110 27 37 9 110 27 37 9 110 27 37 9 110 27 37 9 110 27 37 9 110 27 37 9 110 27 37 9 110 27
print_m256i mmk1: 183 45 0 64 183 45 0 64 183 45 0 64 183 45 0 64 183 45 0 64 183 45 0 64 183 45 0 64 183 45 0 64
print_m256i source: 6 7 8 255 9 10 11 255 12 13 14 255 15 16 17 255 33 34 35 255 36 37 38 255 39 40 41 255 42 43 44 255
print_m256i pix: 6 0 9 0 7 0 10 0 8 0 11 0 255 0 255 0 33 0 36 0 34 0 37 0 35 0 38 0 255 0 255 0
print_m256i sss0: 188 173 1 0 79 210 1 0 226 246 1 0 109 238 36 0 61 137 5 0 208 173 5 0 99 210 5 0 109 238 36 0
print_m256i pix: 12 0 15 0 13 0 16 0 14 0 17 0 255 0 255 0 39 0 42 0 40 0 43 0 41 0 44 0 255 0 255 0
print_m256i sss0: 80 146 7 0 154 36 8 0 228 182 8 0 182 55 146 0 30 0 23 0 104 146 23 0 178 36 24 0 182 55 146 0
print_m256i source: 60 61 62 255 63 64 65 255 66 67 68 255 69 70 71 255 87 88 89 255 90 91 92 255 93 94 95 255 96 97 98 255
print_m256i pix: 60 0 63 0 61 0 64 0 62 0 65 0 255 0 255 0 87 0 90 0 88 0 91 0 89 0 92 0 255 0 255 0
print_m256i sss1: 190 100 9 0 81 137 9 0 228 173 9 0 109 238 36 0 63 64 13 0 210 100 13 0 101 137 13 0 109 238 36 0
print_m256i pix: 66 0 69 0 67 0 70 0 68 0 71 0 255 0 255 0 93 0 96 0 94 0 97 0 95 0 98 0 255 0 255 0
print_m256i sss1: 236 109 38 0 54 0 39 0 128 146 39 0 182 55 146 0 186 219 53 0 4 110 54 0 78 0 55 0 182 55 146 0
- block 2, x: 4
print_m256i mmk: 0 64 183 45 0 64 183 45 0 64 183 45 0 64 183 45 0 64 183 45 0 64 183 45 0 64 183 45 0 64 183 45
print_m128i a0: 18 19 20 255 21 22 23 255 0 0 0 0 0 0 0 0
print_m256i b0: 18 19 20 255 21 22 23 255 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
print_m128i a1: 45 46 47 255 48 49 50 255 0 0 0 0 0 0 0 0
print_m256i 1 pix: 18 19 20 255 21 22 23 255 0 0 0 0 0 0 0 0 45 46 47 255 48 49 50 255 0 0 0 0 0 0 0 0
print_m256i m0: 0 255 4 255 1 255 5 255 2 255 6 255 3 255 7 255 0 255 4 255 1 255 5 255 2 255 6 255 3 255 7 255
print_m256i 2 pix: 18 0 21 0 19 0 22 0 20 0 23 0 255 0 255 0 45 0 48 0 46 0 49 0 47 0 50 0 255 0 255 0
print_m256i tmp0: 3 64 8 0 186 173 8 0 113 27 9 0 73 73 109 0 80 210 19 0 7 64 20 0 190 173 20 0 73 73 109 0
print_m256i sss0: 83 210 15 0 84 210 16 0 85 210 17 0 255 128 255 0 110 210 42 0 111 210 43 0 112 210 44 0 255 128 255 0
print_m256i sss1: 137 210 69 0 138 210 70 0 139 210 71 0 255 128 255 0 164 210 96 0 165 210 97 0 166 210 98 0 255 128 255 0
- block 1, x: 6
print_m256i mmk: 211 4 0 0 211 4 0 0 211 4 0 0 211 4 0 0 211 4 0 0 211 4 0 0 211 4 0 0 211 4 0 0
print_m128i a0: 24 0 0 0 25 0 0 0 26 0 0 0 255 0 0 0
print_m256i b0: 24 0 0 0 25 0 0 0 26 0 0 0 255 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
print_m128i a1: 51 0 0 0 52 0 0 0 53 0 0 0 255 0 0 0
print_m256i pix: 24 0 0 0 25 0 0 0 26 0 0 0 255 0 0 0 51 0 0 0 52 0 0 0 53 0 0 0 255 0 0 0
print_m256i tmp0: 200 115 0 0 155 120 0 0 110 125 0 0 45 206 4 0 9 246 0 0 220 250 0 0 175 255 0 0 45 206 4 0
print_m256i sss0: 27 70 16 0 239 74 17 0 195 79 18 0 44 79 4 1 119 200 43 0 75 205 44 0 31 210 45 0 44 79 4 1
print_m256i pix: 78 0 0 0 79 0 0 0 80 0 0 0 255 0 0 0 105 0 0 0 106 0 0 0 107 0 0 0 255 0 0 0
print_m256i tmp1: 74 120 1 0 29 125 1 0 240 129 1 0 45 206 4 0 139 250 1 0 94 255 1 0 49 4 2 0 45 206 4 0
print_m256i sss1: 211 74 71 0 167 79 72 0 123 84 73 0 44 79 4 1 47 205 98 0 3 210 99 0 215 214 100 0 44 79 4 1
--
print_m256i sss0 ->: 27 70 16 0 239 74 17 0 195 79 18 0 44 79 4 1 119 200 43 0 75 205 44 0 31 210 45 0 44 79 4 1
print_m256i -> sss0: 16 0 0 0 17 0 0 0 18 0 0 0 4 1 0 0 43 0 0 0 44 0 0 0 45 0 0 0 4 1 0 0
print_m256i sss1 ->: 211 74 71 0 167 79 72 0 123 84 73 0 44 79 4 1 47 205 98 0 3 210 99 0 215 214 100 0 44 79 4 1
print_m256i -> sss1: 71 0 0 0 72 0 0 0 73 0 0 0 4 1 0 0 98 0 0 0 99 0 0 0 100 0 0 0 4 1 0 0
print_m256i sss0: 16 0 17 0 18 0 4 1 0 0 0 0 0 0 0 0 43 0 44 0 45 0 4 1 0 0 0 0 0 0 0 0
print_m256i sss1: 71 0 72 0 73 0 4 1 0 0 0 0 0 0 0 0 98 0 99 0 100 0 4 1 0 0 0 0 0 0 0 0
print_m256i sss0: 16 17 18 255 0 0 0 0 0 0 0 0 0 0 0 0 43 44 45 255 0 0 0 0 0 0 0 0 0 0 0 0
print_m256i sss1: 71 72 73 255 0 0 0 0 0 0 0 0 0 0 0 0 98 99 100 255 0 0 0 0 0 0 0 0 0 0 0 0
print_m128i e0: 16 17 18 255 0 0 0 0 0 0 0 0 0 0 0 0
print_m128i e1: 43 44 45 255 0 0 0 0 0 0 0 0 0 0 0 0
print_m128i e2: 71 72 73 255 0 0 0 0 0 0 0 0 0 0 0 0
print_m128i e3: 98 99 100 255 0 0 0 0 0 0 0 0 0 0 0 0
print_uint32 lineOut0: 16 17 18 255
print_uint32 lineOut1: 43 44 45 255
print_uint32 lineOut2: 71 72 73 255
print_uint32 lineOut3: 98 99 100 255
```


## Vertical pass vectorized algorithm on RGBA data

Input data is stored as
```
input = [
      r[0, 0], g[0, 0], b[0, 0], a[0, 0], r[0, 1], g[0, 1], b[0, 1], a[0, 1], r[0, 2], g[0, 2], b[0, 2], a[0, 2], ...
      r[1, 0], g[1, 0], b[1, 0], a[1, 0], r[1, 1], g[1, 1], b[1, 1], a[1, 1], r[1, 2], g[1, 2], b[1, 2], a[1, 2], ...
      ...
      r[K - 1, 0], g[K - 1, 0], b[K - 1, 0], a[K - 1, 0], r[K - 1, 1], g[K - 1, 1], b[K - 1, 1], a[K - 1, 1], r[K - 1, 2], g[K - 1, 2], b[K - 1, 2], a[K - 1, 2], ...
      ...
]
```

Weights are float values computed for each output pixel and rescaled to uint16:
```
weights[i] = [w[i, 0], w[i, 1], ..., w[i, K - 1]]
```

We want to compute the output as following:
```
output = [
      oR[0, 0], oG[0, 0], oB[0, 0], oA[0, 0], oR[0, 1], oG[0, 1], oB[0, 1], oA[0, 1], ...
]
```
where
```
oR[j, i] = r[ymin[j], i] * w[j, 0] + r[ymin[j] + 1, i] * w[j, 1] + ... + r[ymin[j] + K - 1, i] * w[j, K - 1]
oG[j, i] = g[ymin[j], i] * w[j, 0] + g[ymin[j] + 1, i] * w[j, 1] + ... + g[ymin[j] + K - 1, i] * w[j, K - 1]
oB[j, i] = b[ymin[j], i] * w[j, 0] + b[ymin[j] + 1, i] * w[j, 1] + ... + b[ymin[j] + K - 1, i] * w[j, K - 1]
```

### Vectorized version

As data format is RGBA with R,G,B,A being uint8, we can encode 4 values as a single uint32 value.

Working accumulating register, avx2 = 32 uint8 places
```
reg = [0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 0 0]
```

We can split K (size of weight vector for a given output index) as a sum: `K = m * 2 + k`.
We load and process 2 weights values in another loop ("block 2") and finally we process 1 weights value in the final loop ("block 1").

1) As we are doing computations in integer dtype, we add the offset (=`1 << (weights_precision - 1)`):
```
reg = [
      0 128 0 0 0 128 0 0 | 0 128 0 0 0 128 0 0 | 0 128 0 0 0 128 0 0 | 0 128 0 0 0 128 0 0
]
```

2) Load weights. For "block 2" we load 2 int16 values `(w0, w1)`. Each value then will be represented in the register with uint8 values `wl_0` and `wh_0`:
```
w01 = [
      wl_0 wh_0 wl_1 wh_1 wl_0 wh_0 wl_1 wh_1 | ... | ... | wl_0 wh_0 wl_1 wh_1 wl_0 wh_0 wl_1 wh_1
]
```
And in case of "block 1" we will load only 1 int16 value `w0`:
```
w0 = [
      wl_0 wh_0 0 0 wl_0 wh_0 0 0 | ... | ... | wl_0 wh_0 0 0 wl_0 wh_0 0 0
]
```

3) Load source data. Each RGBA pixel has 4 uint8 size, so half of 256-bits register (=16 uint8 places) can be filled with 4 pixels. To fill 32 uint8 places (=256 bits) we can load 8 pixels from each line, e.g. `r0`-`r7` and `rr0`-`rr7` where `ri` is a red value from line0 and `rri` is a red value from line1.

For vertical pass we need to compute together values from different lines.

```
line0 = [
      r0 g0 b0 a0 r1 g1 b1 a1 | r2 g2 b2 a2 r3 g3 b3 a3 | r4 g4 b4 a4 r5 g5 b5 a5 | r6 g6 b6 a6 r7 g7 b7 a7
]

line1 = [
      rr0 gg0 bb0 aa0 rr1 gg1 bb1 aa1 | rr2 gg2 bb2 aa2 rr3 gg3 bb3 aa3 | rr4 gg4 bb4 aa4 rr5 gg5 bb5 aa5 | rr6 gg6 bb6 aa6 rr7 gg7 bb7 aa7
]
```

We process 8 pixels within each line in parallel and two lines contribute to the output: `r0 * w0 + rr0 * w1`.
When it remains less then 8 pixels we can process 2 pixels within each line in parallel and finally just 1 pixel.


4) We loaded weights 2 values as `(wl_0 wh_0 wl_1 wh_1)` thus we have to split and shuffle the source data such we could correctly multiply `r0 * w0 + rr0 * w1` and `g0 * w0 + gg0 * w1`.
```
data_01_ll = [
      r0 0 rr0 0 g0 0 gg0 0 | b0 0 bb0 0 a0 0 aa0 0 | r1 0 rr1 0 g1 0 gg1 0 | b1 0 bb1 0 a1 0 aa1 0
]
data_01_lh = [
      r2 0 rr2 0 g2 0 gg2 0 | b2 0 bb2 0 a2 0 aa2 0 | r3 0 rr3 0 g3 0 gg3 0 | b3 0 bb3 0 a3 0 aa3 0
]
data_01_hl = [
      r4 0 rr4 0 g4 0 gg4 0 | b4 0 bb4 0 a4 0 aa4 0 | r5 0 rr5 0 g5 0 gg5 0 | b5 0 bb5 0 a5 0 aa5 0
]
data_01_hh = [
      r6 0 rr6 0 g6 0 gg6 0 | b6 0 bb6 0 a6 0 aa6 0 | r7 0 rr7 0 g7 0 gg7 0 | b7 0 bb7 0 a7 0 aa7 0
]
```

For "block 1" we will have
```
data_0_ll = [
      r0 0 0 0 g0 0 0 0 | b0 0 0 0 a0 0 0 0 | r1 0 0 0 g1 0 0 0 | b1 0 0 0 a1 0 0 0
]
data_0_lh = [
      r2 0 0 0 g2 0 0 0 | b2 0 0 0 a2 0 0 0 | r3 0 0 0 g3 0 0 0 | b3 0 0 0 a3 0 0 0
]
data_0_hl = [
      r4 0 0 0 g4 0 0 0 | b4 0 0 0 a4 0 0 0 | r5 0 0 0 g5 0 0 0 | b5 0 0 0 a5 0 0 0
]
data_0_hh = [
      r6 0 0 0 g6 0 0 0 | b6 0 0 0 a6 0 0 0 | r7 0 0 0 g7 0 0 0 | b7 0 0 0 a7 0 0 0
]
```

5) Multiply and add weights and source data using integer 32-bits precision. Integer 32-bits precision means the output will take 4 placeholders (a b c d).

```
# w01 = [
#       wl_0 wh_0 wl_1 wh_1 wl_0 wh_0 wl_1 wh_1 | ... | ... | wl_0 wh_0 wl_1 wh_1 wl_0 wh_0 wl_1 wh_1
# ]

out01_ll = data_01_ll * w01

out01_ll = [
      (r0 0) * (wl_0 wh_0) + (rr0 0) * (wl_1 wh_1), (g0 0) * (wl_0 wh_0) + (gg0 0) * (wl_1 wh_1) |
      (b0 0) * (wl_0 wh_0) + (bb0 0) * (wl_1 wh_1), (a0 0) * (wl_0 wh_0), (aa0 0) * (wl_1 wh_1)  |
      (r1 0) * (wl_0 wh_0) + (rr1 0) * (wl_1 wh_1), (g1 0) * (wl_0 wh_0) + (gg1 0) * (wl_1 wh_1) |
      (b1 0) * (wl_0 wh_0) + (bb1 0) * (wl_1 wh_1), (a1 0) * (wl_0 wh_0) + (aa1 0) * (wl_1 wh_1)
]
```
where `(pi 0) * (wl_j wh_j) + (ppi 0) * (wl_n wh_n) = (out_0, out_1, out_2, out_3)`.

```
out01_lh = data_01_lh * w01

out01_lh = [
      (r2 0) * (wl_0 wh_0) + (rr2 0) * (wl_1 wh_1), (g2 0) * (wl_0 wh_0) + (gg2 0) * (wl_1 wh_1) |
      (b2 0) * (wl_0 wh_0) + (bb2 0) * (wl_1 wh_1), (a2 0) * (wl_0 wh_0), (aa2 0) * (wl_1 wh_1)  |
      (r3 0) * (wl_0 wh_0) + (rr3 0) * (wl_1 wh_1), (g3 0) * (wl_0 wh_0) + (gg3 0) * (wl_1 wh_1) |
      (b3 0) * (wl_0 wh_0) + (bb3 0) * (wl_1 wh_1), (a3 0) * (wl_0 wh_0) + (aa3 0) * (wl_1 wh_1)
]

out01_hl = ...
out01_hh = ...
```

For "block 1" we will have
```
out0_ll = [
      (r0 0) * (wl_0 wh_0), (g0 0) * (wl_0 wh_0) |
      (b0 0) * (wl_0 wh_0), (a0 0) * (wl_0 wh_0) |
      (r1 0) * (wl_0 wh_0), (g1 0) * (wl_0 wh_0) |
      (b1 0) * (wl_0 wh_0), (a1 0) * (wl_0 wh_0)
]

out0_lh = ...
out0_hl = ...
out0_hh = ...
```
Here each element like `(r0 0) * (wl_0 wh_0)` represent int32 and takes 4 placeholders.

6) Shift back the output integer values (`output = (output >> weights_precision)`)

```
out01_ll = out01_ll >> weights_precision
out01_lh = out01_lh >> weights_precision
out01_hl = out01_hl >> weights_precision
out01_hh = out01_hh >> weights_precision
```

7) Convert packed signed 32-bit integers to packed 16-bit integers using signed saturation

```
(a a a a b b b b | c c c c d d d d) -> (a' a' b' b' c' c' d' d')

(out01_ll, out01_lh) -> out_01_l
(out01_hl, out01_hh) -> out_01_h
```

9) Convert packed signed 16-bit integers to packed 8-bit integers using unsigned saturation

```
(a a b b | c c d d) -> (a' b' c' d')

(out01_l, out01_h) -> out_01
```

10) Write the output into single uint32
```
(a b c d) -> x_uint32
```


## Vectorize `void basic_loop_aa_vertical<uint8_t>` from `UpSampleKernel.cpp`

- Baseline:
```
PIL version:  9.0.0.post1
[----------------------------------------------- Resize ----------------------------------------------]
                                                                       |  torch (2.1.0a0+git0968a5d) PR
1 threads: --------------------------------------------------------------------------------------------
      4 torch.uint8 channels_last bilinear 256 -> (224, 256) aa=True   |              707.5
      4 torch.uint8 channels_first bilinear 256 -> (224, 256) aa=True  |              719.9

Times are in microseconds (us).
```

vs Vectorized version `ImagingResampleVerticalConvolution8u`:
```
PIL version:  9.0.0.post1
[------------------------------------------------------------ Resize -----------------------------------------------------------]
                                                                        |  Pillow (9.0.0.post1)  |  torch (2.1.0a0+gitcc42a3f) PR
1 threads: ----------------------------------------------------------------------------------------------------------------------
      4 torch.uint8 channels_last bilinear 256 -> (224, 256) aa=True    |                        |               56.1
      4 torch.uint8 channels_first bilinear 256 -> (224, 256) aa=True   |                        |              187.4
```
