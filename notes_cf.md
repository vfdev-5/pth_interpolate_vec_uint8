# WIP on Vectorized bilinear interpolation uint8, channels first

- Install Pillow-SIMD
```
pip uninstall -y pillow && CC="cc -mavx2" pip install --no-cache-dir --force-reinstall pillow-simd
```

## Check consistency

```
cd /tmp/pth/interpolate_vec_uint8/ && python -u verif_interp2.py
```

## Results

- Baseline

```
cd /tmp/pth/interpolate_vec_uint8/ && python -u check_interp_cf.py


PIL version:  9.0.0.post1
[----------------------------------------------- Resize ----------------------------------------------]
                                                                       |  torch (2.1.0a0+git8d955df) PR
1 threads: --------------------------------------------------------------------------------------------
      3 torch.uint8 channels_first bilinear 256 -> (224, 224) aa=True  |               1.2

Times are in milliseconds (ms).
```

- Using "Pillow avx channels_last + copy" codepath (output/20230321-145513-pr_vs_nightly-speedup.md)
```
[-------------------------------------------------------------------------------------------------- Resize -----------------------------------------------------------------------]
                                                                                 |  Pillow (9.0.0.post1)  |  torch (2.1.0a0+git8d955df) PR  |  torch (2.1.0a0+git5309c44) nightly
1 threads: -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
      3 torch.uint8 channels_first bilinear (256, 256) -> (224, 224) aa=True     |   128.337 (+-1.344)    |        296.472 (+-1.524)        |          319.583 (+-8.983)

```

- Vertical pass: Block 4 + Block 1
```
Num threads: 1

PIL version:  9.0.0.post1
[----------------------------------------------- Resize ----------------------------------------------]
                                                                       |  torch (2.1.0a0+gitce4be01) PR
1 threads: --------------------------------------------------------------------------------------------
      3 torch.uint8 channels_first bilinear 256 -> (224, 224) aa=True  |              867.4

Times are in microseconds (us).
```

- Vertical pass: Block 8 + Block 4 + Block 1
```
Num threads: 1

PIL version:  9.0.0.post1
[----------------------------------------------- Resize ----------------------------------------------]
                                                                       |  torch (2.1.0a0+gitd6e220c) PR
1 threads: --------------------------------------------------------------------------------------------
      3 torch.uint8 channels_first bilinear 256 -> (224, 224) aa=True  |              829.1

[----------------------------------------------------------- Resize -----------------------------------------------------------]
                                                                       |  Pillow (9.0.0.post1)  |  torch (2.1.0a0+gitd6e220c) PR
1 threads: ---------------------------------------------------------------------------------------------------------------------
      3 torch.uint8 channels_first bilinear 256 -> (224, 256) aa=True  |          54.1          |               96.8

[----------------------------------------------------------- Resize -----------------------------------------------------------]
                                                                       |  Pillow (9.0.0.post1)  |  torch (2.1.0a0+nightly)
1 threads: ---------------------------------------------------------------------------------------------------------------------
      3 torch.uint8 channels_first bilinear 256 -> (224, 256) aa=True  |          57.8          |              219.3

Times are in microseconds (us).
```

