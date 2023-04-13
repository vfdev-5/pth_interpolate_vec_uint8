

```bash
git clone git@github.com:vfdev-5/pth_interpolate_vec_uint8.git

docker run --rm -it \
    -v $PWD:/ws -w /ws \
    --ipc=host \
    --cap-add=SYS_ADMIN \
    --cap-add=SYS_PTRACE \
    --network=host --security-opt seccomp:unconfined --ipc=host \
    python:3.9-slim-bullseye /bin/bash

pip install torch --index-url https://download.pytorch.org/whl/cpu

python -c "import torch; print(torch.__config__.show())"

pip install Pillow numpy fire

cd pth_interpolate_vec_uint8

# Quick test:
python -u run_bench_interp_custom_cases.py output/test.log --min_run_time=0.1 --with_torchvision

# Benchmarks:
export fileprefix=$(date "+%Y%m%d-%H%M%S") && python -u run_bench_interp_custom_cases.py output/${fileprefix}-PT2.0.pkl --tag=PT2.0 --with_torchvision &> output/${fileprefix}-PT2.0.log

export filename="20230316-135217-PT2.0" && python perf_results_compute_speedup.py output/${filename}_vs_torchvision_speedup.md "['output/${filename}.pkl',]" --col1="torch (2.0.0+cpu) PT2.0" --col2="torchvision resize" --description="Speed-up: PTH vs TV"
```


Results:

```
Timestamp: 20230316-135223
Torch version: 2.0.0+cpu
Torch config: PyTorch built with:
  - GCC 9.3
  - C++ Version: 201703
  - Intel(R) oneAPI Math Kernel Library Version 2022.2-Product Build 20220804 for Intel(R) 64 architecture applications
  - Intel(R) MKL-DNN v2.7.3 (Git Hash 6dbeffbae1f23cbbeae17adb7b5b13f1f37c080e)
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - LAPACK is enabled (usually provided by MKL)
  - NNPACK is enabled
  - CPU capability usage: NO AVX
  - Build settings: BLAS_INFO=mkl, BUILD_TYPE=Release, CXX_COMPILER=/opt/rh/devtoolset-9/root/usr/bin/c++, CXX_FLAGS= -D_GLIBCXX_USE_CXX11_ABI=0 -fabi-version=11 -Wno-deprecated -fvisibil
ity-inlines-hidden -DUSE_PTHREADPOOL -DNDEBUG -DUSE_KINETO -DLIBKINETO_NOCUPTI -DLIBKINETO_NOROCTRACER -DUSE_FBGEMM -DUSE_QNNPACK -DUSE_PYTORCH_QNNPACK -DUSE_XNNPACK -DSYMBOLICATE_MOBILE_
DEBUG_HANDLE -O2 -fPIC -Wall -Wextra -Werror=return-type -Werror=non-virtual-dtor -Werror=bool-operation -Wnarrowing -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wn
o-unknown-pragmas -Wunused-local-typedefs -Wno-unused-parameter -Wno-unused-function -Wno-unused-result -Wno-strict-overflow -Wno-strict-aliasing -Wno-error=deprecated-declarations -Wno-s
tringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=redundant-decls -Wno-error=old-style-cast -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-unini
tialized -fno-math-errno -fno-trapping-math -Werror=format -Werror=cast-function-type -Wno-stringop-overflow, LAPACK_INFO=mkl, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, TORCH
_DISABLE_GPU_ASSERTS=ON, TORCH_VERSION=2.0.0, USE_CUDA=0, USE_CUDNN=OFF, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=ON, USE_MKLDNN=ON, USE_MPI=OFF, USE_NCCL=OFF, USE_NNPAC
K=ON, USE_OPENMP=ON, USE_ROCM=OFF,

Num threads: 1

PIL version:  9.4.0


[---------------------------------------------------------------------------------- Resize ---------------------------------------------------------------------------------]
                                                                                 |  Pillow (9.4.0)  |  torch (2.0.0+cpu) PT2.0  |  torchvision resize  |  Speed-up: PTH vs TV
1 threads: ------------------------------------------------------------------------------------------------------------------------------------------------------------------
      3 torch.uint8 channels_last bilinear (64, 64) -> (224, 224) aa=True        |      1165.0      |           4096.5          |         5535.9       |          1.4
      3 torch.uint8 channels_last bilinear (224, 224) -> (270, 268) aa=True      |      2394.5      |           8030.3          |         9019.4       |          1.1
      3 torch.uint8 channels_last bilinear (256, 256) -> (1024, 1024) aa=True    |     22463.3      |          79869.0          |       134778.1       |          1.7
      3 torch.uint8 channels_last bilinear (224, 224) -> (64, 64) aa=True        |       773.9      |           1827.6          |         1992.6       |          1.1
      3 torch.uint8 channels_last bilinear (270, 268) -> (224, 224) aa=True      |      2484.6      |           7023.5          |         7557.0       |          1.1
      3 torch.uint8 channels_last bilinear (1024, 1024) -> (256, 256) aa=True    |     14698.8      |          30701.2          |        31766.2       |          1.0
      3 torch.uint8 channels_last bilinear (64, 64) -> (224, 224) aa=False       |                  |           4088.7          |         5690.3       |          1.4
      3 torch.uint8 channels_last bilinear (224, 224) -> (270, 268) aa=False     |                  |           8002.9          |         8352.5       |          1.0
      3 torch.uint8 channels_last bilinear (256, 256) -> (1024, 1024) aa=False   |                  |          79812.4          |       111806.3       |          1.4
      3 torch.uint8 channels_last bilinear (224, 224) -> (64, 64) aa=False       |                  |           1213.8          |          824.4       |          0.7
      3 torch.uint8 channels_last bilinear (270, 268) -> (224, 224) aa=False     |                  |           6662.1          |         6029.8       |          0.9
      3 torch.uint8 channels_last bilinear (1024, 1024) -> (256, 256) aa=False   |                  |          19328.3          |        12560.8       |          0.6
      3 torch.uint8 channels_first bilinear (64, 64) -> (224, 224) aa=True       |      1163.0      |           2540.4          |         3858.4       |          1.5
      3 torch.uint8 channels_first bilinear (224, 224) -> (270, 268) aa=True     |      2380.3      |           5862.3          |         6542.7       |          1.1
      3 torch.uint8 channels_first bilinear (256, 256) -> (1024, 1024) aa=True   |     22422.0      |          47722.2          |        69475.9       |          1.5
      3 torch.uint8 channels_first bilinear (224, 224) -> (64, 64) aa=True       |       773.6      |           1591.8          |         1769.6       |          1.1
      3 torch.uint8 channels_first bilinear (270, 268) -> (224, 224) aa=True     |      2483.1      |           5559.1          |         5814.7       |          1.0
      3 torch.uint8 channels_first bilinear (1024, 1024) -> (256, 256) aa=True   |     14645.5      |          27005.1          |        28425.9       |          1.1
      3 torch.uint8 channels_first bilinear (64, 64) -> (224, 224) aa=False      |                  |           2529.4          |         6694.3       |          2.6
      3 torch.uint8 channels_first bilinear (224, 224) -> (270, 268) aa=False    |                  |           5868.4          |        10322.9       |          1.8
      3 torch.uint8 channels_first bilinear (256, 256) -> (1024, 1024) aa=False  |                  |          47667.4          |       133117.0       |          2.8
      3 torch.uint8 channels_first bilinear (224, 224) -> (64, 64) aa=False      |                  |           1100.2          |         1577.7       |          1.4
      3 torch.uint8 channels_first bilinear (270, 268) -> (224, 224) aa=False    |                  |           5209.9          |         7724.3       |          1.5
      3 torch.uint8 channels_first bilinear (1024, 1024) -> (256, 256) aa=False  |                  |          17449.9          |        37994.5       |          2.2

Times are in microseconds (us).
```


```
Timestamp: 20230412-130655
Torch version: 2.0.0+cpu
Torch config: PyTorch built with:
  - GCC 9.3
  - C++ Version: 201703
  - Intel(R) oneAPI Math Kernel Library Version 2022.2-Product Build 20220804 for Intel(R) 64 architecture applications
  - Intel(R) MKL-DNN v2.7.3 (Git Hash 6dbeffbae1f23cbbeae17adb7b5b13f1f37c080e)
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - LAPACK is enabled (usually provided by MKL)
  - NNPACK is enabled
  - CPU capability usage: NO AVX
  - Build settings: BLAS_INFO=mkl, BUILD_TYPE=Release, CXX_COMPILER=/opt/rh/devtoolset-9/root/usr/bin/c++, CXX_FLAGS= -D_GLIBCXX_USE_CXX11_ABI=0 -fabi-version=11 -Wno-deprecated -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -DNDEBUG -DUSE_KINETO -DLIBKINETO_NOCUPTI -DLIBKINETO_NOROCTRACER -DUSE_FBGEMM -DUSE_QNNPACK -DUSE_PYTORCH_QNNPACK -DUSE_XNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -O2 -fPIC -Wall -Wextra -Werror=return-type -Werror=non-virtual-dtor -Werror=bool-operation -Wnarrowing -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wunused-local-typedefs -Wno-unused-parameter -Wno-unused-function -Wno-unused-result -Wno-strict-overflow -Wno-strict-aliasing -Wno-error=deprecated-declarations -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=redundant-decls -Wno-error=old-style-cast -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Werror=cast-function-type -Wno-stringop-overflow, LAPACK_INFO=mkl, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, TORCH_DISABLE_GPU_ASSERTS=ON, TORCH_VERSION=2.0.0, USE_CUDA=0, USE_CUDNN=OFF, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=ON, USE_MKLDNN=ON, USE_MPI=OFF, USE_NCCL=OFF, USE_NNPACK=ON, USE_OPENMP=ON, USE_ROCM=OFF,

Num threads: 1

PIL version:  9.5.0
[----------------------------------------------------------- Resize -----------------------------------------------------------]
                                                                              |  torch (2.0.0+cpu) PT2.0  |   torchvision resize
1 threads: ---------------------------------------------------------------------------------------------------------------------
      3 torch.uint8 channels_last bilinear (224, 224) -> (64, 64) aa=False    |    1198.396 (+-10.264)    |   654.474 (+-19.919)
      3 torch.uint8 channels_last bilinear (270, 268) -> (224, 224) aa=False  |    6628.365 (+-41.949)    |  4422.819 (+-26.323)
      3 torch.uint8 channels_last bilinear (256, 256) -> (224, 224) aa=False  |    6452.640 (+-34.575)    |  4373.281 (+-32.027)

[------------------------------------------------------------- Resize -------------------------------------------------------------]
                                                                                |  torch (2.0.0+cpu) PT2.0  |    torchvision resize
1 threads: -------------------------------------------------------------------------------------------------------------------------
      3 torch.uint8 channels_last bilinear (224, 224) -> (64, 64) aa=False      |    1198.173 (+-18.915)    |    672.927 (+-7.398)
      3 torch.uint8 channels_last bilinear (270, 268) -> (224, 224) aa=False    |    6636.571 (+-29.313)    |   4425.708 (+-18.328)
      3 torch.uint8 channels_last bilinear (1024, 1024) -> (256, 256) aa=False  |    19308.062 (+-94.516)   |  10409.677 (+-304.731)

[------------------------------------------------------------------------ Resize -----------------------------------------------------------------------]
                                                                               |     Pillow (9.5.0)    |  torch (2.0.0+cpu) PT2.0  |   torchvision resize
1 threads: ----------------------------------------------------------------------------------------------------------------------------------------------
      3 torch.uint8 channels_first bilinear (224, 224) -> (64, 64) aa=True     |   772.394 (+-9.628)   |    1578.853 (+-27.930)    |  1617.512 (+-18.584)
      3 torch.uint8 channels_first bilinear (270, 268) -> (224, 224) aa=True   |  2418.122 (+-15.719)  |    5542.286 (+-48.034)    |  4207.568 (+-27.673)
      3 torch.uint8 channels_first bilinear (256, 256) -> (224, 224) aa=True   |  2268.907 (+-40.491)  |    5383.758 (+-22.676)    |  4013.067 (+-29.038)
      3 torch.uint8 channels_first bilinear (224, 224) -> (64, 64) aa=False    |                       |     1094.699 (+-4.541)    |  1461.899 (+-14.930)
      3 torch.uint8 channels_first bilinear (270, 268) -> (224, 224) aa=False  |                       |    5190.978 (+-23.072)    |  6178.028 (+-22.463)
      3 torch.uint8 channels_first bilinear (256, 256) -> (224, 224) aa=False  |                       |    5008.875 (+-25.759)    |  6077.749 (+-27.984)
      3 torch.uint8 channels_last bilinear (224, 224) -> (64, 64) aa=True      |   772.950 (+-2.355)   |    1797.769 (+-13.817)    |  1853.699 (+-22.517)
      3 torch.uint8 channels_last bilinear (270, 268) -> (224, 224) aa=True    |  2417.311 (+-14.537)  |    6993.483 (+-27.891)    |  5948.212 (+-19.839)
      3 torch.uint8 channels_last bilinear (256, 256) -> (224, 224) aa=True    |  2266.818 (+-17.316)  |    6847.144 (+-29.578)    |  5765.666 (+-24.931)
      3 torch.uint8 channels_last bilinear (224, 224) -> (64, 64) aa=False     |                       |     1201.132 (+-8.392)    |   683.725 (+-8.224)
      3 torch.uint8 channels_last bilinear (270, 268) -> (224, 224) aa=False   |                       |    6642.182 (+-28.036)    |  4427.898 (+-19.370)
      3 torch.uint8 channels_last bilinear (256, 256) -> (224, 224) aa=False   |                       |    6460.126 (+-30.211)    |  4371.593 (+-18.639)

Times are in microseconds (us).
```


```
export fileprefix=$(date "+%Y%m%d-%H%M%S") && python -u run_bench_interp_custom_cases.py output/${fileprefix}-PT2.0.pkl --tag=PT2.0 --with_torchvision --num_threads=4 &> output/${fileprefix}-PT2.0.log

[------------------------------------------------------------------------ Resize ------------------------------------------------------------------------]
                                                                               |     Pillow (9.5.0)    |  torch (2.0.0+cpu) PT2.0  |   torchvision resize
4 threads: -----------------------------------------------------------------------------------------------------------------------------------------------
      3 torch.uint8 channels_first bilinear (224, 224) -> (64, 64) aa=True     |   773.336 (+-3.005)   |    1146.438 (+-135.516)   |  1889.563 (+-308.527)
      3 torch.uint8 channels_first bilinear (270, 268) -> (224, 224) aa=True   |  2415.853 (+-38.638)  |    3669.654 (+-296.317)   |  4094.434 (+-539.771)
      3 torch.uint8 channels_first bilinear (256, 256) -> (224, 224) aa=True   |  2266.887 (+-16.078)  |    3608.822 (+-321.846)   |  3754.457 (+-573.692)
      3 torch.uint8 channels_first bilinear (224, 224) -> (64, 64) aa=False    |                       |     855.721 (+-98.304)    |  2279.939 (+-444.201)
      3 torch.uint8 channels_first bilinear (270, 268) -> (224, 224) aa=False  |                       |    3379.701 (+-316.814)   |  4473.115 (+-571.146)
      3 torch.uint8 channels_first bilinear (256, 256) -> (224, 224) aa=False  |                       |    3358.919 (+-326.584)   |  4076.735 (+-543.611)
      3 torch.uint8 channels_last bilinear (224, 224) -> (64, 64) aa=True      |   772.118 (+-11.325)  |    1384.033 (+-172.656)   |  1725.062 (+-339.790)
      3 torch.uint8 channels_last bilinear (270, 268) -> (224, 224) aa=True    |  2418.731 (+-15.845)  |    4420.662 (+-309.929)   |  4965.778 (+-655.978)
      3 torch.uint8 channels_last bilinear (256, 256) -> (224, 224) aa=True    |  2265.271 (+-41.195)  |    4357.622 (+-836.751)   |  5223.831 (+-949.238)
      3 torch.uint8 channels_last bilinear (224, 224) -> (64, 64) aa=False     |                       |    1019.681 (+-101.090)   |  856.581 (+-208.634)
      3 torch.uint8 channels_last bilinear (270, 268) -> (224, 224) aa=False   |                       |    4315.346 (+-319.867)   |  5365.436 (+-605.186)
      3 torch.uint8 channels_last bilinear (256, 256) -> (224, 224) aa=False   |                       |    4076.089 (+-316.879)   |  5447.182 (+-552.134)

Times are in microseconds (us).
```


```
[--------------------------------------------------------------------------------------- Resize ---------------------------------------------------------------------------------------]
                                                                                 |      Pillow (9.5.0)     |  torch (2.0.0+cpu) PT2.0  |    torchvision resize    |  Speed-up: PTH vs TV
1 threads: -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      3 torch.uint8 channels_first bilinear (64, 64) -> (224, 224) aa=True       |    1162.659 (+-9.277)   |    2521.496 (+-19.013)    |   2204.474 (+-23.472)    |    0.874 (+-0.000)
      3 torch.uint8 channels_first bilinear (224, 224) -> (270, 268) aa=True     |   2381.221 (+-16.570)   |    5845.022 (+-82.954)    |   4194.588 (+-33.179)    |    0.718 (+-0.000)
      3 torch.uint8 channels_first bilinear (256, 256) -> (1024, 1024) aa=True   |  22443.577 (+-147.357)  |   47697.633 (+-299.258)   |  73080.779 (+-332.071)   |    1.532 (+-0.000)
      3 torch.uint8 channels_first bilinear (224, 224) -> (64, 64) aa=True       |    771.892 (+-3.193)    |    1581.846 (+-15.787)    |   1616.360 (+-18.834)    |    1.022 (+-0.000)
      3 torch.uint8 channels_first bilinear (270, 268) -> (224, 224) aa=True     |   2416.589 (+-42.885)   |    5536.582 (+-35.725)    |   4220.128 (+-32.747)    |    0.762 (+-0.000)
      3 torch.uint8 channels_first bilinear (256, 256) -> (224, 224) aa=True     |   2268.977 (+-24.074)   |    5377.582 (+-35.796)    |   4028.528 (+-26.684)    |    0.749 (+-0.000)
      3 torch.uint8 channels_first bilinear (1024, 1024) -> (256, 256) aa=True   |   14552.534 (+-63.907)  |   26995.708 (+-157.719)   |  26244.053 (+-505.652)   |    0.972 (+-0.000)
      3 torch.uint8 channels_first bilinear (64, 64) -> (224, 224) aa=False      |                         |    2518.830 (+-26.949)    |   5047.908 (+-36.319)    |    2.004 (+-0.000)
      3 torch.uint8 channels_first bilinear (224, 224) -> (270, 268) aa=False    |                         |    5835.826 (+-37.958)    |   7991.453 (+-39.369)    |    1.369 (+-0.000)
      3 torch.uint8 channels_first bilinear (256, 256) -> (1024, 1024) aa=False  |                         |   47617.245 (+-222.802)   |  99347.405 (+-9938.670)  |    2.086 (+-0.000)
      3 torch.uint8 channels_first bilinear (224, 224) -> (64, 64) aa=False      |                         |     1093.910 (+-5.958)    |   1462.177 (+-17.812)    |    1.337 (+-0.000)
      3 torch.uint8 channels_first bilinear (270, 268) -> (224, 224) aa=False    |                         |    5216.105 (+-23.758)    |   6179.793 (+-27.608)    |    1.185 (+-0.000)
      3 torch.uint8 channels_first bilinear (256, 256) -> (224, 224) aa=False    |                         |    5031.895 (+-20.022)    |   6053.712 (+-30.991)    |    1.203 (+-0.000)
      3 torch.uint8 channels_first bilinear (1024, 1024) -> (256, 256) aa=False  |                         |    17456.630 (+-97.065)   |  21857.553 (+-1854.403)  |    1.252 (+-0.000)
      3 torch.uint8 channels_last bilinear (64, 64) -> (224, 224) aa=True        |    1160.227 (+-3.609)   |    4082.689 (+-22.202)    |   3861.223 (+-21.499)    |    0.946 (+-0.000)
      3 torch.uint8 channels_last bilinear (224, 224) -> (270, 268) aa=True      |   2384.119 (+-14.900)   |    7999.037 (+-22.943)    |   6643.047 (+-23.467)    |    0.830 (+-0.000)
      3 torch.uint8 channels_last bilinear (256, 256) -> (1024, 1024) aa=True    |   22409.429 (+-96.580)  |   79961.696 (+-171.308)   |  71110.487 (+-202.312)   |    0.889 (+-0.000)
      3 torch.uint8 channels_last bilinear (224, 224) -> (64, 64) aa=True        |    772.015 (+-4.578)    |    1800.263 (+-14.872)    |   1866.762 (+-20.758)    |    1.037 (+-0.000)
      3 torch.uint8 channels_last bilinear (270, 268) -> (224, 224) aa=True      |   2416.149 (+-16.251)   |    6997.100 (+-25.028)    |   5951.925 (+-19.468)    |    0.851 (+-0.000)
      3 torch.uint8 channels_last bilinear (256, 256) -> (224, 224) aa=True      |   2268.443 (+-14.540)   |    6847.604 (+-35.899)    |   5765.120 (+-25.442)    |    0.842 (+-0.000)
      3 torch.uint8 channels_last bilinear (1024, 1024) -> (256, 256) aa=True    |   14600.545 (+-38.523)  |   30667.736 (+-169.536)   |  29565.635 (+-214.455)   |    0.964 (+-0.000)
      3 torch.uint8 channels_last bilinear (64, 64) -> (224, 224) aa=False       |                         |    4068.408 (+-21.312)    |   4086.201 (+-18.979)    |    1.004 (+-0.000)
      3 torch.uint8 channels_last bilinear (224, 224) -> (270, 268) aa=False     |                         |    7983.073 (+-26.481)    |   5996.475 (+-13.924)    |    0.751 (+-0.000)
      3 torch.uint8 channels_last bilinear (256, 256) -> (1024, 1024) aa=False   |                         |   79798.151 (+-204.083)   |  114505.879 (+-250.330)  |    1.435 (+-0.000)
      3 torch.uint8 channels_last bilinear (224, 224) -> (64, 64) aa=False       |                         |     1202.207 (+-5.063)    |    674.901 (+-7.845)     |    0.561 (+-0.000)
      3 torch.uint8 channels_last bilinear (270, 268) -> (224, 224) aa=False     |                         |    6639.229 (+-26.410)    |   4419.228 (+-19.705)    |    0.666 (+-0.000)
      3 torch.uint8 channels_last bilinear (256, 256) -> (224, 224) aa=False     |                         |    6464.267 (+-28.937)    |   4378.899 (+-17.474)    |    0.677 (+-0.000)
      3 torch.uint8 channels_last bilinear (1024, 1024) -> (256, 256) aa=False   |                         |    19301.811 (+-88.155)   |  10439.297 (+-128.231)   |    0.541 (+-0.000)

[---------------------------------------------------------------------------------------- Resize ---------------------------------------------------------------------------------------]
                                                                                 |      Pillow (9.5.0)     |  torch (2.0.0+cpu) PT2.0  |     torchvision resize    |  Speed-up: PTH vs TV
4 threads: ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      3 torch.uint8 channels_first bilinear (64, 64) -> (224, 224) aa=True       |    1161.438 (+-5.021)   |    2070.279 (+-278.197)   |    2199.172 (+-411.388)   |    1.062 (+-0.000)
      3 torch.uint8 channels_first bilinear (224, 224) -> (270, 268) aa=True     |   2385.159 (+-18.863)   |    3753.838 (+-296.045)   |    3967.097 (+-573.030)   |    1.057 (+-0.000)
      3 torch.uint8 channels_first bilinear (256, 256) -> (1024, 1024) aa=True   |  22418.360 (+-155.276)  |   24684.685 (+-2328.687)  |   44760.830 (+-2338.789)  |    1.813 (+-0.000)
      3 torch.uint8 channels_first bilinear (224, 224) -> (64, 64) aa=True       |    771.304 (+-12.648)   |    1175.675 (+-108.534)   |    1876.239 (+-313.336)   |    1.596 (+-0.000)
      3 torch.uint8 channels_first bilinear (270, 268) -> (224, 224) aa=True     |   2421.832 (+-38.422)   |    3693.946 (+-323.162)   |    4077.606 (+-519.560)   |    1.104 (+-0.000)
      3 torch.uint8 channels_first bilinear (256, 256) -> (224, 224) aa=True     |   2272.879 (+-17.042)   |    3599.072 (+-308.874)   |    3759.007 (+-555.049)   |    1.044 (+-0.000)
      3 torch.uint8 channels_first bilinear (1024, 1024) -> (256, 256) aa=True   |  14575.356 (+-275.280)  |   14514.828 (+-1583.701)  |   18056.410 (+-1642.454)  |    1.244 (+-0.000)
      3 torch.uint8 channels_first bilinear (64, 64) -> (224, 224) aa=False      |                         |    2006.106 (+-266.245)   |    4186.242 (+-496.524)   |    2.087 (+-0.000)
      3 torch.uint8 channels_first bilinear (224, 224) -> (270, 268) aa=False    |                         |    3655.054 (+-328.735)   |    5902.464 (+-580.066)   |    1.615 (+-0.000)
      3 torch.uint8 channels_first bilinear (256, 256) -> (1024, 1024) aa=False  |                         |   24662.329 (+-2496.422)  |   79244.389 (+-2627.873)  |    3.213 (+-0.000)
      3 torch.uint8 channels_first bilinear (224, 224) -> (64, 64) aa=False      |                         |     885.135 (+-82.131)    |    1701.482 (+-446.010)   |    1.922 (+-0.000)
      3 torch.uint8 channels_first bilinear (270, 268) -> (224, 224) aa=False    |                         |    3434.543 (+-321.671)   |    4597.465 (+-556.127)   |    1.339 (+-0.000)
      3 torch.uint8 channels_first bilinear (256, 256) -> (224, 224) aa=False    |                         |    3245.340 (+-299.193)   |    4659.978 (+-566.993)   |    1.436 (+-0.000)
      3 torch.uint8 channels_first bilinear (1024, 1024) -> (256, 256) aa=False  |                         |    9948.110 (+-492.166)   |   17259.842 (+-1486.700)  |    1.735 (+-0.000)
      3 torch.uint8 channels_last bilinear (64, 64) -> (224, 224) aa=True        |    1158.137 (+-3.490)   |    2954.896 (+-376.242)   |    3528.245 (+-483.355)   |    1.194 (+-0.000)
      3 torch.uint8 channels_last bilinear (224, 224) -> (270, 268) aa=True      |   2384.988 (+-13.964)   |    4922.980 (+-352.862)   |    5525.646 (+-586.000)   |    1.122 (+-0.000)
      3 torch.uint8 channels_last bilinear (256, 256) -> (1024, 1024) aa=True    |  22424.745 (+-345.546)  |   40993.638 (+-3168.632)  |   40653.079 (+-3222.297)  |    0.992 (+-0.000)
      3 torch.uint8 channels_last bilinear (224, 224) -> (64, 64) aa=True        |    958.246 (+-69.478)   |    1609.529 (+-90.227)    |    2210.745 (+-214.632)   |    1.374 (+-0.000)
      3 torch.uint8 channels_last bilinear (270, 268) -> (224, 224) aa=True      |   2994.045 (+-52.502)   |    5139.361 (+-246.665)   |    5315.607 (+-527.112)   |    1.034 (+-0.000)
      3 torch.uint8 channels_last bilinear (256, 256) -> (224, 224) aa=True      |   2804.541 (+-38.886)   |    5103.128 (+-320.630)   |    5145.396 (+-422.129)   |    1.008 (+-0.000)
      3 torch.uint8 channels_last bilinear (1024, 1024) -> (256, 256) aa=True    |  18035.657 (+-123.132)  |   19756.529 (+-1743.070)  |   22218.457 (+-1772.135)  |    1.125 (+-0.000)
      3 torch.uint8 channels_last bilinear (64, 64) -> (224, 224) aa=False       |                         |    3303.501 (+-267.370)   |    5620.020 (+-428.549)   |    1.701 (+-0.000)
      3 torch.uint8 channels_last bilinear (224, 224) -> (270, 268) aa=False     |                         |    5789.249 (+-264.209)   |    7850.477 (+-448.321)   |    1.356 (+-0.000)
      3 torch.uint8 channels_last bilinear (256, 256) -> (1024, 1024) aa=False   |                         |   50476.069 (+-3324.810)  |  116164.765 (+-3572.311)  |    2.301 (+-0.000)
      3 torch.uint8 channels_last bilinear (224, 224) -> (64, 64) aa=False       |                         |    1152.213 (+-47.431)    |    991.313 (+-159.023)    |    0.860 (+-0.000)
      3 torch.uint8 channels_last bilinear (270, 268) -> (224, 224) aa=False     |                         |    4935.385 (+-272.005)   |    6081.713 (+-436.962)   |    1.232 (+-0.000)
      3 torch.uint8 channels_last bilinear (256, 256) -> (224, 224) aa=False     |                         |    4706.655 (+-235.813)   |    5882.973 (+-417.491)   |    1.250 (+-0.000)
      3 torch.uint8 channels_last bilinear (1024, 1024) -> (256, 256) aa=False   |                         |   13289.821 (+-458.713)   |   12258.655 (+-1115.548)  |    0.922 (+-0.000)
```

```
[---------------------------------------------------------------------------------------- Resize ---------------------------------------------------------------------------------------]
                                                                                 |      Pillow (9.5.0)     |  torch (2.0.0+cpu) PT2.0  |     torchvision resize    |  Speed-up: PTH vs TV
1 threads: ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      3 torch.uint8 channels_first bilinear (64, 64) -> (224, 224) aa=True       |   1441.872 (+-31.634)   |    3133.999 (+-41.797)    |    2650.083 (+-18.547)    |    0.846 (+-0.000)
      3 torch.uint8 channels_first bilinear (224, 224) -> (270, 268) aa=True     |   2948.083 (+-18.823)   |    7241.491 (+-28.487)    |    5062.526 (+-193.631)   |    0.699 (+-0.000)
      3 torch.uint8 channels_first bilinear (256, 256) -> (1024, 1024) aa=True   |  27729.240 (+-118.639)  |   59187.996 (+-159.481)   |   86632.115 (+-208.828)   |    1.464 (+-0.000)
      3 torch.uint8 channels_first bilinear (224, 224) -> (64, 64) aa=True       |    959.863 (+-3.092)    |    1959.764 (+-15.033)    |    1945.851 (+-17.779)    |    0.993 (+-0.000)
      3 torch.uint8 channels_first bilinear (270, 268) -> (224, 224) aa=True     |   2996.568 (+-18.702)   |    6876.854 (+-29.206)    |    5041.044 (+-18.342)    |    0.733 (+-0.000)
      3 torch.uint8 channels_first bilinear (256, 256) -> (224, 224) aa=True     |   2809.417 (+-19.196)   |    6676.665 (+-25.245)    |    4840.393 (+-23.858)    |    0.725 (+-0.000)
      3 torch.uint8 channels_first bilinear (1024, 1024) -> (256, 256) aa=True   |   17794.827 (+-99.422)  |   33402.649 (+-121.194)   |   30894.338 (+-1091.873)  |    0.925 (+-0.000)
      3 torch.uint8 channels_first bilinear (64, 64) -> (224, 224) aa=False      |                         |    3123.706 (+-18.243)    |    6165.287 (+-24.642)    |    1.974 (+-0.000)
      3 torch.uint8 channels_first bilinear (224, 224) -> (270, 268) aa=False    |                         |    7237.290 (+-26.091)    |    9618.087 (+-41.331)    |    1.329 (+-0.000)
      3 torch.uint8 channels_first bilinear (256, 256) -> (1024, 1024) aa=False  |                         |   59074.332 (+-180.395)   |  120177.328 (+-2954.017)  |    2.034 (+-0.000)
      3 torch.uint8 channels_first bilinear (224, 224) -> (64, 64) aa=False      |                         |     1355.573 (+-5.806)    |    1708.724 (+-17.911)    |    1.261 (+-0.000)
      3 torch.uint8 channels_first bilinear (270, 268) -> (224, 224) aa=False    |                         |    6429.536 (+-28.004)    |    7396.379 (+-26.848)    |    1.150 (+-0.000)
      3 torch.uint8 channels_first bilinear (256, 256) -> (224, 224) aa=False    |                         |    6199.967 (+-33.862)    |    7285.841 (+-25.835)    |    1.175 (+-0.000)
      3 torch.uint8 channels_first bilinear (1024, 1024) -> (256, 256) aa=False  |                         |   21562.549 (+-111.605)   |   25778.662 (+-2399.403)  |    1.196 (+-0.000)
      3 torch.uint8 channels_last bilinear (64, 64) -> (224, 224) aa=True        |    1442.826 (+-3.937)   |    5051.518 (+-20.701)    |    4728.632 (+-19.484)    |    0.936 (+-0.000)
      3 torch.uint8 channels_last bilinear (224, 224) -> (270, 268) aa=True      |   2950.145 (+-18.128)   |    9895.382 (+-35.222)    |    8076.809 (+-23.960)    |    0.816 (+-0.000)
      3 torch.uint8 channels_last bilinear (256, 256) -> (1024, 1024) aa=True    |  27734.895 (+-120.465)  |   99032.332 (+-182.978)   |   87661.256 (+-233.591)   |    0.885 (+-0.000)
      3 torch.uint8 channels_last bilinear (224, 224) -> (64, 64) aa=True        |    958.957 (+-6.354)    |    2231.181 (+-17.082)    |    2241.562 (+-20.083)    |    1.005 (+-0.000)
      3 torch.uint8 channels_last bilinear (270, 268) -> (224, 224) aa=True      |   2992.891 (+-20.158)   |    8683.729 (+-30.855)    |    7221.833 (+-49.896)    |    0.832 (+-0.000)
      3 torch.uint8 channels_last bilinear (256, 256) -> (224, 224) aa=True      |   2823.888 (+-57.261)   |    8486.350 (+-30.461)    |    6993.682 (+-23.502)    |    0.824 (+-0.000)
      3 torch.uint8 channels_last bilinear (1024, 1024) -> (256, 256) aa=True    |  17877.953 (+-100.248)  |   37954.479 (+-181.297)   |   35374.774 (+-216.727)   |    0.932 (+-0.000)
      3 torch.uint8 channels_last bilinear (64, 64) -> (224, 224) aa=False       |                         |    5043.179 (+-17.658)    |    5007.617 (+-18.053)    |    0.993 (+-0.000)
      3 torch.uint8 channels_last bilinear (224, 224) -> (270, 268) aa=False     |                         |    9891.439 (+-31.783)    |    7301.735 (+-21.052)    |    0.738 (+-0.000)
      3 torch.uint8 channels_last bilinear (256, 256) -> (1024, 1024) aa=False   |                         |   98928.769 (+-203.887)   |   96727.184 (+-1851.547)  |    0.978 (+-0.000)
      3 torch.uint8 channels_last bilinear (224, 224) -> (64, 64) aa=False       |                         |     1490.242 (+-5.917)    |     800.316 (+-8.328)     |    0.537 (+-0.000)
      3 torch.uint8 channels_last bilinear (270, 268) -> (224, 224) aa=False     |                         |    8224.615 (+-25.164)    |    5364.108 (+-18.070)    |    0.652 (+-0.000)
      3 torch.uint8 channels_last bilinear (256, 256) -> (224, 224) aa=False     |                         |    8005.172 (+-24.583)    |    5328.746 (+-15.021)    |    0.666 (+-0.000)
      3 torch.uint8 channels_last bilinear (1024, 1024) -> (256, 256) aa=False   |                         |    23815.525 (+-93.504)   |   11805.628 (+-159.437)   |    0.496 (+-0.000)


[---------------------------------------------------------------------------------------- Resize ---------------------------------------------------------------------------------------]
                                                                                 |      Pillow (9.5.0)     |  torch (2.0.0+cpu) PT2.0  |     torchvision resize    |  Speed-up: PTH vs TV
1 threads: ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      3 torch.uint8 channels_first bilinear (64, 64) -> (224, 224) aa=True       |    1441.650 (+-7.793)   |    3133.112 (+-19.848)    |    2658.643 (+-22.010)    |    0.849 (+-0.000)
      3 torch.uint8 channels_first bilinear (224, 224) -> (270, 268) aa=True     |   2945.307 (+-27.857)   |    7234.375 (+-37.813)    |    5048.484 (+-166.233)   |    0.698 (+-0.000)
      3 torch.uint8 channels_first bilinear (256, 256) -> (1024, 1024) aa=True   |  27771.867 (+-244.659)  |   59112.784 (+-380.406)   |   85453.461 (+-191.911)   |    1.446 (+-0.000)
      3 torch.uint8 channels_first bilinear (224, 224) -> (64, 64) aa=True       |    959.119 (+-5.230)    |    1961.959 (+-15.160)    |    1946.299 (+-17.000)    |    0.992 (+-0.000)
      3 torch.uint8 channels_first bilinear (270, 268) -> (224, 224) aa=True     |   2992.920 (+-16.803)   |    6870.023 (+-25.454)    |    5045.447 (+-19.603)    |    0.734 (+-0.000)
      3 torch.uint8 channels_first bilinear (256, 256) -> (224, 224) aa=True     |   2805.420 (+-14.460)   |    6675.696 (+-19.798)    |    4842.727 (+-20.864)    |    0.725 (+-0.000)
      3 torch.uint8 channels_first bilinear (1024, 1024) -> (256, 256) aa=True   |   17809.711 (+-99.375)  |   33419.405 (+-119.643)   |   30904.472 (+-7709.134)  |    0.925 (+-0.000)
      3 torch.uint8 channels_first bilinear (64, 64) -> (224, 224) aa=False      |                         |    3126.211 (+-21.191)    |    6148.521 (+-20.971)    |    1.967 (+-0.000)
      3 torch.uint8 channels_first bilinear (224, 224) -> (270, 268) aa=False    |                         |    7234.122 (+-25.336)    |    9623.514 (+-24.003)    |    1.330 (+-0.000)
      3 torch.uint8 channels_first bilinear (256, 256) -> (1024, 1024) aa=False  |                         |   59102.144 (+-168.034)   |  119818.449 (+-4686.629)  |    2.027 (+-0.000)
      3 torch.uint8 channels_first bilinear (224, 224) -> (64, 64) aa=False      |                         |     1356.032 (+-4.773)    |    1712.139 (+-11.092)    |    1.263 (+-0.000)
      3 torch.uint8 channels_first bilinear (270, 268) -> (224, 224) aa=False    |                         |    6427.439 (+-25.443)    |    7400.255 (+-22.011)    |    1.151 (+-0.000)
      3 torch.uint8 channels_first bilinear (256, 256) -> (224, 224) aa=False    |                         |    6202.273 (+-24.032)    |    7285.175 (+-22.074)    |    1.175 (+-0.000)
      3 torch.uint8 channels_first bilinear (1024, 1024) -> (256, 256) aa=False  |                         |    21569.624 (+-87.971)   |   25735.221 (+-1915.450)  |    1.193 (+-0.000)
      3 torch.uint8 channels_last bilinear (64, 64) -> (224, 224) aa=True        |    1440.204 (+-3.152)   |    5048.634 (+-19.188)    |    4730.987 (+-17.437)    |    0.937 (+-0.000)
      3 torch.uint8 channels_last bilinear (224, 224) -> (270, 268) aa=True      |   2959.496 (+-15.789)   |    9899.122 (+-25.893)    |    8065.141 (+-32.074)    |    0.815 (+-0.000)
      3 torch.uint8 channels_last bilinear (256, 256) -> (1024, 1024) aa=True    |  27739.778 (+-135.324)  |   98951.890 (+-207.316)   |   129421.902 (+-292.965)  |    1.308 (+-0.000)
      3 torch.uint8 channels_last bilinear (224, 224) -> (64, 64) aa=True        |    958.800 (+-2.536)    |    2232.145 (+-14.823)    |    2234.152 (+-20.739)    |    1.001 (+-0.000)
      3 torch.uint8 channels_last bilinear (270, 268) -> (224, 224) aa=True      |   2994.136 (+-17.790)   |    8666.108 (+-29.444)    |    7199.160 (+-21.471)    |    0.831 (+-0.000)
      3 torch.uint8 channels_last bilinear (256, 256) -> (224, 224) aa=True      |   2806.804 (+-16.336)   |    8484.599 (+-24.164)    |    6975.706 (+-24.476)    |    0.822 (+-0.000)
      3 torch.uint8 channels_last bilinear (1024, 1024) -> (256, 256) aa=True    |   17748.965 (+-93.011)  |   37918.864 (+-213.322)   |   35278.220 (+-244.379)   |    0.930 (+-0.000)
      3 torch.uint8 channels_last bilinear (64, 64) -> (224, 224) aa=False       |                         |    5041.425 (+-19.732)    |    5009.649 (+-18.735)    |    0.994 (+-0.000)
      3 torch.uint8 channels_last bilinear (224, 224) -> (270, 268) aa=False     |                         |    9902.715 (+-30.552)    |    7311.017 (+-20.245)    |    0.738 (+-0.000)
      3 torch.uint8 channels_last bilinear (256, 256) -> (1024, 1024) aa=False   |                         |   98949.853 (+-199.764)   |   137463.340 (+-256.288)  |    1.389 (+-0.000)
      3 torch.uint8 channels_last bilinear (224, 224) -> (64, 64) aa=False       |                         |     1488.239 (+-4.232)    |     797.002 (+-5.500)     |    0.536 (+-0.000)
      3 torch.uint8 channels_last bilinear (270, 268) -> (224, 224) aa=False     |                         |    8218.452 (+-25.542)    |    5357.390 (+-19.943)    |    0.652 (+-0.000)
      3 torch.uint8 channels_last bilinear (256, 256) -> (224, 224) aa=False     |                         |    8001.817 (+-23.861)    |    5326.541 (+-17.561)    |    0.666 (+-0.000)
      3 torch.uint8 channels_last bilinear (1024, 1024) -> (256, 256) aa=False   |                         |   23820.776 (+-122.727)   |   11831.602 (+-148.078)   |    0.497 (+-0.000)

Times are in microseconds (us).


[--------------------------------------------------------------------------------------- Resize ---------------------------------------------------------------------------------------]
                                                                                 |      Pillow (9.5.0)     |  torch (2.0.0+cpu) PT2.0  |    torchvision resize    |  Speed-up: PTH vs TV
1 threads: -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      3 torch.uint8 channels_first bilinear (64, 64) -> (224, 224) aa=True       |    1436.075 (+-6.280)   |    3151.910 (+-24.741)    |   4688.737 (+-35.597)    |    1.488 (+-0.000)
      3 torch.uint8 channels_first bilinear (224, 224) -> (270, 268) aa=True     |   2950.526 (+-20.434)   |    7266.164 (+-28.356)    |   7948.703 (+-77.560)    |    1.094 (+-0.000)
      3 torch.uint8 channels_first bilinear (224, 224) -> (64, 64) aa=True       |    957.229 (+-4.152)    |    1973.032 (+-20.967)    |   2130.992 (+-19.875)    |    1.080 (+-0.000)
      3 torch.uint8 channels_first bilinear (270, 268) -> (224, 224) aa=True     |   2994.220 (+-21.532)   |    6894.122 (+-42.373)    |   7057.545 (+-38.806)    |    1.024 (+-0.000)
      3 torch.uint8 channels_first bilinear (256, 256) -> (224, 224) aa=True     |   2803.826 (+-23.697)   |    6688.503 (+-43.002)    |   6851.118 (+-35.586)    |    1.024 (+-0.000)
      3 torch.uint8 channels_first bilinear (1024, 1024) -> (256, 256) aa=True   |  17930.750 (+-158.646)  |   33458.326 (+-204.824)   |  33744.952 (+-5812.943)  |    1.009 (+-0.000)
      3 torch.uint8 channels_first bilinear (64, 64) -> (224, 224) aa=False      |                         |    3138.330 (+-26.213)    |   8178.735 (+-40.076)    |    2.606 (+-0.000)
      3 torch.uint8 channels_first bilinear (224, 224) -> (270, 268) aa=False    |                         |    7254.269 (+-32.447)    |   12503.833 (+-63.279)   |    1.724 (+-0.000)
      3 torch.uint8 channels_first bilinear (224, 224) -> (64, 64) aa=False      |                         |     1365.775 (+-5.829)    |   1878.564 (+-23.301)    |    1.375 (+-0.000)
      3 torch.uint8 channels_first bilinear (270, 268) -> (224, 224) aa=False    |                         |    6440.069 (+-33.179)    |   9432.776 (+-49.253)    |    1.465 (+-0.000)
      3 torch.uint8 channels_first bilinear (256, 256) -> (224, 224) aa=False    |                         |    6217.160 (+-35.983)    |   9305.477 (+-46.104)    |    1.497 (+-0.000)
      3 torch.uint8 channels_first bilinear (1024, 1024) -> (256, 256) aa=False  |                         |   21623.857 (+-145.938)   |  28329.188 (+-1290.288)  |    1.310 (+-0.000)
      3 torch.uint8 channels_last bilinear (64, 64) -> (224, 224) aa=True        |    1437.819 (+-4.783)   |    5271.796 (+-34.490)    |   6791.724 (+-32.413)    |    1.288 (+-0.000)
      3 torch.uint8 channels_last bilinear (224, 224) -> (270, 268) aa=True      |   2952.816 (+-21.517)   |    10210.752 (+-58.947)   |   11016.906 (+-43.166)   |    1.079 (+-0.000)
      3 torch.uint8 channels_last bilinear (224, 224) -> (64, 64) aa=True        |    956.793 (+-4.347)    |    2252.003 (+-21.105)    |   2421.985 (+-20.903)    |    1.075 (+-0.000)
      3 torch.uint8 channels_last bilinear (270, 268) -> (224, 224) aa=True      |   2993.040 (+-25.153)   |    8852.263 (+-32.597)    |   9244.390 (+-44.435)    |    1.044 (+-0.000)
      3 torch.uint8 channels_last bilinear (256, 256) -> (224, 224) aa=True      |   2803.835 (+-26.469)   |    8656.983 (+-38.008)    |   9026.343 (+-43.841)    |    1.043 (+-0.000)
      3 torch.uint8 channels_last bilinear (1024, 1024) -> (256, 256) aa=True    |  17984.565 (+-162.995)  |   37920.598 (+-261.030)   |  37930.408 (+-348.332)   |    1.000 (+-0.000)
      3 torch.uint8 channels_last bilinear (64, 64) -> (224, 224) aa=False       |                         |    5236.299 (+-29.053)    |   7008.583 (+-31.664)    |    1.338 (+-0.000)
      3 torch.uint8 channels_last bilinear (224, 224) -> (270, 268) aa=False     |                         |    10159.084 (+-51.936)   |   10190.819 (+-39.036)   |    1.003 (+-0.000)
      3 torch.uint8 channels_last bilinear (224, 224) -> (64, 64) aa=False       |                         |    1516.840 (+-17.513)    |    967.104 (+-8.784)     |    0.638 (+-0.000)
      3 torch.uint8 channels_last bilinear (270, 268) -> (224, 224) aa=False     |                         |    8409.836 (+-35.272)    |   7376.530 (+-43.325)    |    0.877 (+-0.000)
      3 torch.uint8 channels_last bilinear (256, 256) -> (224, 224) aa=False     |                         |    8203.791 (+-40.240)    |   7347.508 (+-33.443)    |    0.896 (+-0.000)
      3 torch.uint8 channels_last bilinear (1024, 1024) -> (256, 256) aa=False   |                         |   24097.730 (+-222.318)   |  14288.680 (+-132.777)   |    0.593 (+-0.000)

Times are in microseconds (us).
```



- lscpu
```
Architecture:                    x86_64
CPU op-mode(s):                  32-bit, 64-bit
Byte Order:                      Little Endian
Address sizes:                   36 bits physical, 48 bits virtual
CPU(s):                          2
On-line CPU(s) list:             0,1
Thread(s) per core:              1
Core(s) per socket:              2
Socket(s):                       1
NUMA node(s):                    1
Vendor ID:                       GenuineIntel
CPU family:                      6
Model:                           55
Model name:                      Intel(R) Celeron(R) CPU  N2840  @ 2.16GHz
Stepping:                        8
CPU MHz:                         977.421
CPU max MHz:                     2582,3000
CPU min MHz:                     499,8000
BogoMIPS:                        4333.33
Virtualization:                  VT-x
L1d cache:                       48 KiB
L1i cache:                       64 KiB
L2 cache:                        1 MiB
NUMA node0 CPU(s):               0,1
Vulnerability Itlb multihit:     Not affected
Vulnerability L1tf:              Not affected
Vulnerability Mds:               Mitigation; Clear CPU buffers; SMT disabled
Vulnerability Meltdown:          Mitigation; PTI
Vulnerability Spec store bypass: Not affected
Vulnerability Spectre v1:        Mitigation; usercopy/swapgs barriers and __user pointer sanitization
Vulnerability Spectre v2:        Mitigation; Full generic retpoline, IBPB conditional, IBRS_FW, STIBP disabled, RSB filling
Vulnerability Srbds:             Not affected
Vulnerability Tsx async abort:   Not affected
Flags:                           fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx rdtscp lm constant_tsc
                                 arch_perfmon pebs bts rep_good nopl xtopology tsc_reliable nonstop_tsc cpuid aperfmperf tsc_known_freq pni pclmulqdq dtes64 monitor ds_cpl vmx est tm2 sss
                                 e3 cx16 xtpr pdcm sse4_1 sse4_2 movbe popcnt tsc_deadline_timer rdrand lahf_lm 3dnowprefetch epb pti ibrs ibpb stibp tpr_shadow vnmi flexpriority ept vpid
                                  tsc_adjust smep erms dtherm ida arat md_clear
```