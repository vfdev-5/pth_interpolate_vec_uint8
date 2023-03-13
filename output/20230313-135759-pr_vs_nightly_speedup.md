Description:
- 20230313-133243-pr
Torch version: 2.1.0a0+git1d3a939
Torch config: PyTorch built with:
  - GCC 9.4
  - C++ Version: 201703
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - CPU capability usage: AVX2
  - Build settings: BUILD_TYPE=Release, CXX_COMPILER=/usr/bin/c++, CXX_FLAGS= -D_GLIBCXX_USE_CXX11_ABI=1 -Wno-deprecated -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -DNDEBUG -DUSE_KINETO -DLIBKINETO_NOCUPTI -DLIBKINETO_NOROCTRACER -DUSE_PYTORCH_QNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -O2 -fPIC -Wall -Wextra -Werror=return-type -Werror=non-virtual-dtor -Werror=bool-operation -Wnarrowing -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-unused-parameter -Wno-unused-function -Wno-unused-result -Wno-strict-overflow -Wno-strict-aliasing -Wno-error=deprecated-declarations -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=old-style-cast -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Werror=cast-function-type -Wno-stringop-overflow, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, TORCH_DISABLE_GPU_ASSERTS=ON, TORCH_VERSION=2.1.0, USE_CUDA=0, USE_CUDNN=OFF, USE_EIGEN_FOR_BLAS=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=OFF, USE_MKLDNN=0, USE_MPI=OFF, USE_NCCL=OFF, USE_NNPACK=0, USE_OPENMP=ON, USE_ROCM=OFF,


- 20230313-134520-nightly
Torch version: 2.1.0a0+git5309c44
Torch config: PyTorch built with:
  - GCC 9.4
  - C++ Version: 201703
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - CPU capability usage: AVX2
  - Build settings: BUILD_TYPE=Release, CXX_COMPILER=/usr/bin/c++, CXX_FLAGS= -D_GLIBCXX_USE_CXX11_ABI=1 -Wno-deprecated -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -DNDEBUG -DUSE_KINETO -DLIBKINETO_NOCUPTI -DLIBKINETO_NOROCTRACER -DUSE_PYTORCH_QNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -O2 -fPIC -Wall -Wextra -Werror=return-type -Werror=non-virtual-dtor -Werror=bool-operation -Wnarrowing -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-unused-parameter -Wno-unused-function -Wno-unused-result -Wno-strict-overflow -Wno-strict-aliasing -Wno-error=deprecated-declarations -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=old-style-cast -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Werror=cast-function-type -Wno-stringop-overflow, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, TORCH_DISABLE_GPU_ASSERTS=ON, TORCH_VERSION=2.1.0, USE_CUDA=0, USE_CUDNN=OFF, USE_EIGEN_FOR_BLAS=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=OFF, USE_MKLDNN=0, USE_MPI=OFF, USE_NCCL=OFF, USE_NNPACK=0, USE_OPENMP=ON, USE_ROCM=OFF,



[------------------------------------------------------------------------------------------ Resize -----------------------------------------------------------------------------------------]
                                                                 |  Pillow (9.0.0.post1)  |  torch (2.1.0a0+git1d3a939) PR  |  torch (2.1.0a0+git5309c44) nightly  |  Speed-up: PR vs nightly
1 threads: ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      3 torch.uint8 channels_last bilinear 256 -> 32 aa=True     |          38.5          |                56.3             |                 132.5                |            2.4
      3 torch.uint8 channels_last bilinear 256 -> 32 aa=False    |                        |                36.2             |                 110.6                |            3.1
      3 torch.uint8 channels_last bilinear 256 -> 224 aa=True    |         127.0          |               149.9             |                 292.2                |            1.9
      3 torch.uint8 channels_last bilinear 256 -> 224 aa=False   |                        |               134.2             |                 276.8                |            2.1
      3 torch.uint8 channels_last bilinear 256 -> 320 aa=True    |         178.1          |               200.3             |                 416.4                |            2.1
      3 torch.uint8 channels_last bilinear 256 -> 320 aa=False   |                        |               198.0             |                 414.4                |            2.1
      3 torch.uint8 channels_last bilinear 520 -> 32 aa=True     |         112.9          |               129.3             |                 441.3                |            3.4
      3 torch.uint8 channels_last bilinear 520 -> 32 aa=False    |                        |                54.9             |                 364.2                |            6.6
      3 torch.uint8 channels_last bilinear 520 -> 224 aa=True    |         282.7          |               324.8             |                 691.6                |            2.1
      3 torch.uint8 channels_last bilinear 520 -> 224 aa=False   |                        |               211.9             |                 583.1                |            2.8
      3 torch.uint8 channels_last bilinear 712 -> 32 aa=True     |         185.9          |               201.1             |                 783.1                |            3.9
      3 torch.uint8 channels_last bilinear 712 -> 32 aa=False    |                        |                72.1             |                 649.8                |            9.0
      3 torch.uint8 channels_last bilinear 712 -> 224 aa=True    |         408.7          |               436.7             |                1100.5                |            2.5
      3 torch.uint8 channels_last bilinear 712 -> 224 aa=False   |                        |               268.8             |                 906.6                |            3.4
      4 torch.uint8 channels_last bilinear 256 -> 32 aa=True     |                        |                53.5             |                  54.4                |            1.0
      4 torch.uint8 channels_last bilinear 256 -> 32 aa=False    |                        |                34.8             |                  33.5                |            1.0
      4 torch.uint8 channels_last bilinear 256 -> 224 aa=True    |                        |               142.8             |                 140.2                |            1.0
      4 torch.uint8 channels_last bilinear 256 -> 224 aa=False   |                        |               127.5             |                 125.6                |            1.0
      4 torch.uint8 channels_last bilinear 256 -> 320 aa=True    |                        |               190.8             |                 185.2                |            1.0
      4 torch.uint8 channels_last bilinear 256 -> 320 aa=False   |                        |               188.9             |                 183.2                |            1.0
      4 torch.uint8 channels_last bilinear 520 -> 32 aa=True     |                        |               124.6             |                 132.6                |            1.1
      4 torch.uint8 channels_last bilinear 520 -> 32 aa=False    |                        |                55.9             |                  54.6                |            1.0
      4 torch.uint8 channels_last bilinear 520 -> 224 aa=True    |                        |               289.7             |                 307.9                |            1.1
      4 torch.uint8 channels_last bilinear 520 -> 224 aa=False   |                        |               200.1             |                 200.0                |            1.0
      4 torch.uint8 channels_last bilinear 712 -> 32 aa=True     |                        |               193.8             |                 207.9                |            1.1
      4 torch.uint8 channels_last bilinear 712 -> 32 aa=False    |                        |                72.9             |                  71.5                |            1.0
      4 torch.uint8 channels_last bilinear 712 -> 224 aa=True    |                        |               422.5             |                 451.9                |            1.1
      4 torch.uint8 channels_last bilinear 712 -> 224 aa=False   |                        |               253.3             |                 254.8                |            1.0
      3 torch.uint8 channels_first bilinear 256 -> 32 aa=True    |          38.4          |               131.0             |                 132.6                |            1.0
      3 torch.uint8 channels_first bilinear 256 -> 32 aa=False   |                        |               112.6             |                 111.0                |            1.0
      3 torch.uint8 channels_first bilinear 256 -> 224 aa=True   |         127.8          |               295.3             |                 317.4                |            1.1
      3 torch.uint8 channels_first bilinear 256 -> 224 aa=False  |                        |               280.2             |                 310.7                |            1.1
      3 torch.uint8 channels_first bilinear 256 -> 320 aa=True   |         178.6          |               422.9             |                 461.1                |            1.1
      3 torch.uint8 channels_first bilinear 256 -> 320 aa=False  |                        |               420.1             |                 474.3                |            1.1
      3 torch.uint8 channels_first bilinear 520 -> 32 aa=True    |         113.3          |               433.2             |                 442.1                |            1.0
      3 torch.uint8 channels_first bilinear 520 -> 32 aa=False   |                        |               364.7             |                 364.0                |            1.0
      3 torch.uint8 channels_first bilinear 520 -> 224 aa=True   |         282.5          |               674.1             |                 724.1                |            1.1
      3 torch.uint8 channels_first bilinear 520 -> 224 aa=False  |                        |               584.8             |                 619.1                |            1.1
      3 torch.uint8 channels_first bilinear 712 -> 32 aa=True    |         185.7          |               771.0             |                 787.8                |            1.0
      3 torch.uint8 channels_first bilinear 712 -> 32 aa=False   |                        |               652.6             |                 647.9                |            1.0
      3 torch.uint8 channels_first bilinear 712 -> 224 aa=True   |         408.3          |              1076.5             |                1132.7                |            1.1
      3 torch.uint8 channels_first bilinear 712 -> 224 aa=False  |                        |               905.9             |                 937.3                |            1.0
      4 torch.uint8 channels_first bilinear 256 -> 32 aa=True    |                        |                72.9             |                  74.0                |            1.0
      4 torch.uint8 channels_first bilinear 256 -> 32 aa=False   |                        |                54.1             |                  53.1                |            1.0
      4 torch.uint8 channels_first bilinear 256 -> 224 aa=True   |                        |               250.8             |                 276.4                |            1.1
      4 torch.uint8 channels_first bilinear 256 -> 224 aa=False  |                        |               235.7             |                 266.3                |            1.1
      4 torch.uint8 channels_first bilinear 256 -> 320 aa=True   |                        |               393.3             |                 451.1                |            1.1
      4 torch.uint8 channels_first bilinear 256 -> 320 aa=False  |                        |               390.0             |                 441.4                |            1.1
      4 torch.uint8 channels_first bilinear 520 -> 32 aa=True    |                        |               193.0             |                 201.1                |            1.0
      4 torch.uint8 channels_first bilinear 520 -> 32 aa=False   |                        |               124.0             |                 123.2                |            1.0
      4 torch.uint8 channels_first bilinear 520 -> 224 aa=True   |                        |               446.3             |                 481.7                |            1.1
      4 torch.uint8 channels_first bilinear 520 -> 224 aa=False  |                        |               356.2             |                 392.5                |            1.1
      4 torch.uint8 channels_first bilinear 712 -> 32 aa=True    |                        |               321.5             |                 333.7                |            1.0
      4 torch.uint8 channels_first bilinear 712 -> 32 aa=False   |                        |               198.8             |                 197.6                |            1.0
      4 torch.uint8 channels_first bilinear 712 -> 224 aa=True   |                        |               637.6             |                 692.5                |            1.1
      4 torch.uint8 channels_first bilinear 712 -> 224 aa=False  |                        |               466.5             |                 499.1                |            1.1

Times are in microseconds (us).
