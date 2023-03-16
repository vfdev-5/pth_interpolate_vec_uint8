Description:
- 20230315-135753-pr
Torch version: 2.1.0a0+gitcc42a3f
Torch config: PyTorch built with:
  - GCC 9.4
  - C++ Version: 201703
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - CPU capability usage: AVX2
  - Build settings: BUILD_TYPE=Release, CXX_COMPILER=/usr/bin/c++, CXX_FLAGS= -D_GLIBCXX_USE_CXX11_ABI=1 -Wno-deprecated -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -DNDEBUG -DUSE_KINETO -DLIBKINETO_NOCUPTI -DLIBKINETO_NOROCTRACER -DUSE_PYTORCH_QNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -O2 -fPIC -Wall -Wextra -Werror=return-type -Werror=non-virtual-dtor -Werror=bool-operation -Wnarrowing -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-unused-parameter -Wno-unused-function -Wno-unused-result -Wno-strict-overflow -Wno-strict-aliasing -Wno-error=deprecated-declarations -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=old-style-cast -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Werror=cast-function-type -Wno-stringop-overflow, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, TORCH_DISABLE_GPU_ASSERTS=ON, TORCH_VERSION=2.1.0, USE_CUDA=0, USE_CUDNN=OFF, USE_EIGEN_FOR_BLAS=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=OFF, USE_MKLDNN=0, USE_MPI=OFF, USE_NCCL=OFF, USE_NNPACK=0, USE_OPENMP=ON, USE_ROCM=OFF,


- 20230315-134541-nightly
Torch version: 2.1.0a0+git5309c44
Torch config: PyTorch built with:
  - GCC 9.4
  - C++ Version: 201703
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - CPU capability usage: AVX2
  - Build settings: BUILD_TYPE=Release, CXX_COMPILER=/usr/bin/c++, CXX_FLAGS= -D_GLIBCXX_USE_CXX11_ABI=1 -Wno-deprecated -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -DNDEBUG -DUSE_KINETO -DLIBKINETO_NOCUPTI -DLIBKINETO_NOROCTRACER -DUSE_PYTORCH_QNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -O2 -fPIC -Wall -Wextra -Werror=return-type -Werror=non-virtual-dtor -Werror=bool-operation -Wnarrowing -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-unused-parameter -Wno-unused-function -Wno-unused-result -Wno-strict-overflow -Wno-strict-aliasing -Wno-error=deprecated-declarations -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=old-style-cast -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Werror=cast-function-type -Wno-stringop-overflow, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, TORCH_DISABLE_GPU_ASSERTS=ON, TORCH_VERSION=2.1.0, USE_CUDA=0, USE_CUDNN=OFF, USE_EIGEN_FOR_BLAS=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=OFF, USE_MKLDNN=0, USE_MPI=OFF, USE_NCCL=OFF, USE_NNPACK=0, USE_OPENMP=ON, USE_ROCM=OFF,



[------------------------------------------------------------------------------------------ Resize -----------------------------------------------------------------------------------------]
                                                                 |  Pillow (9.0.0.post1)  |  torch (2.1.0a0+gitcc42a3f) PR  |  torch (2.1.0a0+git5309c44) nightly  |  Speed-up: PR vs nightly
1 threads: ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      3 torch.uint8 channels_last bilinear 256 -> 32 aa=True     |          38.8          |                56.0             |                 133.2                |            2.4
      3 torch.uint8 channels_last bilinear 256 -> 32 aa=False    |                        |                37.5             |                 112.8                |            3.0
      3 torch.uint8 channels_last bilinear 256 -> 224 aa=True    |         128.7          |               157.0             |                 305.4                |            1.9
      3 torch.uint8 channels_last bilinear 256 -> 224 aa=False   |                        |               146.4             |                 288.7                |            2.0
      3 torch.uint8 channels_last bilinear 256 -> 320 aa=True    |         179.4          |               215.8             |                 442.5                |            2.1
      3 torch.uint8 channels_last bilinear 256 -> 320 aa=False   |                        |               212.5             |                 436.9                |            2.1
      3 torch.uint8 channels_last bilinear 520 -> 32 aa=True     |         113.3          |               127.9             |                 464.8                |            3.6
      3 torch.uint8 channels_last bilinear 520 -> 32 aa=False    |                        |                56.8             |                 365.5                |            6.4
      3 torch.uint8 channels_last bilinear 520 -> 224 aa=True    |         281.7          |               325.2             |                 722.4                |            2.2
      3 torch.uint8 channels_last bilinear 520 -> 224 aa=False   |                        |               239.1             |                 593.5                |            2.5
      3 torch.uint8 channels_last bilinear 712 -> 32 aa=True     |         186.2          |               200.7             |                 833.8                |            4.2
      3 torch.uint8 channels_last bilinear 712 -> 32 aa=False    |                        |                75.2             |                 651.4                |            8.7
      3 torch.uint8 channels_last bilinear 712 -> 224 aa=True    |         410.0          |               444.5             |                1128.4                |            2.5
      3 torch.uint8 channels_last bilinear 712 -> 224 aa=False   |                        |               309.3             |                 917.6                |            3.0
      4 torch.uint8 channels_last bilinear 256 -> 32 aa=True     |                        |                52.4             |                  54.7                |            1.0
      4 torch.uint8 channels_last bilinear 256 -> 32 aa=False    |                        |                35.0             |                  34.3                |            1.0
      4 torch.uint8 channels_last bilinear 256 -> 224 aa=True    |                        |               147.1             |                 139.7                |            0.9
      4 torch.uint8 channels_last bilinear 256 -> 224 aa=False   |                        |               133.7             |                 125.1                |            0.9
      4 torch.uint8 channels_last bilinear 256 -> 320 aa=True    |                        |               200.1             |                 185.1                |            0.9
      4 torch.uint8 channels_last bilinear 256 -> 320 aa=False   |                        |               197.6             |                 183.1                |            0.9
      4 torch.uint8 channels_last bilinear 520 -> 32 aa=True     |                        |               124.5             |                 135.0                |            1.1
      4 torch.uint8 channels_last bilinear 520 -> 32 aa=False    |                        |                56.4             |                  54.7                |            1.0
      4 torch.uint8 channels_last bilinear 520 -> 224 aa=True    |                        |               289.1             |                 325.7                |            1.1
      4 torch.uint8 channels_last bilinear 520 -> 224 aa=False   |                        |               216.1             |                 199.9                |            0.9
      4 torch.uint8 channels_last bilinear 712 -> 32 aa=True     |                        |               194.6             |                 211.0                |            1.1
      4 torch.uint8 channels_last bilinear 712 -> 32 aa=False    |                        |                74.0             |                  71.6                |            1.0
      4 torch.uint8 channels_last bilinear 712 -> 224 aa=True    |                        |               426.1             |                 464.0                |            1.1
      4 torch.uint8 channels_last bilinear 712 -> 224 aa=False   |                        |               276.2             |                 254.5                |            0.9
      3 torch.uint8 channels_first bilinear 256 -> 32 aa=True    |          39.0          |               129.9             |                 132.8                |            1.0
      3 torch.uint8 channels_first bilinear 256 -> 32 aa=False   |                        |               112.7             |                 112.5                |            1.0
      3 torch.uint8 channels_first bilinear 256 -> 224 aa=True   |         128.6          |               300.4             |                 330.7                |            1.1
      3 torch.uint8 channels_first bilinear 256 -> 224 aa=False  |                        |               288.3             |                 312.5                |            1.1
      3 torch.uint8 channels_first bilinear 256 -> 320 aa=True   |         179.6          |               436.5             |                 473.0                |            1.1
      3 torch.uint8 channels_first bilinear 256 -> 320 aa=False  |                        |               432.7             |                 494.7                |            1.1
      3 torch.uint8 channels_first bilinear 520 -> 32 aa=True    |         113.5          |               437.0             |                 446.0                |            1.0
      3 torch.uint8 channels_first bilinear 520 -> 32 aa=False   |                        |               367.5             |                 365.3                |            1.0
      3 torch.uint8 channels_first bilinear 520 -> 224 aa=True   |         281.9          |               676.2             |                 748.5                |            1.1
      3 torch.uint8 channels_first bilinear 520 -> 224 aa=False  |                        |               603.4             |                 614.0                |            1.0
      3 torch.uint8 channels_first bilinear 712 -> 32 aa=True    |         186.4          |               774.5             |                 804.5                |            1.0
      3 torch.uint8 channels_first bilinear 712 -> 32 aa=False   |                        |               803.3             |                 788.4                |            1.0
      3 torch.uint8 channels_first bilinear 712 -> 224 aa=True   |         412.2          |              1089.3             |                1155.5                |            1.1
      3 torch.uint8 channels_first bilinear 712 -> 224 aa=False  |                        |               934.9             |                 943.9                |            1.0
      4 torch.uint8 channels_first bilinear 256 -> 32 aa=True    |                        |                71.7             |                  74.4                |            1.0
      4 torch.uint8 channels_first bilinear 256 -> 32 aa=False   |                        |                54.3             |                  53.3                |            1.0
      4 torch.uint8 channels_first bilinear 256 -> 224 aa=True   |                        |               285.9             |                 258.3                |            0.9
      4 torch.uint8 channels_first bilinear 256 -> 224 aa=False  |                        |               273.2             |                 250.2                |            0.9
      4 torch.uint8 channels_first bilinear 256 -> 320 aa=True   |                        |               459.3             |                 418.8                |            0.9
      4 torch.uint8 channels_first bilinear 256 -> 320 aa=False  |                        |               458.8             |                 421.2                |            0.9
      4 torch.uint8 channels_first bilinear 520 -> 32 aa=True    |                        |               193.0             |                 203.9                |            1.1
      4 torch.uint8 channels_first bilinear 520 -> 32 aa=False   |                        |               125.7             |                 123.1                |            1.0
      4 torch.uint8 channels_first bilinear 520 -> 224 aa=True   |                        |               477.6             |                 498.6                |            1.0
      4 torch.uint8 channels_first bilinear 520 -> 224 aa=False  |                        |               404.5             |                 367.8                |            0.9
      4 torch.uint8 channels_first bilinear 712 -> 32 aa=True    |                        |               473.1             |                 384.8                |            0.8
      4 torch.uint8 channels_first bilinear 712 -> 32 aa=False   |                        |               201.0             |                 198.7                |            1.0
      4 torch.uint8 channels_first bilinear 712 -> 224 aa=True   |                        |               944.9             |                 905.4                |            1.0
      4 torch.uint8 channels_first bilinear 712 -> 224 aa=False  |                        |               534.1             |                 690.9                |            1.3

Times are in microseconds (us).