Description:
- 20230315-161003-pr
Torch version: 2.1.0a0+git0968a5d
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
                                                                 |  Pillow (9.0.0.post1)  |  torch (2.1.0a0+git0968a5d) PR  |  torch (2.1.0a0+git5309c44) nightly  |  Speed-up: PR vs nightly
1 threads: ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      3 torch.uint8 channels_last bilinear 256 -> 32 aa=True     |          39.0          |                56.6             |                 133.2                |            2.4
      3 torch.uint8 channels_last bilinear 256 -> 32 aa=False    |                        |                36.9             |                 112.8                |            3.1
      3 torch.uint8 channels_last bilinear 256 -> 224 aa=True    |         128.1          |               152.5             |                 305.4                |            2.0
      3 torch.uint8 channels_last bilinear 256 -> 224 aa=False   |                        |               141.1             |                 288.7                |            2.0
      3 torch.uint8 channels_last bilinear 256 -> 320 aa=True    |         179.6          |               208.8             |                 442.5                |            2.1
      3 torch.uint8 channels_last bilinear 256 -> 320 aa=False   |                        |               206.4             |                 436.9                |            2.1
      3 torch.uint8 channels_last bilinear 520 -> 32 aa=True     |         113.3          |               132.1             |                 464.8                |            3.5
      3 torch.uint8 channels_last bilinear 520 -> 32 aa=False    |                        |                57.2             |                 365.5                |            6.4
      3 torch.uint8 channels_last bilinear 520 -> 224 aa=True    |         281.7          |               327.4             |                 722.4                |            2.2
      3 torch.uint8 channels_last bilinear 520 -> 224 aa=False   |                        |               230.2             |                 593.5                |            2.6
      3 torch.uint8 channels_last bilinear 712 -> 32 aa=True     |         186.9          |               210.5             |                 833.8                |            4.0
      3 torch.uint8 channels_last bilinear 712 -> 32 aa=False    |                        |                75.6             |                 651.4                |            8.6
      3 torch.uint8 channels_last bilinear 712 -> 224 aa=True    |         410.3          |               450.9             |                1128.4                |            2.5
      3 torch.uint8 channels_last bilinear 712 -> 224 aa=False   |                        |               298.7             |                 917.6                |            3.1
      4 torch.uint8 channels_last bilinear 256 -> 32 aa=True     |                        |                51.9             |                  54.7                |            1.1
      4 torch.uint8 channels_last bilinear 256 -> 32 aa=False    |                        |                35.3             |                  34.3                |            1.0
      4 torch.uint8 channels_last bilinear 256 -> 224 aa=True    |                        |               144.4             |                 139.7                |            1.0
      4 torch.uint8 channels_last bilinear 256 -> 224 aa=False   |                        |               131.1             |                 125.1                |            1.0
      4 torch.uint8 channels_last bilinear 256 -> 320 aa=True    |                        |               195.8             |                 185.1                |            0.9
      4 torch.uint8 channels_last bilinear 256 -> 320 aa=False   |                        |               193.1             |                 183.1                |            0.9
      4 torch.uint8 channels_last bilinear 520 -> 32 aa=True     |                        |               124.4             |                 135.0                |            1.1
      4 torch.uint8 channels_last bilinear 520 -> 32 aa=False    |                        |                56.6             |                  54.7                |            1.0
      4 torch.uint8 channels_last bilinear 520 -> 224 aa=True    |                        |               285.2             |                 325.7                |            1.1
      4 torch.uint8 channels_last bilinear 520 -> 224 aa=False   |                        |               213.0             |                 199.9                |            0.9
      4 torch.uint8 channels_last bilinear 712 -> 32 aa=True     |                        |               192.4             |                 211.0                |            1.1
      4 torch.uint8 channels_last bilinear 712 -> 32 aa=False    |                        |                73.9             |                  71.6                |            1.0
      4 torch.uint8 channels_last bilinear 712 -> 224 aa=True    |                        |               423.0             |                 464.0                |            1.1
      4 torch.uint8 channels_last bilinear 712 -> 224 aa=False   |                        |               273.5             |                 254.5                |            0.9
      3 torch.uint8 channels_first bilinear 256 -> 32 aa=True    |          38.9          |               129.3             |                 132.8                |            1.0
      3 torch.uint8 channels_first bilinear 256 -> 32 aa=False   |                        |               112.6             |                 112.5                |            1.0
      3 torch.uint8 channels_first bilinear 256 -> 224 aa=True   |         128.1          |               295.9             |                 330.7                |            1.1
      3 torch.uint8 channels_first bilinear 256 -> 224 aa=False  |                        |               283.8             |                 312.5                |            1.1
      3 torch.uint8 channels_first bilinear 256 -> 320 aa=True   |         179.0          |               428.8             |                 473.0                |            1.1
      3 torch.uint8 channels_first bilinear 256 -> 320 aa=False  |                        |               425.8             |                 494.7                |            1.2
      3 torch.uint8 channels_first bilinear 520 -> 32 aa=True    |         113.4          |               434.3             |                 446.0                |            1.0
      3 torch.uint8 channels_first bilinear 520 -> 32 aa=False   |                        |               366.3             |                 365.3                |            1.0
      3 torch.uint8 channels_first bilinear 520 -> 224 aa=True   |         281.4          |               669.4             |                 748.5                |            1.1
      3 torch.uint8 channels_first bilinear 520 -> 224 aa=False  |                        |               598.7             |                 614.0                |            1.0
      3 torch.uint8 channels_first bilinear 712 -> 32 aa=True    |         186.4          |               780.5             |                 804.5                |            1.0
      3 torch.uint8 channels_first bilinear 712 -> 32 aa=False   |                        |               653.3             |                 788.4                |            1.2
      3 torch.uint8 channels_first bilinear 712 -> 224 aa=True   |         410.9          |              1246.4             |                1155.5                |            0.9
      3 torch.uint8 channels_first bilinear 712 -> 224 aa=False  |                        |               926.1             |                 943.9                |            1.0
      4 torch.uint8 channels_first bilinear 256 -> 32 aa=True    |                        |                70.7             |                  74.4                |            1.1
      4 torch.uint8 channels_first bilinear 256 -> 32 aa=False   |                        |                53.8             |                  53.3                |            1.0
      4 torch.uint8 channels_first bilinear 256 -> 224 aa=True   |                        |               251.1             |                 258.3                |            1.0
      4 torch.uint8 channels_first bilinear 256 -> 224 aa=False  |                        |               238.2             |                 250.2                |            1.1
      4 torch.uint8 channels_first bilinear 256 -> 320 aa=True   |                        |               398.1             |                 418.8                |            1.1
      4 torch.uint8 channels_first bilinear 256 -> 320 aa=False  |                        |               394.8             |                 421.2                |            1.1
      4 torch.uint8 channels_first bilinear 520 -> 32 aa=True    |                        |               193.8             |                 203.9                |            1.1
      4 torch.uint8 channels_first bilinear 520 -> 32 aa=False   |                        |               124.8             |                 123.1                |            1.0
      4 torch.uint8 channels_first bilinear 520 -> 224 aa=True   |                        |               441.9             |                 498.6                |            1.1
      4 torch.uint8 channels_first bilinear 520 -> 224 aa=False  |                        |               369.7             |                 367.8                |            1.0
      4 torch.uint8 channels_first bilinear 712 -> 32 aa=True    |                        |               321.9             |                 384.8                |            1.2
      4 torch.uint8 channels_first bilinear 712 -> 32 aa=False   |                        |               200.8             |                 198.7                |            1.0
      4 torch.uint8 channels_first bilinear 712 -> 224 aa=True   |                        |               686.7             |                 905.4                |            1.3
      4 torch.uint8 channels_first bilinear 712 -> 224 aa=False  |                        |               771.7             |                 690.9                |            0.9

Times are in microseconds (us).