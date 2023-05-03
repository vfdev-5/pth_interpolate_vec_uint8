Description:
- 20230503-162051-pr
Torch version: 2.1.0a0+git5e1bf10
Torch config: PyTorch built with:
  - GCC 9.4
  - C++ Version: 201703
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - CPU capability usage: AVX2
  - CUDA Runtime 11.7
  - NVCC architecture flags: -gencode;arch=compute_61,code=sm_61
  - CuDNN 8.5
  - Build settings: BUILD_TYPE=Release, CUDA_VERSION=11.7, CUDNN_VERSION=8.5.0, CXX_COMPILER=/usr/bin/c++, CXX_FLAGS= -D_GLIBCXX_USE_CXX11_ABI=1 -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -DNDEBUG -DUSE_KINETO -DLIBKINETO_NOROCTRACER -DUSE_PYTORCH_QNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -O2 -fPIC -Wall -Wextra -Werror=return-type -Werror=non-virtual-dtor -Werror=bool-operation -Wnarrowing -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-unused-parameter -Wno-unused-function -Wno-unused-result -Wno-strict-overflow -Wno-strict-aliasing -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=old-style-cast -Wno-invalid-partial-specialization -Wno-unused-private-field -Wno-aligned-allocation-unavailable -Wno-missing-braces -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Werror=cast-function-type -Wno-stringop-overflow, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, TORCH_DISABLE_GPU_ASSERTS=ON, TORCH_VERSION=2.1.0, USE_CUDA=1, USE_CUDNN=1, USE_EIGEN_FOR_BLAS=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=OFF, USE_MKLDNN=0, USE_MPI=OFF, USE_NCCL=ON, USE_NNPACK=0, USE_OPENMP=ON, USE_ROCM=OFF, 


- 20230503-164105-nightly
Torch version: 2.1.0a0+git2b75955
Torch config: PyTorch built with:
  - GCC 9.4
  - C++ Version: 201703
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - CPU capability usage: AVX2
  - Build settings: BUILD_TYPE=Release, CXX_COMPILER=/usr/bin/c++, CXX_FLAGS= -D_GLIBCXX_USE_CXX11_ABI=1 -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -DNDEBUG -DUSE_KINETO -DLIBKINETO_NOCUPTI -DLIBKINETO_NOROCTRACER -DUSE_PYTORCH_QNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -O2 -fPIC -Wall -Wextra -Werror=return-type -Werror=non-virtual-dtor -Werror=bool-operation -Wnarrowing -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-unused-parameter -Wno-unused-function -Wno-unused-result -Wno-strict-overflow -Wno-strict-aliasing -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=old-style-cast -Wno-invalid-partial-specialization -Wno-unused-private-field -Wno-aligned-allocation-unavailable -Wno-missing-braces -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Werror=cast-function-type -Wno-stringop-overflow, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, TORCH_DISABLE_GPU_ASSERTS=ON, TORCH_VERSION=2.1.0, USE_CUDA=0, USE_CUDNN=OFF, USE_EIGEN_FOR_BLAS=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=OFF, USE_MKLDNN=0, USE_MPI=OFF, USE_NCCL=OFF, USE_NNPACK=0, USE_OPENMP=ON, USE_ROCM=OFF, 



[-------------------------------------------------------------------------------------------- Resize --------------------------------------------------------------------------------------------]
                                                                                                   |  torch (2.1.0a0+git5e1bf10) PR  |  torch (2.1.0a0+git2b75955) nightly  |  Ratio PR vs Nightly
1 threads: ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      3 torch.uint8 channels_last(squeeze/unsqueeze) bilinear (256, 256) -> (32, 32) aa=False      |         40.543 (+-0.127)        |           38.566 (+-0.334)           |    0.951 (+-0.000)  
      3 torch.uint8 channels_last(squeeze/unsqueeze) bilinear (256, 256) -> (224, 224) aa=True     |        237.936 (+-0.653)        |          153.477 (+-2.497)           |    0.645 (+-0.000)  
      3 torch.uint8 channels_last(squeeze/unsqueeze) bilinear (256, 256) -> (320, 320) aa=True     |        377.910 (+-0.882)        |          210.050 (+-1.204)           |    0.556 (+-0.000)  
      3 torch.uint8 channels_last(squeeze/unsqueeze) bilinear (256, 256) -> (320, 320) aa=False    |        375.758 (+-1.004)        |          208.156 (+-2.144)           |    0.554 (+-0.000)  
      3 torch.uint8 channels_last(squeeze/unsqueeze) bilinear (520, 520) -> (32, 32) aa=False      |         61.544 (+-0.123)        |           58.835 (+-0.292)           |    0.956 (+-0.000)  
      3 torch.uint8 channels_last(squeeze/unsqueeze) bilinear (520, 520) -> (224, 224) aa=True     |        419.544 (+-0.553)        |          320.355 (+-2.999)           |    0.764 (+-0.000)  
      3 torch.uint8 channels_last(squeeze/unsqueeze) bilinear (712, 712) -> (32, 32) aa=False      |         80.612 (+-0.418)        |           77.484 (+-0.324)           |    0.961 (+-0.000)  
      3 torch.uint8 channels_last(squeeze/unsqueeze) bilinear (712, 712) -> (224, 224) aa=True     |        551.300 (+-0.903)        |          443.435 (+-5.117)           |    0.804 (+-0.000)  
      3 torch.uint8 channels_last(squeeze/unsqueeze) bilinear (64, 64) -> (224, 224) aa=True       |        156.963 (+-0.708)        |           78.598 (+-0.416)           |    0.501 (+-0.000)  
      3 torch.uint8 channels_last(squeeze/unsqueeze) bilinear (224, 224) -> (270, 268) aa=True     |        278.948 (+-0.426)        |          161.866 (+-1.599)           |    0.580 (+-0.000)  
      3 torch.uint8 channels_last(squeeze/unsqueeze) bilinear (224, 224) -> (64, 64) aa=True       |         79.192 (+-0.302)        |           71.203 (+-0.273)           |    0.899 (+-0.000)  
      3 torch.uint8 channels_last(squeeze/unsqueeze) bilinear (270, 268) -> (224, 224) aa=True     |        250.236 (+-0.983)        |          165.684 (+-1.667)           |    0.662 (+-0.000)  
      3 torch.uint8 channels_last(squeeze/unsqueeze) bilinear (1024, 1024) -> (256, 256) aa=True   |        894.841 (+-4.009)        |          740.653 (+-3.795)           |    0.828 (+-0.000)  
      3 torch.uint8 channels_last(squeeze/unsqueeze) bilinear (64, 64) -> (224, 224) aa=False      |        155.342 (+-3.313)        |           77.185 (+-1.808)           |    0.497 (+-0.000)  
      3 torch.uint8 channels_last(squeeze/unsqueeze) bilinear (224, 224) -> (270, 268) aa=False    |        277.344 (+-0.598)        |          159.713 (+-1.382)           |    0.576 (+-0.000)  
      3 torch.uint8 channels_last(squeeze/unsqueeze) bilinear (224, 224) -> (64, 64) aa=False      |         61.039 (+-0.096)        |           53.302 (+-0.355)           |    0.873 (+-0.000)  
      3 torch.uint8 channels_last(squeeze/unsqueeze) bilinear (270, 268) -> (224, 224) aa=False    |        233.955 (+-0.814)        |          148.587 (+-1.196)           |    0.635 (+-0.000)  
      3 torch.uint8 channels_last(squeeze/unsqueeze) bilinear (256, 256) -> (224, 224) aa=False    |        226.629 (+-0.563)        |          142.197 (+-0.906)           |    0.627 (+-0.000)  
      3 torch.uint8 channels_last(squeeze/unsqueeze) bilinear (1024, 1024) -> (256, 256) aa=False  |        640.012 (+-2.759)        |          469.483 (+-1.886)           |    0.734 (+-0.000)  
      4 torch.uint8 channels_last(squeeze/unsqueeze) bilinear (256, 256) -> (32, 32) aa=False      |         36.372 (+-0.086)        |           36.333 (+-0.259)           |    0.999 (+-0.000)  
      4 torch.uint8 channels_last(squeeze/unsqueeze) bilinear (256, 256) -> (224, 224) aa=True     |        233.769 (+-0.758)        |          145.390 (+-1.055)           |    0.622 (+-0.000)  
      4 torch.uint8 channels_last(squeeze/unsqueeze) bilinear (256, 256) -> (320, 320) aa=True     |        383.813 (+-1.395)        |          195.840 (+-1.971)           |    0.510 (+-0.000)  
      4 torch.uint8 channels_last(squeeze/unsqueeze) bilinear (256, 256) -> (320, 320) aa=False    |        383.198 (+-1.230)        |          195.166 (+-1.494)           |    0.509 (+-0.000)  
      4 torch.uint8 channels_last(squeeze/unsqueeze) bilinear (520, 520) -> (32, 32) aa=False      |         57.995 (+-0.057)        |           58.012 (+-0.178)           |    1.000 (+-0.000)  
      4 torch.uint8 channels_last(squeeze/unsqueeze) bilinear (520, 520) -> (224, 224) aa=True     |        373.266 (+-0.588)        |          286.909 (+-1.758)           |    0.769 (+-0.000)  
      4 torch.uint8 channels_last(squeeze/unsqueeze) bilinear (712, 712) -> (32, 32) aa=False      |         75.265 (+-0.299)        |           75.493 (+-0.362)           |    1.003 (+-0.000)  
      4 torch.uint8 channels_last(squeeze/unsqueeze) bilinear (712, 712) -> (224, 224) aa=True     |        510.393 (+-0.966)        |          424.139 (+-3.526)           |    0.831 (+-0.000)  
      4 torch.uint8 channels_last(squeeze/unsqueeze) bilinear (64, 64) -> (224, 224) aa=True       |        166.812 (+-0.267)        |           77.872 (+-0.302)           |    0.467 (+-0.000)  
      4 torch.uint8 channels_last(squeeze/unsqueeze) bilinear (224, 224) -> (270, 268) aa=True     |        280.533 (+-1.370)        |          153.904 (+-1.299)           |    0.549 (+-0.000)  
      4 torch.uint8 channels_last(squeeze/unsqueeze) bilinear (224, 224) -> (64, 64) aa=True       |         74.612 (+-0.269)        |           68.829 (+-0.328)           |    0.922 (+-0.000)  
      4 torch.uint8 channels_last(squeeze/unsqueeze) bilinear (270, 268) -> (224, 224) aa=True     |        244.518 (+-0.842)        |          156.383 (+-1.091)           |    0.640 (+-0.000)  
      4 torch.uint8 channels_last(squeeze/unsqueeze) bilinear (1024, 1024) -> (256, 256) aa=True   |        779.667 (+-5.052)        |          658.256 (+-2.761)           |    0.844 (+-0.000)  
      4 torch.uint8 channels_last(squeeze/unsqueeze) bilinear (64, 64) -> (224, 224) aa=False      |        165.211 (+-1.611)        |           76.139 (+-0.215)           |    0.461 (+-0.000)  
      4 torch.uint8 channels_last(squeeze/unsqueeze) bilinear (224, 224) -> (270, 268) aa=False    |        279.188 (+-1.060)        |          152.335 (+-1.111)           |    0.546 (+-0.000)  
      4 torch.uint8 channels_last(squeeze/unsqueeze) bilinear (224, 224) -> (64, 64) aa=False      |         56.359 (+-0.072)        |           50.788 (+-0.228)           |    0.901 (+-0.000)  
      4 torch.uint8 channels_last(squeeze/unsqueeze) bilinear (270, 268) -> (224, 224) aa=False    |        226.707 (+-0.741)        |          138.531 (+-1.023)           |    0.611 (+-0.000)  
      4 torch.uint8 channels_last(squeeze/unsqueeze) bilinear (256, 256) -> (224, 224) aa=False    |        221.642 (+-0.568)        |          132.777 (+-1.372)           |    0.599 (+-0.000)  
      4 torch.uint8 channels_last(squeeze/unsqueeze) bilinear (1024, 1024) -> (256, 256) aa=False  |        546.038 (+-1.222)        |          421.163 (+-2.376)           |    0.771 (+-0.000)  

Times are in microseconds (us).
