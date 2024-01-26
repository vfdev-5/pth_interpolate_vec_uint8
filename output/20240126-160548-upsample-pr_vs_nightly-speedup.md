Description:

- 20240126-160339-upsample-PR
Torch version: 2.3.0a0+gitfda85a6
Torch config: PyTorch built with:
  - GCC 9.4
  - C++ Version: 201703
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - CPU capability usage: AVX2
  - CUDA Runtime 12.1
  - NVCC architecture flags: -gencode;arch=compute_89,code=sm_89;-gencode;arch=compute_61,code=sm_61
  - CuDNN 8.9
  - Build settings: BUILD_TYPE=Release, CUDA_VERSION=12.1, CUDNN_VERSION=8.9.0, CXX_COMPILER=/usr/bin/c++, CXX_FLAGS= -D_GLIBCXX_USE_CXX11_ABI=1 -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -DNDEBUG -DUSE_PYTORCH_QNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -O2 -fPIC -Wall -Wextra -Werror=return-type -Werror=non-virtual-dtor -Werror=bool-operation -Wnarrowing -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-unused-parameter -Wno-unused-function -Wno-unused-result -Wno-strict-overflow -Wno-strict-aliasing -Wno-stringop-overflow -Wsuggest-override -Wno-psabi -Wno-error=pedantic -Wno-error=old-style-cast -Wno-missing-braces -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Wno-stringop-overflow, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, TORCH_VERSION=2.3.0, USE_CUDA=1, USE_CUDNN=1, USE_CUSPARSELT=OFF, USE_EIGEN_FOR_BLAS=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_GLOO=OFF, USE_MKL=OFF, USE_MKLDNN=0, USE_MPI=OFF, USE_NCCL=0, USE_NNPACK=0, USE_OPENMP=ON, USE_ROCM=OFF, USE_ROCM_KERNEL_ASSERT=OFF, 


- 20240126-155646-upsample-Nightly
Torch version: 2.3.0a0+git0d1e705
Torch config: PyTorch built with:
  - GCC 9.4
  - C++ Version: 201703
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - CPU capability usage: AVX2
  - CUDA Runtime 12.1
  - NVCC architecture flags: -gencode;arch=compute_89,code=sm_89;-gencode;arch=compute_61,code=sm_61
  - CuDNN 8.9
  - Build settings: BUILD_TYPE=Release, CUDA_VERSION=12.1, CUDNN_VERSION=8.9.0, CXX_COMPILER=/usr/bin/c++, CXX_FLAGS= -D_GLIBCXX_USE_CXX11_ABI=1 -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -DNDEBUG -DUSE_PYTORCH_QNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -O2 -fPIC -Wall -Wextra -Werror=return-type -Werror=non-virtual-dtor -Werror=bool-operation -Wnarrowing -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-unused-parameter -Wno-unused-function -Wno-unused-result -Wno-strict-overflow -Wno-strict-aliasing -Wno-stringop-overflow -Wsuggest-override -Wno-psabi -Wno-error=pedantic -Wno-error=old-style-cast -Wno-missing-braces -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Wno-stringop-overflow, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, TORCH_VERSION=2.3.0, USE_CUDA=1, USE_CUDNN=1, USE_CUSPARSELT=OFF, USE_EIGEN_FOR_BLAS=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_GLOO=OFF, USE_MKL=OFF, USE_MKLDNN=0, USE_MPI=OFF, USE_NCCL=0, USE_NNPACK=0, USE_OPENMP=ON, USE_ROCM=OFF, USE_ROCM_KERNEL_ASSERT=OFF, 



[------------------------------------------------------------------------------------ Resize ------------------------------------------------------------------------------------]
                                                                               |  torch (2.3.0a0+gitfda85a6) PR  |  torch (2.3.0a0+git0d1e705) Nightly  |  Speed-up: PR vs Nightly
1 threads: -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
      3 torch.uint8 channels_first bilinear (400, 400) -> (224, 224) aa=False  |        440.996 (+-2.044)        |          470.824 (+-5.927)           |      1.068 (+-0.000)    
      3 torch.uint8 channels_first bicubic (400, 400) -> (224, 224) aa=False   |        463.565 (+-1.519)        |          497.231 (+-10.825)          |      1.073 (+-0.000)    
      3 torch.uint8 channels_first bilinear (400, 400) -> (700, 700) aa=False  |       1717.000 (+-28.589)       |         1915.570 (+-43.397)          |      1.116 (+-0.000)    
      3 torch.uint8 channels_first bicubic (400, 400) -> (700, 700) aa=False   |       1801.954 (+-22.391)       |         1981.501 (+-37.034)          |      1.100 (+-0.000)    
      3 torch.uint8 channels_last bilinear (400, 400) -> (224, 224) aa=False   |        199.599 (+-0.851)        |          196.535 (+-3.788)           |      0.985 (+-0.000)    
      3 torch.uint8 channels_last bicubic (400, 400) -> (224, 224) aa=False    |        243.126 (+-0.681)        |          240.695 (+-2.306)           |      0.990 (+-0.000)    
      3 torch.uint8 channels_last bilinear (400, 400) -> (700, 700) aa=False   |        686.270 (+-2.870)        |          687.769 (+-17.863)          |      1.002 (+-0.000)    
      3 torch.uint8 channels_last bicubic (400, 400) -> (700, 700) aa=False    |        899.509 (+-5.377)        |          899.063 (+-9.001)           |      1.000 (+-0.000)    

Times are in microseconds (us).
