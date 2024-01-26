Description:
- 20240126-153121-upsample-PR
Torch version: 2.3.0a0+git76ea7ac
Torch config: PyTorch built with:
  - GCC 9.4
  - C++ Version: 201703
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - CPU capability usage: AVX2
  - CUDA Runtime 12.1
  - NVCC architecture flags: -gencode;arch=compute_89,code=sm_89;-gencode;arch=compute_61,code=sm_61
  - CuDNN 8.9
  - Build settings: BUILD_TYPE=Release, CUDA_VERSION=12.1, CUDNN_VERSION=8.9.0, CXX_COMPILER=/usr/bin/c++, CXX_FLAGS= -D_GLIBCXX_USE_CXX11_ABI=1 -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -DNDEBUG -DUSE_PYTORCH_QNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -O2 -fPIC -Wall -Wextra -Werror=return-type -Werror=non-virtual-dtor -Werror=bool-operation -Wnarrowing -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-unused-parameter -Wno-unused-function -Wno-unused-result -Wno-strict-overflow -Wno-strict-aliasing -Wno-stringop-overflow -Wsuggest-override -Wno-psabi -Wno-error=pedantic -Wno-error=old-style-cast -Wno-missing-braces -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Wno-stringop-overflow, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, TORCH_VERSION=2.3.0, USE_CUDA=1, USE_CUDNN=1, USE_CUSPARSELT=OFF, USE_EIGEN_FOR_BLAS=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_GLOO=OFF, USE_MKL=OFF, USE_MKLDNN=0, USE_MPI=OFF, USE_NCCL=0, USE_NNPACK=0, USE_OPENMP=ON, USE_ROCM=OFF, USE_ROCM_KERNEL_ASSERT=OFF, 


- 20240126-152854-upsample-Nightly
Torch version: 2.3.0a0+gitde89a53
Torch config: PyTorch built with:
  - GCC 9.4
  - C++ Version: 201703
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - CPU capability usage: AVX2
  - CUDA Runtime 12.1
  - NVCC architecture flags: -gencode;arch=compute_89,code=sm_89;-gencode;arch=compute_61,code=sm_61
  - CuDNN 8.9
  - Build settings: BUILD_TYPE=Release, CUDA_VERSION=12.1, CUDNN_VERSION=8.9.0, CXX_COMPILER=/usr/bin/c++, CXX_FLAGS= -D_GLIBCXX_USE_CXX11_ABI=1 -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -DNDEBUG -DUSE_PYTORCH_QNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -O2 -fPIC -Wall -Wextra -Werror=return-type -Werror=non-virtual-dtor -Werror=bool-operation -Wnarrowing -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-unused-parameter -Wno-unused-function -Wno-unused-result -Wno-strict-overflow -Wno-strict-aliasing -Wno-stringop-overflow -Wsuggest-override -Wno-psabi -Wno-error=pedantic -Wno-error=old-style-cast -Wno-missing-braces -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Wno-stringop-overflow, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, TORCH_VERSION=2.3.0, USE_CUDA=1, USE_CUDNN=1, USE_CUSPARSELT=OFF, USE_EIGEN_FOR_BLAS=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_GLOO=OFF, USE_MKL=OFF, USE_MKLDNN=0, USE_MPI=OFF, USE_NCCL=0, USE_NNPACK=0, USE_OPENMP=ON, USE_ROCM=OFF, USE_ROCM_KERNEL_ASSERT=OFF, 



[---------------------------------------------------------------------- Resize ----------------------------------------------------------------------]
                                                                               |  torch (2.3.0a0+git76ea7ac) PR  |  torch (2.3.0a0+gitde89a53) Nightly
1 threads: -------------------------------------------------------------------------------------------------------------------------------------------
      3 torch.uint8 channels_first bilinear (400, 400) -> (224, 224) aa=False  |        434.993 (+-2.928)        |          442.750 (+-3.734)         
      3 torch.uint8 channels_first bicubic (400, 400) -> (224, 224) aa=False   |        459.381 (+-1.109)        |          471.058 (+-1.642)         
      3 torch.uint8 channels_first bilinear (400, 400) -> (700, 700) aa=False  |       2687.804 (+-178.151)      |         2677.397 (+-213.588)       
      3 torch.uint8 channels_first bicubic (400, 400) -> (700, 700) aa=False   |       1779.563 (+-18.781)       |         1816.712 (+-21.341)        
      3 torch.uint8 channels_last bilinear (400, 400) -> (224, 224) aa=False   |        198.439 (+-0.702)        |          196.839 (+-0.624)         
      3 torch.uint8 channels_last bicubic (400, 400) -> (224, 224) aa=False    |        242.129 (+-0.659)        |          241.271 (+-0.879)         
      3 torch.uint8 channels_last bilinear (400, 400) -> (700, 700) aa=False   |        690.161 (+-2.943)        |          685.099 (+-5.838)         
      3 torch.uint8 channels_last bicubic (400, 400) -> (700, 700) aa=False    |        896.004 (+-4.117)        |          930.177 (+-19.051)        

Times are in microseconds (us).
