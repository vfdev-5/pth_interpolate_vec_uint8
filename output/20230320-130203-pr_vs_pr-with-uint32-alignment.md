Description:
- 20230315-161003-pr
Torch version: 2.1.0a0+git0968a5d
Torch config: PyTorch built with:
  - GCC 9.4
  - C++ Version: 201703
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - CPU capability usage: AVX2
  - Build settings: BUILD_TYPE=Release, CXX_COMPILER=/usr/bin/c++, CXX_FLAGS= -D_GLIBCXX_USE_CXX11_ABI=1 -Wno-deprecated -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -DNDEBUG -DUSE_KINETO -DLIBKINETO_NOCUPTI -DLIBKINETO_NOROCTRACER -DUSE_PYTORCH_QNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -O2 -fPIC -Wall -Wextra -Werror=return-type -Werror=non-virtual-dtor -Werror=bool-operation -Wnarrowing -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-unused-parameter -Wno-unused-function -Wno-unused-result -Wno-strict-overflow -Wno-strict-aliasing -Wno-error=deprecated-declarations -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=old-style-cast -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Werror=cast-function-type -Wno-stringop-overflow, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, TORCH_DISABLE_GPU_ASSERTS=ON, TORCH_VERSION=2.1.0, USE_CUDA=0, USE_CUDNN=OFF, USE_EIGEN_FOR_BLAS=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=OFF, USE_MKLDNN=0, USE_MPI=OFF, USE_NCCL=OFF, USE_NNPACK=0, USE_OPENMP=ON, USE_ROCM=OFF, 


- 20230320-120138-pr
Torch version: 2.1.0a0+gitbe96393
Torch config: PyTorch built with:
  - GCC 9.4
  - C++ Version: 201703
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - CPU capability usage: AVX2
  - Build settings: BUILD_TYPE=Release, CXX_COMPILER=/usr/bin/c++, CXX_FLAGS= -D_GLIBCXX_USE_CXX11_ABI=1 -Wno-deprecated -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -DNDEBUG -DUSE_KINETO -DLIBKINETO_NOCUPTI -DLIBKINETO_NOROCTRACER -DUSE_PYTORCH_QNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -O2 -fPIC -Wall -Wextra -Werror=return-type -Werror=non-virtual-dtor -Werror=bool-operation -Wnarrowing -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-unused-parameter -Wno-unused-function -Wno-unused-result -Wno-strict-overflow -Wno-strict-aliasing -Wno-error=deprecated-declarations -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=old-style-cast -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Werror=cast-function-type -Wno-stringop-overflow, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, TORCH_DISABLE_GPU_ASSERTS=ON, TORCH_VERSION=2.1.0, USE_CUDA=0, USE_CUDNN=OFF, USE_EIGEN_FOR_BLAS=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=OFF, USE_MKLDNN=0, USE_MPI=OFF, USE_NCCL=OFF, USE_NNPACK=0, USE_OPENMP=ON, USE_ROCM=OFF, 



[------------------------------------------------------------------------- Resize -------------------------------------------------------------------------]
                                                                 |  Pillow (9.0.0.post1)  |  torch (2.1.0a0+git0968a5d) PR  |  torch (2.1.0a0+gitbe96393) PR
1 threads: -------------------------------------------------------------------------------------------------------------------------------------------------
      3 torch.uint8 channels_last bilinear 256 -> 32 aa=True     |    38.847 (+-0.355)    |         56.639 (+-0.688)        |         55.468 (+-0.356)      
      3 torch.uint8 channels_last bilinear 256 -> 32 aa=False    |                        |         36.852 (+-0.242)        |         36.476 (+-0.168)      
      3 torch.uint8 channels_last bilinear 256 -> 224 aa=True    |   127.457 (+-1.093)    |        152.491 (+-1.016)        |        151.420 (+-1.105)      
      3 torch.uint8 channels_last bilinear 256 -> 224 aa=False   |                        |        141.122 (+-1.249)        |        140.580 (+-1.550)      
      3 torch.uint8 channels_last bilinear 256 -> 320 aa=True    |   179.007 (+-1.389)    |        208.752 (+-3.570)        |        209.069 (+-1.601)      
      3 torch.uint8 channels_last bilinear 256 -> 320 aa=False   |                        |        206.377 (+-1.920)        |        206.791 (+-1.349)      
      3 torch.uint8 channels_last bilinear 520 -> 32 aa=True     |   113.624 (+-1.129)    |        132.119 (+-0.976)        |        130.918 (+-1.240)      
      3 torch.uint8 channels_last bilinear 520 -> 32 aa=False    |                        |         57.153 (+-1.154)        |         57.027 (+-0.225)      
      3 torch.uint8 channels_last bilinear 520 -> 224 aa=True    |   282.461 (+-5.087)    |        327.359 (+-2.021)        |        321.758 (+-2.305)      
      3 torch.uint8 channels_last bilinear 520 -> 224 aa=False   |                        |        230.250 (+-1.707)        |        230.008 (+-1.772)      
      3 torch.uint8 channels_last bilinear 712 -> 32 aa=True     |   186.439 (+-1.718)    |        210.530 (+-1.524)        |        199.729 (+-1.588)      
      3 torch.uint8 channels_last bilinear 712 -> 32 aa=False    |                        |         75.578 (+-0.384)        |         75.289 (+-0.317)      
      3 torch.uint8 channels_last bilinear 712 -> 224 aa=True    |   410.050 (+-2.545)    |        450.945 (+-3.326)        |        448.719 (+-2.329)      
      3 torch.uint8 channels_last bilinear 712 -> 224 aa=False   |                        |        298.673 (+-1.821)        |        297.855 (+-1.414)      
      4 torch.uint8 channels_last bilinear 256 -> 32 aa=True     |                        |         51.937 (+-0.729)        |         52.436 (+-0.243)      
      4 torch.uint8 channels_last bilinear 256 -> 32 aa=False    |                        |         35.267 (+-0.187)        |         34.707 (+-0.172)      
      4 torch.uint8 channels_last bilinear 256 -> 224 aa=True    |                        |        144.444 (+-1.987)        |        143.676 (+-1.037)      
      4 torch.uint8 channels_last bilinear 256 -> 224 aa=False   |                        |        131.150 (+-0.922)        |        129.787 (+-1.077)      
      4 torch.uint8 channels_last bilinear 256 -> 320 aa=True    |                        |        195.818 (+-1.386)        |        194.540 (+-1.470)      
      4 torch.uint8 channels_last bilinear 256 -> 320 aa=False   |                        |        193.129 (+-1.089)        |        192.183 (+-1.283)      
      4 torch.uint8 channels_last bilinear 520 -> 32 aa=True     |                        |        124.417 (+-1.151)        |        124.630 (+-1.436)      
      4 torch.uint8 channels_last bilinear 520 -> 32 aa=False    |                        |         56.563 (+-0.295)        |         56.674 (+-0.261)      
      4 torch.uint8 channels_last bilinear 520 -> 224 aa=True    |                        |        285.209 (+-2.073)        |        285.944 (+-2.820)      
      4 torch.uint8 channels_last bilinear 520 -> 224 aa=False   |                        |        213.006 (+-1.435)        |        213.521 (+-3.459)      
      4 torch.uint8 channels_last bilinear 712 -> 32 aa=True     |                        |        192.368 (+-2.478)        |        193.147 (+-3.638)      
      4 torch.uint8 channels_last bilinear 712 -> 32 aa=False    |                        |         73.897 (+-0.207)        |         73.874 (+-0.256)      
      4 torch.uint8 channels_last bilinear 712 -> 224 aa=True    |                        |        422.955 (+-2.609)        |        421.302 (+-3.619)      
      4 torch.uint8 channels_last bilinear 712 -> 224 aa=False   |                        |        273.497 (+-1.930)        |        271.863 (+-1.694)      
      3 torch.uint8 channels_first bilinear 256 -> 32 aa=True    |    38.909 (+-0.307)    |        129.316 (+-1.082)        |        128.518 (+-0.948)      
      3 torch.uint8 channels_first bilinear 256 -> 32 aa=False   |                        |        112.555 (+-1.180)        |        111.931 (+-0.910)      
      3 torch.uint8 channels_first bilinear 256 -> 224 aa=True   |   128.044 (+-1.063)    |        295.897 (+-1.861)        |        296.909 (+-1.861)      
      3 torch.uint8 channels_first bilinear 256 -> 224 aa=False  |                        |        283.760 (+-1.675)        |        282.657 (+-1.644)      
      3 torch.uint8 channels_first bilinear 256 -> 320 aa=True   |   178.242 (+-3.556)    |        428.809 (+-2.829)        |        427.292 (+-10.454)     
      3 torch.uint8 channels_first bilinear 256 -> 320 aa=False  |                        |        425.761 (+-3.370)        |        425.962 (+-5.284)      
      3 torch.uint8 channels_first bilinear 520 -> 32 aa=True    |   113.665 (+-1.179)    |        434.279 (+-2.296)        |        469.924 (+-6.481)      
      3 torch.uint8 channels_first bilinear 520 -> 32 aa=False   |                        |        366.346 (+-1.958)        |        365.756 (+-1.918)      
      3 torch.uint8 channels_first bilinear 520 -> 224 aa=True   |   281.591 (+-1.958)    |        669.354 (+-2.336)        |        671.422 (+-3.119)      
      3 torch.uint8 channels_first bilinear 520 -> 224 aa=False  |                        |        598.671 (+-3.548)        |        597.461 (+-3.241)      
      3 torch.uint8 channels_first bilinear 712 -> 32 aa=True    |   185.709 (+-1.386)    |        780.482 (+-10.068)       |        771.339 (+-4.534)      
      3 torch.uint8 channels_first bilinear 712 -> 32 aa=False   |                        |        653.348 (+-3.395)        |        713.670 (+-10.287)     
      3 torch.uint8 channels_first bilinear 712 -> 224 aa=True   |   410.688 (+-2.840)    |       1246.382 (+-101.843)      |       1075.254 (+-179.831)    
      3 torch.uint8 channels_first bilinear 712 -> 224 aa=False  |                        |        926.085 (+-5.736)        |        927.910 (+-4.932)      
      4 torch.uint8 channels_first bilinear 256 -> 32 aa=True    |                        |         70.656 (+-0.276)        |         71.067 (+-0.356)      
      4 torch.uint8 channels_first bilinear 256 -> 32 aa=False   |                        |         53.848 (+-0.233)        |         53.822 (+-0.283)      
      4 torch.uint8 channels_first bilinear 256 -> 224 aa=True   |                        |        251.071 (+-1.609)        |        262.795 (+-1.436)      
      4 torch.uint8 channels_first bilinear 256 -> 224 aa=False  |                        |        238.187 (+-1.872)        |        249.845 (+-1.818)      
      4 torch.uint8 channels_first bilinear 256 -> 320 aa=True   |                        |        398.135 (+-2.249)        |        424.147 (+-2.293)      
      4 torch.uint8 channels_first bilinear 256 -> 320 aa=False  |                        |        394.836 (+-2.478)        |        419.775 (+-2.724)      
      4 torch.uint8 channels_first bilinear 520 -> 32 aa=True    |                        |        193.845 (+-1.013)        |        234.404 (+-10.588)     
      4 torch.uint8 channels_first bilinear 520 -> 32 aa=False   |                        |        124.757 (+-0.778)        |        125.255 (+-0.989)      
      4 torch.uint8 channels_first bilinear 520 -> 224 aa=True   |                        |        441.936 (+-2.096)        |        454.526 (+-3.137)      
      4 torch.uint8 channels_first bilinear 520 -> 224 aa=False  |                        |        369.701 (+-3.230)        |        381.531 (+-2.511)      
      4 torch.uint8 channels_first bilinear 712 -> 32 aa=True    |                        |        321.851 (+-4.140)        |        424.572 (+-17.113)     
      4 torch.uint8 channels_first bilinear 712 -> 32 aa=False   |                        |       200.803 (+-107.464)       |        200.332 (+-1.269)      
      4 torch.uint8 channels_first bilinear 712 -> 224 aa=True   |                        |        686.683 (+-5.951)        |        651.885 (+-3.583)      
      4 torch.uint8 channels_first bilinear 712 -> 224 aa=False  |                        |        771.705 (+-4.404)        |        500.596 (+-3.471)      

Times are in microseconds (us).
