Description:
- 20230511-102028-pr-custom-cases
Torch version: 2.1.0a0+git05154b0
Torch config: PyTorch built with:
  - GCC 9.4
  - C++ Version: 201703
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - CPU capability usage: AVX2
  - CUDA Runtime 11.7
  - NVCC architecture flags: -gencode;arch=compute_61,code=sm_61
  - CuDNN 8.5
  - Build settings: BUILD_TYPE=Release, CXX_COMPILER=/usr/bin/c++, CXX_FLAGS= -D_GLIBCXX_USE_CXX11_ABI=1 -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -DNDEBUG -DUSE_KINETO -DLIBKINETO_NOCUPTI -DLIBKINETO_NOROCTRACER -DUSE_PYTORCH_QNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -O2 -fPIC -Wall -Wextra -Werror=return-type -Werror=non-virtual-dtor -Werror=bool-operation -Wnarrowing -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-unused-parameter -Wno-unused-function -Wno-unused-result -Wno-strict-overflow -Wno-strict-aliasing -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=old-style-cast -Wno-invalid-partial-specialization -Wno-unused-private-field -Wno-aligned-allocation-unavailable -Wno-missing-braces -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Werror=cast-function-type -Wno-stringop-overflow, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, TORCH_DISABLE_GPU_ASSERTS=ON, TORCH_VERSION=2.1.0, USE_CUDA=0, USE_CUDNN=OFF, USE_EIGEN_FOR_BLAS=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=OFF, USE_MKLDNN=0, USE_MPI=OFF, USE_NCCL=OFF, USE_NNPACK=0, USE_OPENMP=ON, USE_ROCM=OFF, 


- 20230511-095327-nightly-custom-cases
Torch version: 2.1.0a0+git5a933d0
Torch config: PyTorch built with:
  - GCC 9.4
  - C++ Version: 201703
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - CPU capability usage: AVX2
  - Build settings: BUILD_TYPE=Release, CXX_COMPILER=/usr/bin/c++, CXX_FLAGS= -D_GLIBCXX_USE_CXX11_ABI=1 -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -DNDEBUG -DUSE_KINETO -DLIBKINETO_NOCUPTI -DLIBKINETO_NOROCTRACER -DUSE_PYTORCH_QNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -O2 -fPIC -Wall -Wextra -Werror=return-type -Werror=non-virtual-dtor -Werror=bool-operation -Wnarrowing -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-unused-parameter -Wno-unused-function -Wno-unused-result -Wno-strict-overflow -Wno-strict-aliasing -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=old-style-cast -Wno-invalid-partial-specialization -Wno-unused-private-field -Wno-aligned-allocation-unavailable -Wno-missing-braces -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Werror=cast-function-type -Wno-stringop-overflow, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, TORCH_DISABLE_GPU_ASSERTS=ON, TORCH_VERSION=2.1.0, USE_CUDA=0, USE_CUDNN=OFF, USE_EIGEN_FOR_BLAS=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=OFF, USE_MKLDNN=0, USE_MPI=OFF, USE_NCCL=OFF, USE_NNPACK=0, USE_OPENMP=ON, USE_ROCM=OFF, 



[----------------------------------------------------------------------------------- Resize ----------------------------------------------------------------------------------]
                                                                               |  Pillow (9.0.0.post1)  |  torch (2.1.0a0+git05154b0) PR  |  torch (2.1.0a0+git5a933d0) nightly
1 threads: --------------------------------------------------------------------------------------------------------------------------------------------------------------------
      3 torch.uint8 channels_first bilinear (256, 256) -> (32, 32) aa=True     |    38.943 (+-0.301)    |        132.319 (+-1.570)        |          129.764 (+-1.318)         
      3 torch.uint8 channels_first bilinear (256, 256) -> (32, 32) aa=False    |                        |        114.214 (+-1.427)        |          112.199 (+-2.807)         
      3 torch.uint8 channels_first bilinear (256, 256) -> (224, 224) aa=True   |   127.725 (+-1.774)    |        299.653 (+-2.784)        |          296.480 (+-1.829)         
      3 torch.uint8 channels_first bilinear (256, 256) -> (224, 224) aa=False  |                        |        287.058 (+-2.783)        |          283.804 (+-2.071)         
      3 torch.uint8 channels_first bilinear (256, 256) -> (320, 320) aa=True   |   179.337 (+-1.639)    |        433.085 (+-2.733)        |          427.981 (+-2.482)         
      3 torch.uint8 channels_first bilinear (256, 256) -> (320, 320) aa=False  |                        |        428.379 (+-2.561)        |          424.882 (+-2.350)         
      3 torch.uint8 channels_first bilinear (520, 520) -> (32, 32) aa=True     |   112.906 (+-1.125)    |        437.536 (+-4.830)        |          475.507 (+-25.314)        
      3 torch.uint8 channels_first bilinear (520, 520) -> (32, 32) aa=False    |                        |        369.696 (+-2.177)        |          366.532 (+-16.445)        
      3 torch.uint8 channels_first bilinear (520, 520) -> (224, 224) aa=True   |   281.916 (+-1.979)    |        714.093 (+-12.612)       |          802.611 (+-6.663)         
      3 torch.uint8 channels_first bilinear (520, 520) -> (224, 224) aa=False  |                        |        603.571 (+-3.907)        |          599.566 (+-4.914)         
      3 torch.uint8 channels_first bilinear (712, 712) -> (32, 32) aa=True     |   186.385 (+-2.180)    |        802.128 (+-18.522)       |          848.287 (+-16.141)        
      3 torch.uint8 channels_first bilinear (712, 712) -> (32, 32) aa=False    |                        |        657.027 (+-3.631)        |          707.306 (+-43.622)        
      3 torch.uint8 channels_first bilinear (712, 712) -> (224, 224) aa=True   |   410.042 (+-2.257)    |       1081.956 (+-27.136)       |         1079.330 (+-107.496)       
      3 torch.uint8 channels_first bilinear (712, 712) -> (224, 224) aa=False  |                        |        937.798 (+-21.380)       |          928.112 (+-2.986)         
      4 torch.uint8 channels_first bilinear (256, 256) -> (32, 32) aa=True     |                        |         75.832 (+-0.429)        |           71.608 (+-0.323)         
      4 torch.uint8 channels_first bilinear (256, 256) -> (32, 32) aa=False    |                        |         58.605 (+-0.272)        |           53.322 (+-0.885)         
      4 torch.uint8 channels_first bilinear (256, 256) -> (224, 224) aa=True   |                        |        257.650 (+-1.798)        |          251.266 (+-5.352)         
      4 torch.uint8 channels_first bilinear (256, 256) -> (224, 224) aa=False  |                        |        244.396 (+-1.798)        |          239.049 (+-4.130)         
      4 torch.uint8 channels_first bilinear (256, 256) -> (320, 320) aa=True   |                        |        408.449 (+-3.584)        |          402.257 (+-7.397)         
      4 torch.uint8 channels_first bilinear (256, 256) -> (320, 320) aa=False  |                        |        405.076 (+-3.242)        |          399.495 (+-7.518)         
      4 torch.uint8 channels_first bilinear (520, 520) -> (32, 32) aa=True     |                        |        227.364 (+-2.566)        |          193.871 (+-4.008)         
      4 torch.uint8 channels_first bilinear (520, 520) -> (32, 32) aa=False    |                        |        160.878 (+-0.996)        |          124.803 (+-2.231)         
      4 torch.uint8 channels_first bilinear (520, 520) -> (224, 224) aa=True   |                        |        481.082 (+-2.510)        |          442.466 (+-9.712)         
      4 torch.uint8 channels_first bilinear (520, 520) -> (224, 224) aa=False  |                        |        409.110 (+-12.062)       |          370.618 (+-5.506)         
      4 torch.uint8 channels_first bilinear (712, 712) -> (32, 32) aa=True     |                        |        380.505 (+-2.700)        |          380.165 (+-47.392)        
      4 torch.uint8 channels_first bilinear (712, 712) -> (32, 32) aa=False    |                        |        399.175 (+-21.729)       |          222.327 (+-40.410)        
      4 torch.uint8 channels_first bilinear (712, 712) -> (224, 224) aa=True   |                        |        702.089 (+-3.752)        |          646.472 (+-5.180)         
      4 torch.uint8 channels_first bilinear (712, 712) -> (224, 224) aa=False  |                        |        795.081 (+-84.777)       |          489.255 (+-19.505)        
      3 torch.uint8 channels_last bilinear (256, 256) -> (32, 32) aa=True      |    38.723 (+-0.307)    |         56.075 (+-0.260)        |           58.334 (+-0.344)         
      3 torch.uint8 channels_last bilinear (256, 256) -> (32, 32) aa=False     |                        |         37.671 (+-0.218)        |           38.774 (+-0.189)         
      3 torch.uint8 channels_last bilinear (256, 256) -> (224, 224) aa=True    |   127.398 (+-1.151)    |        153.049 (+-1.348)        |          161.690 (+-3.792)         
      3 torch.uint8 channels_last bilinear (256, 256) -> (224, 224) aa=False   |                        |        141.931 (+-2.112)        |          153.355 (+-2.529)         
      3 torch.uint8 channels_last bilinear (256, 256) -> (320, 320) aa=True    |   178.662 (+-2.345)    |        209.490 (+-1.924)        |          223.171 (+-2.028)         
      3 torch.uint8 channels_last bilinear (256, 256) -> (320, 320) aa=False   |                        |        207.528 (+-1.701)        |          222.559 (+-2.634)         
      3 torch.uint8 channels_last bilinear (520, 520) -> (32, 32) aa=True      |   112.790 (+-1.163)    |        130.676 (+-1.998)        |          136.929 (+-1.576)         
      3 torch.uint8 channels_last bilinear (520, 520) -> (32, 32) aa=False     |                        |         57.102 (+-0.371)        |           60.228 (+-0.204)         
      3 torch.uint8 channels_last bilinear (520, 520) -> (224, 224) aa=True    |   282.302 (+-2.040)    |        328.881 (+-2.056)        |          346.151 (+-5.729)         
      3 torch.uint8 channels_last bilinear (520, 520) -> (224, 224) aa=False   |                        |        231.981 (+-1.813)        |          258.213 (+-5.367)         
      3 torch.uint8 channels_last bilinear (712, 712) -> (32, 32) aa=True      |   186.981 (+-4.687)    |        202.737 (+-2.250)        |          212.205 (+-3.140)         
      3 torch.uint8 channels_last bilinear (712, 712) -> (32, 32) aa=False     |                        |         76.345 (+-0.241)        |           79.435 (+-0.968)         
      3 torch.uint8 channels_last bilinear (712, 712) -> (224, 224) aa=True    |   408.340 (+-2.972)    |        448.730 (+-2.850)        |          478.867 (+-2.611)         
      3 torch.uint8 channels_last bilinear (712, 712) -> (224, 224) aa=False   |                        |        303.085 (+-1.991)        |          340.433 (+-2.943)         
      4 torch.uint8 channels_last bilinear (256, 256) -> (32, 32) aa=True      |                        |         53.308 (+-0.232)        |           52.584 (+-0.267)         
      4 torch.uint8 channels_last bilinear (256, 256) -> (32, 32) aa=False     |                        |         36.197 (+-0.216)        |           34.642 (+-0.187)         
      4 torch.uint8 channels_last bilinear (256, 256) -> (224, 224) aa=True    |                        |        145.908 (+-1.082)        |          144.123 (+-1.188)         
      4 torch.uint8 channels_last bilinear (256, 256) -> (224, 224) aa=False   |                        |        133.643 (+-1.177)        |          132.248 (+-1.062)         
      4 torch.uint8 channels_last bilinear (256, 256) -> (320, 320) aa=True    |                        |        198.717 (+-1.513)        |          195.972 (+-1.718)         
      4 torch.uint8 channels_last bilinear (256, 256) -> (320, 320) aa=False   |                        |        196.239 (+-1.501)        |          194.776 (+-3.429)         
      4 torch.uint8 channels_last bilinear (520, 520) -> (32, 32) aa=True      |                        |        125.645 (+-1.703)        |          125.502 (+-2.750)         
      4 torch.uint8 channels_last bilinear (520, 520) -> (32, 32) aa=False     |                        |         57.455 (+-0.258)        |           56.432 (+-0.678)         
      4 torch.uint8 channels_last bilinear (520, 520) -> (224, 224) aa=True    |                        |        289.600 (+-1.906)        |          283.961 (+-1.443)         
      4 torch.uint8 channels_last bilinear (520, 520) -> (224, 224) aa=False   |                        |        216.525 (+-1.812)        |          214.160 (+-2.753)         
      4 torch.uint8 channels_last bilinear (712, 712) -> (32, 32) aa=True      |                        |        193.339 (+-2.152)        |          194.618 (+-2.350)         
      4 torch.uint8 channels_last bilinear (712, 712) -> (32, 32) aa=False     |                        |         74.541 (+-0.322)        |           73.852 (+-0.334)         
      4 torch.uint8 channels_last bilinear (712, 712) -> (224, 224) aa=True    |                        |        426.521 (+-2.853)        |          422.634 (+-2.714)         
      4 torch.uint8 channels_last bilinear (712, 712) -> (224, 224) aa=False   |                        |        277.947 (+-2.444)        |          273.958 (+-2.125)         

Times are in microseconds (us).
