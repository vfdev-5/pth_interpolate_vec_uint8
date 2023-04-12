Description:
- 20230321-142033-pr
Torch version: 2.1.0a0+git8d955df
Torch config: PyTorch built with:
  - GCC 9.4
  - C++ Version: 201703
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - CPU capability usage: AVX2
  - Build settings: BUILD_TYPE=Release, CXX_COMPILER=/usr/bin/c++, CXX_FLAGS= -D_GLIBCXX_USE_CXX11_ABI=1 -Wno-deprecated -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -DNDEBUG -DUSE_KINETO -DLIBKINETO_NOCUPTI -DLIBKINETO_NOROCTRACER -DUSE_PYTORCH_QNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -O2 -fPIC -Wall -Wextra -Werror=return-type -Werror=non-virtual-dtor -Werror=bool-operation -Wnarrowing -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-unused-parameter -Wno-unused-function -Wno-unused-result -Wno-strict-overflow -Wno-strict-aliasing -Wno-error=deprecated-declarations -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=old-style-cast -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Werror=cast-function-type -Wno-stringop-overflow, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, TORCH_DISABLE_GPU_ASSERTS=ON, TORCH_VERSION=2.1.0, USE_CUDA=0, USE_CUDNN=OFF, USE_EIGEN_FOR_BLAS=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=OFF, USE_MKLDNN=0, USE_MPI=OFF, USE_NCCL=OFF, USE_NNPACK=0, USE_OPENMP=ON, USE_ROCM=OFF, 


- 20230320-153534-nightly
Torch version: 2.1.0a0+git5309c44
Torch config: PyTorch built with:
  - GCC 9.4
  - C++ Version: 201703
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - CPU capability usage: AVX2
  - Build settings: BUILD_TYPE=Release, CXX_COMPILER=/usr/bin/c++, CXX_FLAGS= -D_GLIBCXX_USE_CXX11_ABI=1 -Wno-deprecated -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -DNDEBUG -DUSE_KINETO -DLIBKINETO_NOCUPTI -DLIBKINETO_NOROCTRACER -DUSE_PYTORCH_QNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -O2 -fPIC -Wall -Wextra -Werror=return-type -Werror=non-virtual-dtor -Werror=bool-operation -Wnarrowing -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-unused-parameter -Wno-unused-function -Wno-unused-result -Wno-strict-overflow -Wno-strict-aliasing -Wno-error=deprecated-declarations -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=old-style-cast -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Werror=cast-function-type -Wno-stringop-overflow, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, TORCH_DISABLE_GPU_ASSERTS=ON, TORCH_VERSION=2.1.0, USE_CUDA=0, USE_CUDNN=OFF, USE_EIGEN_FOR_BLAS=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=OFF, USE_MKLDNN=0, USE_MPI=OFF, USE_NCCL=OFF, USE_NNPACK=0, USE_OPENMP=ON, USE_ROCM=OFF, 



[-------------------------------------------------------------------------------------------------- Resize -------------------------------------------------------------------------------------------------]
                                                                                 |  Pillow (9.0.0.post1)  |  torch (2.1.0a0+git8d955df) PR  |  torch (2.1.0a0+git5309c44) nightly  |  Speed-up: PR vs nightly
1 threads: --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      3 torch.uint8 channels_last bilinear (256, 256) -> (32, 32) aa=True        |    38.649 (+-0.306)    |         55.828 (+-0.370)        |          132.147 (+-1.236)           |      2.367 (+-0.000)    
      3 torch.uint8 channels_last bilinear (256, 256) -> (32, 32) aa=False       |                        |         36.826 (+-0.229)        |          111.789 (+-1.175)           |      3.036 (+-0.000)    
      3 torch.uint8 channels_last bilinear (256, 256) -> (224, 224) aa=True      |   128.233 (+-1.313)    |        153.827 (+-1.229)        |          302.518 (+-2.632)           |      1.967 (+-0.000)    
      3 torch.uint8 channels_last bilinear (256, 256) -> (224, 224) aa=False     |                        |        143.886 (+-1.409)        |          286.663 (+-2.494)           |      1.992 (+-0.000)    
      3 torch.uint8 channels_last bilinear (256, 256) -> (320, 320) aa=True      |   179.504 (+-1.825)    |        211.569 (+-1.336)        |          439.375 (+-4.014)           |      2.077 (+-0.000)    
      3 torch.uint8 channels_last bilinear (256, 256) -> (320, 320) aa=False     |                        |        209.888 (+-1.443)        |          438.537 (+-4.143)           |      2.089 (+-0.000)    
      3 torch.uint8 channels_last bilinear (520, 520) -> (32, 32) aa=True        |   112.891 (+-1.118)    |        129.373 (+-1.396)        |          446.804 (+-3.283)           |      3.454 (+-0.000)    
      3 torch.uint8 channels_last bilinear (520, 520) -> (32, 32) aa=False       |                        |         56.858 (+-0.227)        |          374.244 (+-13.598)          |      6.582 (+-0.000)    
      3 torch.uint8 channels_last bilinear (520, 520) -> (224, 224) aa=True      |   282.917 (+-2.992)    |        324.378 (+-1.694)        |          720.197 (+-3.467)           |      2.220 (+-0.000)    
      3 torch.uint8 channels_last bilinear (520, 520) -> (224, 224) aa=False     |                        |        236.078 (+-1.679)        |          592.834 (+-3.903)           |      2.511 (+-0.000)    
      3 torch.uint8 channels_last bilinear (712, 712) -> (32, 32) aa=True        |   185.595 (+-1.633)    |        202.000 (+-1.920)        |          787.868 (+-3.648)           |      3.900 (+-0.000)    
      3 torch.uint8 channels_last bilinear (712, 712) -> (32, 32) aa=False       |                        |         75.421 (+-0.512)        |          651.016 (+-3.926)           |      8.632 (+-0.000)    
      3 torch.uint8 channels_last bilinear (712, 712) -> (224, 224) aa=True      |   409.691 (+-2.735)    |        449.927 (+-2.500)        |         1123.923 (+-14.988)          |      2.498 (+-0.000)    
      3 torch.uint8 channels_last bilinear (712, 712) -> (224, 224) aa=False     |                        |        306.691 (+-2.095)        |          915.347 (+-4.486)           |      2.985 (+-0.000)    
      3 torch.uint8 channels_last bilinear (64, 64) -> (224, 224) aa=True        |    60.740 (+-0.278)    |         78.745 (+-0.286)        |          170.465 (+-1.830)           |      2.165 (+-0.000)    
      3 torch.uint8 channels_last bilinear (224, 224) -> (270, 268) aa=True      |   133.029 (+-1.619)    |        162.393 (+-1.289)        |          330.971 (+-3.249)           |      2.038 (+-0.000)    
      3 torch.uint8 channels_last bilinear (256, 256) -> (1024, 1024) aa=True    |   948.849 (+-2.749)    |        896.127 (+-3.696)        |         2805.510 (+-25.503)          |      3.131 (+-0.000)    
      3 torch.uint8 channels_last bilinear (224, 224) -> (64, 64) aa=True        |    52.505 (+-0.319)    |         70.617 (+-0.344)        |          135.933 (+-1.625)           |      1.925 (+-0.000)    
      3 torch.uint8 channels_last bilinear (270, 268) -> (224, 224) aa=True      |   138.671 (+-1.953)    |        165.638 (+-1.473)        |          321.112 (+-2.904)           |      1.939 (+-0.000)    
      3 torch.uint8 channels_last bilinear (1024, 1024) -> (256, 256) aa=True    |   689.492 (+-2.917)    |        758.162 (+-3.719)        |         2050.880 (+-22.188)          |      2.705 (+-0.000)    
      3 torch.uint8 channels_last bilinear (64, 64) -> (224, 224) aa=False       |                        |         77.300 (+-0.307)        |          169.646 (+-1.640)           |      2.195 (+-0.000)    
      3 torch.uint8 channels_last bilinear (224, 224) -> (270, 268) aa=False     |                        |        159.525 (+-1.225)        |          329.754 (+-2.590)           |      2.067 (+-0.000)    
      3 torch.uint8 channels_last bilinear (256, 256) -> (1024, 1024) aa=False   |                        |        890.106 (+-3.358)        |         2815.870 (+-22.589)          |      3.164 (+-0.000)    
      3 torch.uint8 channels_last bilinear (224, 224) -> (64, 64) aa=False       |                        |         52.399 (+-0.314)        |          112.024 (+-1.225)           |      2.138 (+-0.000)    
      3 torch.uint8 channels_last bilinear (270, 268) -> (224, 224) aa=False     |                        |        148.780 (+-1.282)        |          299.152 (+-3.353)           |      2.011 (+-0.000)    
      3 torch.uint8 channels_last bilinear (1024, 1024) -> (256, 256) aa=False   |                        |        479.273 (+-3.432)        |         1698.601 (+-16.785)          |      3.544 (+-0.000)    
      4 torch.uint8 channels_last bilinear (256, 256) -> (32, 32) aa=True        |                        |         52.871 (+-0.198)        |           55.867 (+-0.340)           |      1.057 (+-0.000)    
      4 torch.uint8 channels_last bilinear (256, 256) -> (32, 32) aa=False       |                        |         35.115 (+-0.183)        |           34.679 (+-0.234)           |      0.988 (+-0.000)    
      4 torch.uint8 channels_last bilinear (256, 256) -> (224, 224) aa=True      |                        |        144.210 (+-1.401)        |          141.508 (+-3.320)           |      0.981 (+-0.000)    
      4 torch.uint8 channels_last bilinear (256, 256) -> (224, 224) aa=False     |                        |        132.139 (+-1.088)        |          125.776 (+-1.390)           |      0.952 (+-0.000)    
      4 torch.uint8 channels_last bilinear (256, 256) -> (320, 320) aa=True      |                        |        200.041 (+-1.704)        |          186.273 (+-1.759)           |      0.931 (+-0.000)    
      4 torch.uint8 channels_last bilinear (256, 256) -> (320, 320) aa=False     |                        |        197.051 (+-1.808)        |          184.272 (+-2.248)           |      0.935 (+-0.000)    
      4 torch.uint8 channels_last bilinear (520, 520) -> (32, 32) aa=True        |                        |        126.316 (+-1.058)        |          134.619 (+-2.274)           |      1.066 (+-0.000)    
      4 torch.uint8 channels_last bilinear (520, 520) -> (32, 32) aa=False       |                        |         56.674 (+-0.167)        |           55.222 (+-0.322)           |      0.974 (+-0.000)    
      4 torch.uint8 channels_last bilinear (520, 520) -> (224, 224) aa=True      |                        |        286.746 (+-1.981)        |          328.410 (+-3.177)           |      1.145 (+-0.000)    
      4 torch.uint8 channels_last bilinear (520, 520) -> (224, 224) aa=False     |                        |        214.245 (+-1.590)        |          200.096 (+-1.838)           |      0.934 (+-0.000)    
      4 torch.uint8 channels_last bilinear (712, 712) -> (32, 32) aa=True        |                        |        196.027 (+-1.477)        |          211.621 (+-2.260)           |      1.080 (+-0.000)    
      4 torch.uint8 channels_last bilinear (712, 712) -> (32, 32) aa=False       |                        |         74.214 (+-0.285)        |           72.102 (+-0.259)           |      0.972 (+-0.000)    
      4 torch.uint8 channels_last bilinear (712, 712) -> (224, 224) aa=True      |                        |        422.793 (+-2.097)        |          463.809 (+-2.832)           |      1.097 (+-0.000)    
      4 torch.uint8 channels_last bilinear (712, 712) -> (224, 224) aa=False     |                        |        273.864 (+-1.839)        |          254.029 (+-2.062)           |      0.928 (+-0.000)    
      4 torch.uint8 channels_last bilinear (64, 64) -> (224, 224) aa=True        |                        |         77.143 (+-0.302)        |           75.474 (+-0.444)           |      0.978 (+-0.000)    
      4 torch.uint8 channels_last bilinear (224, 224) -> (270, 268) aa=True      |                        |        154.489 (+-1.553)        |          145.627 (+-1.644)           |      0.943 (+-0.000)    
      4 torch.uint8 channels_last bilinear (256, 256) -> (1024, 1024) aa=True    |                        |        918.553 (+-4.831)        |          871.563 (+-4.312)           |      0.949 (+-0.000)    
      4 torch.uint8 channels_last bilinear (224, 224) -> (64, 64) aa=True        |                        |         68.217 (+-0.334)        |           71.459 (+-0.313)           |      1.048 (+-0.000)    
      4 torch.uint8 channels_last bilinear (270, 268) -> (224, 224) aa=True      |                        |        154.942 (+-1.146)        |          152.151 (+-1.813)           |      0.982 (+-0.000)    
      4 torch.uint8 channels_last bilinear (1024, 1024) -> (256, 256) aa=True    |                        |        667.957 (+-3.795)        |          707.206 (+-4.806)           |      1.059 (+-0.000)    
      4 torch.uint8 channels_last bilinear (64, 64) -> (224, 224) aa=False       |                        |         75.647 (+-0.368)        |           73.886 (+-0.608)           |      0.977 (+-0.000)    
      4 torch.uint8 channels_last bilinear (224, 224) -> (270, 268) aa=False     |                        |        151.773 (+-1.835)        |          144.312 (+-2.003)           |      0.951 (+-0.000)    
      4 torch.uint8 channels_last bilinear (256, 256) -> (1024, 1024) aa=False   |                        |        910.488 (+-4.692)        |          868.346 (+-7.407)           |      0.954 (+-0.000)    
      4 torch.uint8 channels_last bilinear (224, 224) -> (64, 64) aa=False       |                        |         49.094 (+-0.349)        |           46.541 (+-0.365)           |      0.948 (+-0.000)    
      4 torch.uint8 channels_last bilinear (270, 268) -> (224, 224) aa=False     |                        |        138.071 (+-1.345)        |          130.285 (+-1.068)           |      0.944 (+-0.000)    
      4 torch.uint8 channels_last bilinear (1024, 1024) -> (256, 256) aa=False   |                        |        425.524 (+-2.772)        |          391.126 (+-4.645)           |      0.919 (+-0.000)    
      3 torch.uint8 channels_first bilinear (256, 256) -> (32, 32) aa=True       |    38.743 (+-0.536)    |        129.997 (+-1.286)        |          133.604 (+-1.105)           |      1.028 (+-0.000)    
      3 torch.uint8 channels_first bilinear (256, 256) -> (32, 32) aa=False      |                        |        112.714 (+-0.982)        |          112.761 (+-1.316)           |      1.000 (+-0.000)    
      3 torch.uint8 channels_first bilinear (256, 256) -> (224, 224) aa=True     |   128.337 (+-1.344)    |        296.472 (+-1.524)        |          319.583 (+-8.983)           |      1.078 (+-0.000)    
      3 torch.uint8 channels_first bilinear (256, 256) -> (224, 224) aa=False    |                        |        284.393 (+-2.036)        |          299.492 (+-6.821)           |      1.053 (+-0.000)    
      3 torch.uint8 channels_first bilinear (256, 256) -> (320, 320) aa=True     |   179.598 (+-6.878)    |        431.908 (+-2.406)        |          473.004 (+-21.422)          |      1.095 (+-0.000)    
      3 torch.uint8 channels_first bilinear (256, 256) -> (320, 320) aa=False    |                        |        430.534 (+-1.984)        |          472.150 (+-11.553)          |      1.097 (+-0.000)    
      3 torch.uint8 channels_first bilinear (520, 520) -> (32, 32) aa=True       |   113.142 (+-1.166)    |        435.256 (+-1.966)        |          442.585 (+-2.833)           |      1.017 (+-0.000)    
      3 torch.uint8 channels_first bilinear (520, 520) -> (32, 32) aa=False      |                        |        366.183 (+-2.257)        |          365.845 (+-2.525)           |      0.999 (+-0.000)    
      3 torch.uint8 channels_first bilinear (520, 520) -> (224, 224) aa=True     |   281.153 (+-2.692)    |        671.467 (+-3.013)        |          730.902 (+-4.980)           |      1.089 (+-0.000)    
      3 torch.uint8 channels_first bilinear (520, 520) -> (224, 224) aa=False    |                        |        598.342 (+-1.861)        |          603.011 (+-5.821)           |      1.008 (+-0.000)    
      3 torch.uint8 channels_first bilinear (712, 712) -> (32, 32) aa=True       |   185.719 (+-1.482)    |        773.958 (+-2.505)        |          787.653 (+-4.355)           |      1.018 (+-0.000)    
      3 torch.uint8 channels_first bilinear (712, 712) -> (32, 32) aa=False      |                        |        652.422 (+-2.559)        |          651.387 (+-4.101)           |      0.998 (+-0.000)    
      3 torch.uint8 channels_first bilinear (712, 712) -> (224, 224) aa=True     |   410.564 (+-2.562)    |       1071.961 (+-10.358)       |         1135.825 (+-18.089)          |      1.060 (+-0.000)    
      3 torch.uint8 channels_first bilinear (712, 712) -> (224, 224) aa=False    |                        |        927.200 (+-3.157)        |          924.521 (+-6.439)           |      0.997 (+-0.000)    
      3 torch.uint8 channels_first bilinear (64, 64) -> (224, 224) aa=True       |    60.680 (+-0.317)    |        160.523 (+-1.206)        |          176.417 (+-4.513)           |      1.099 (+-0.000)    
      3 torch.uint8 channels_first bilinear (224, 224) -> (270, 268) aa=True     |   132.675 (+-1.355)    |        322.550 (+-2.617)        |          342.063 (+-7.450)           |      1.060 (+-0.000)    
      3 torch.uint8 channels_first bilinear (256, 256) -> (1024, 1024) aa=True   |   947.072 (+-3.541)    |       2620.586 (+-20.543)       |         2801.026 (+-151.881)         |      1.069 (+-0.000)    
      3 torch.uint8 channels_first bilinear (224, 224) -> (64, 64) aa=True       |    52.027 (+-0.547)    |        132.895 (+-1.181)        |          135.916 (+-1.537)           |      1.023 (+-0.000)    
      3 torch.uint8 channels_first bilinear (270, 268) -> (224, 224) aa=True     |   138.496 (+-1.206)    |        315.960 (+-2.362)        |          336.857 (+-8.444)           |      1.066 (+-0.000)    
      3 torch.uint8 channels_first bilinear (1024, 1024) -> (256, 256) aa=True   |   697.292 (+-2.985)    |       1982.663 (+-19.293)       |         2046.552 (+-20.762)          |      1.032 (+-0.000)    
      3 torch.uint8 channels_first bilinear (64, 64) -> (224, 224) aa=False      |                        |        158.532 (+-1.565)        |          179.128 (+-5.313)           |      1.130 (+-0.000)    
      3 torch.uint8 channels_first bilinear (224, 224) -> (270, 268) aa=False    |                        |        321.899 (+-2.345)        |          337.444 (+-6.771)           |      1.048 (+-0.000)    
      3 torch.uint8 channels_first bilinear (256, 256) -> (1024, 1024) aa=False  |                        |       2630.426 (+-16.696)       |         2848.585 (+-165.408)         |      1.083 (+-0.000)    
      3 torch.uint8 channels_first bilinear (224, 224) -> (64, 64) aa=False      |                        |        112.903 (+-1.431)        |          113.394 (+-2.022)           |      1.004 (+-0.000)    
      3 torch.uint8 channels_first bilinear (270, 268) -> (224, 224) aa=False    |                        |        298.212 (+-1.772)        |          342.987 (+-14.569)          |      1.150 (+-0.000)    
      3 torch.uint8 channels_first bilinear (1024, 1024) -> (256, 256) aa=False  |                        |       1736.509 (+-18.054)       |         1965.949 (+-91.831)          |      1.132 (+-0.000)    
      4 torch.uint8 channels_first bilinear (256, 256) -> (32, 32) aa=True       |                        |         72.014 (+-0.253)        |           82.381 (+-2.008)           |      1.144 (+-0.000)    
      4 torch.uint8 channels_first bilinear (256, 256) -> (32, 32) aa=False      |                        |         53.951 (+-0.232)        |           60.169 (+-1.717)           |      1.115 (+-0.000)    
      4 torch.uint8 channels_first bilinear (256, 256) -> (224, 224) aa=True     |                        |        263.308 (+-1.422)        |          310.981 (+-12.270)          |      1.181 (+-0.000)    
      4 torch.uint8 channels_first bilinear (256, 256) -> (224, 224) aa=False    |                        |        251.195 (+-1.910)        |          288.923 (+-11.073)          |      1.150 (+-0.000)    
      4 torch.uint8 channels_first bilinear (256, 256) -> (320, 320) aa=True     |                        |        429.647 (+-2.033)        |          502.568 (+-14.300)          |      1.170 (+-0.000)    
      4 torch.uint8 channels_first bilinear (256, 256) -> (320, 320) aa=False    |                        |        425.057 (+-2.512)        |          499.103 (+-20.046)          |      1.174 (+-0.000)    
      4 torch.uint8 channels_first bilinear (520, 520) -> (32, 32) aa=True       |                        |        194.436 (+-1.339)        |          237.446 (+-9.721)           |      1.221 (+-0.000)    
      4 torch.uint8 channels_first bilinear (520, 520) -> (32, 32) aa=False      |                        |        125.226 (+-0.769)        |          145.077 (+-6.869)           |      1.159 (+-0.000)    
      4 torch.uint8 channels_first bilinear (520, 520) -> (224, 224) aa=True     |                        |        456.368 (+-2.147)        |          611.324 (+-22.827)          |      1.340 (+-0.000)    
      4 torch.uint8 channels_first bilinear (520, 520) -> (224, 224) aa=False    |                        |        383.453 (+-2.330)        |          459.649 (+-20.135)          |      1.199 (+-0.000)    
      4 torch.uint8 channels_first bilinear (712, 712) -> (32, 32) aa=True       |                        |        319.408 (+-1.498)        |          422.896 (+-21.371)          |      1.324 (+-0.000)    
      4 torch.uint8 channels_first bilinear (712, 712) -> (32, 32) aa=False      |                        |        199.822 (+-1.182)        |          242.089 (+-11.014)          |      1.212 (+-0.000)    
      4 torch.uint8 channels_first bilinear (712, 712) -> (224, 224) aa=True     |                        |        654.798 (+-2.749)        |          958.660 (+-56.497)          |      1.464 (+-0.000)    
      4 torch.uint8 channels_first bilinear (712, 712) -> (224, 224) aa=False    |                        |        506.319 (+-2.343)        |          677.322 (+-39.818)          |      1.338 (+-0.000)    
      4 torch.uint8 channels_first bilinear (64, 64) -> (224, 224) aa=True       |                        |        182.060 (+-1.707)        |          246.961 (+-17.434)          |      1.356 (+-0.000)    
      4 torch.uint8 channels_first bilinear (224, 224) -> (270, 268) aa=True     |                        |        315.260 (+-2.098)        |          425.588 (+-20.522)          |      1.350 (+-0.000)    
      4 torch.uint8 channels_first bilinear (256, 256) -> (1024, 1024) aa=True   |                        |       3194.462 (+-21.972)       |         4466.712 (+-216.667)         |      1.398 (+-0.000)    
      4 torch.uint8 channels_first bilinear (224, 224) -> (64, 64) aa=True       |                        |         89.350 (+-0.400)        |          132.108 (+-12.481)          |      1.479 (+-0.000)    
      4 torch.uint8 channels_first bilinear (270, 268) -> (224, 224) aa=True     |                        |        276.040 (+-2.108)        |          392.639 (+-27.002)          |      1.422 (+-0.000)    
      4 torch.uint8 channels_first bilinear (1024, 1024) -> (256, 256) aa=True   |                        |       1232.263 (+-68.984)       |         2098.807 (+-161.057)         |      1.703 (+-0.000)    
      4 torch.uint8 channels_first bilinear (64, 64) -> (224, 224) aa=False      |                        |        180.369 (+-1.479)        |          264.726 (+-21.965)          |      1.468 (+-0.000)    
      4 torch.uint8 channels_first bilinear (224, 224) -> (270, 268) aa=False    |                        |        312.997 (+-2.790)        |          452.938 (+-31.511)          |      1.447 (+-0.000)    
      4 torch.uint8 channels_first bilinear (256, 256) -> (1024, 1024) aa=False  |                        |       3192.712 (+-28.040)       |         4682.081 (+-300.886)         |      1.466 (+-0.000)    
      4 torch.uint8 channels_first bilinear (224, 224) -> (64, 64) aa=False      |                        |         70.236 (+-0.357)        |          108.193 (+-9.565)           |      1.540 (+-0.000)    
      4 torch.uint8 channels_first bilinear (270, 268) -> (224, 224) aa=False    |                        |        258.770 (+-1.598)        |          386.781 (+-26.346)          |      1.495 (+-0.000)    
      4 torch.uint8 channels_first bilinear (1024, 1024) -> (256, 256) aa=False  |                        |        920.269 (+-4.392)        |         1724.653 (+-108.171)         |      1.874 (+-0.000)    

Times are in microseconds (us).