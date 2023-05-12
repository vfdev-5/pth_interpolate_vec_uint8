Description:
- 20230512-111221-pr-custom-cases
Torch version: 2.1.0a0+git34b9937
Torch config: PyTorch built with:
  - GCC 9.4
  - C++ Version: 201703
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - CPU capability usage: AVX2
  - CUDA Runtime 11.7
  - NVCC architecture flags: -gencode;arch=compute_61,code=sm_61
  - CuDNN 8.5
  - Build settings: BUILD_TYPE=Release, CXX_COMPILER=/usr/lib/ccache/c++, CXX_FLAGS= -D_GLIBCXX_USE_CXX11_ABI=1 -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -DNDEBUG -DUSE_KINETO -DLIBKINETO_NOCUPTI -DLIBKINETO_NOROCTRACER -DUSE_PYTORCH_QNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -O2 -fPIC -Wall -Wextra -Werror=return-type -Werror=non-virtual-dtor -Werror=bool-operation -Wnarrowing -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-unused-parameter -Wno-unused-function -Wno-unused-result -Wno-strict-overflow -Wno-strict-aliasing -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=old-style-cast -Wno-invalid-partial-specialization -Wno-unused-private-field -Wno-aligned-allocation-unavailable -Wno-missing-braces -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Werror=cast-function-type -Wno-stringop-overflow, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, TORCH_DISABLE_GPU_ASSERTS=ON, TORCH_VERSION=2.1.0, USE_CUDA=0, USE_CUDNN=OFF, USE_EIGEN_FOR_BLAS=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=OFF, USE_MKLDNN=0, USE_MPI=OFF, USE_NCCL=OFF, USE_NNPACK=0, USE_OPENMP=ON, USE_ROCM=OFF,


- 20230512-113728-nightly
Torch version: 2.1.0a0+git5a933d0
Torch config: PyTorch built with:
  - GCC 9.4
  - C++ Version: 201703
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - CPU capability usage: AVX2
  - Build settings: BUILD_TYPE=Release, CXX_COMPILER=/usr/lib/ccache/c++, CXX_FLAGS= -D_GLIBCXX_USE_CXX11_ABI=1 -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -DNDEBUG -DUSE_KINETO -DLIBKINETO_NOCUPTI -DLIBKINETO_NOROCTRACER -DUSE_PYTORCH_QNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -O2 -fPIC -Wall -Wextra -Werror=return-type -Werror=non-virtual-dtor -Werror=bool-operation -Wnarrowing -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-unused-parameter -Wno-unused-function -Wno-unused-result -Wno-strict-overflow -Wno-strict-aliasing -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=old-style-cast -Wno-invalid-partial-specialization -Wno-unused-private-field -Wno-aligned-allocation-unavailable -Wno-missing-braces -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Werror=cast-function-type -Wno-stringop-overflow, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, TORCH_DISABLE_GPU_ASSERTS=ON, TORCH_VERSION=2.1.0, USE_CUDA=0, USE_CUDNN=OFF, USE_EIGEN_FOR_BLAS=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=OFF, USE_MKLDNN=0, USE_MPI=OFF, USE_NCCL=OFF, USE_NNPACK=0, USE_OPENMP=ON, USE_ROCM=OFF,



[------------------------------------------------------------------------------------------------ Resize ------------------------------------------------------------------------------------------------]
                                                                                 |  Pillow (9.0.0.post1)  |  torch (2.1.0a0+git34b9937) PR  |  torch (2.1.0a0+git5a933d0) nightly  |   Ratio PR vs Nightly
1 threads: -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      3 torch.uint8 channels_last bilinear (256, 256) -> (32, 32) aa=True        |    38.589 (+-0.418)    |         56.349 (+-0.291)        |           59.263 (+-0.367)           |    1.052 (+-0.000)
      3 torch.uint8 channels_last bilinear (256, 256) -> (32, 32) aa=False       |                        |         35.911 (+-0.232)        |           39.116 (+-0.301)           |    1.089 (+-0.000)
      3 torch.uint8 channels_last bilinear (256, 256) -> (224, 224) aa=True      |   127.763 (+-1.359)    |        160.356 (+-1.372)        |          160.621 (+-1.688)           |    1.002 (+-0.000)
      3 torch.uint8 channels_last bilinear (256, 256) -> (224, 224) aa=False     |                        |        148.063 (+-1.206)        |          150.477 (+-1.865)           |    1.016 (+-0.000)
      3 torch.uint8 channels_last bilinear (256, 256) -> (320, 320) aa=True      |   179.124 (+-1.617)    |        222.014 (+-1.720)        |          221.234 (+-1.802)           |    0.996 (+-0.000)
      3 torch.uint8 channels_last bilinear (256, 256) -> (320, 320) aa=False     |                        |        217.735 (+-1.806)        |          220.391 (+-1.657)           |    1.012 (+-0.000)
      3 torch.uint8 channels_last bilinear (520, 520) -> (32, 32) aa=True        |   113.177 (+-1.132)    |        131.578 (+-1.836)        |          134.197 (+-1.389)           |    1.020 (+-0.000)
      3 torch.uint8 channels_last bilinear (520, 520) -> (32, 32) aa=False       |                        |         56.474 (+-1.089)        |           60.503 (+-1.149)           |    1.071 (+-0.000)
      3 torch.uint8 channels_last bilinear (520, 520) -> (224, 224) aa=True      |   280.858 (+-2.289)    |        333.084 (+-2.390)        |          343.019 (+-2.193)           |    1.030 (+-0.000)
      3 torch.uint8 channels_last bilinear (520, 520) -> (224, 224) aa=False     |                        |        245.592 (+-4.021)        |          252.354 (+-1.737)           |    1.028 (+-0.000)
      3 torch.uint8 channels_last bilinear (712, 712) -> (32, 32) aa=True        |   185.420 (+-1.856)    |        206.781 (+-3.183)        |          208.361 (+-2.477)           |    1.008 (+-0.000)
      3 torch.uint8 channels_last bilinear (712, 712) -> (32, 32) aa=False       |                        |         75.320 (+-0.303)        |           79.738 (+-0.393)           |    1.059 (+-0.000)
      3 torch.uint8 channels_last bilinear (712, 712) -> (224, 224) aa=True      |   407.314 (+-2.600)    |        463.019 (+-2.586)        |          475.129 (+-3.023)           |    1.026 (+-0.000)
      3 torch.uint8 channels_last bilinear (712, 712) -> (224, 224) aa=False     |                        |        318.959 (+-2.455)        |          332.983 (+-2.337)           |    1.044 (+-0.000)
      3 torch.uint8 channels_last bilinear (64, 64) -> (224, 224) aa=True        |    60.350 (+-0.312)    |         78.235 (+-0.518)        |           80.420 (+-0.488)           |    1.028 (+-0.000)
      3 torch.uint8 channels_last bilinear (224, 224) -> (270, 268) aa=True      |   132.794 (+-1.502)    |        168.851 (+-1.654)        |          169.022 (+-1.521)           |    1.001 (+-0.000)
      3 torch.uint8 channels_last bilinear (256, 256) -> (1024, 1024) aa=True    |   951.273 (+-6.093)    |        922.144 (+-12.406)       |          929.452 (+-5.349)           |    1.008 (+-0.000)
      3 torch.uint8 channels_last bilinear (224, 224) -> (64, 64) aa=True        |    52.297 (+-0.451)    |         70.354 (+-0.463)        |           72.933 (+-0.473)           |    1.037 (+-0.000)
      3 torch.uint8 channels_last bilinear (270, 268) -> (224, 224) aa=True      |   138.703 (+-1.476)    |        173.169 (+-1.743)        |          174.470 (+-1.648)           |    1.008 (+-0.000)
      3 torch.uint8 channels_last bilinear (1024, 1024) -> (256, 256) aa=True    |   686.549 (+-4.609)    |        790.586 (+-4.475)        |          799.185 (+-4.120)           |    1.011 (+-0.000)
      3 torch.uint8 channels_last bilinear (64, 64) -> (224, 224) aa=False       |                        |         76.816 (+-0.516)        |           79.206 (+-0.449)           |    1.031 (+-0.000)
      3 torch.uint8 channels_last bilinear (224, 224) -> (270, 268) aa=False     |                        |        166.117 (+-1.773)        |          167.266 (+-1.574)           |    1.007 (+-0.000)
      3 torch.uint8 channels_last bilinear (256, 256) -> (1024, 1024) aa=False   |                        |        921.701 (+-5.172)        |          911.224 (+-4.563)           |    0.989 (+-0.000)
      3 torch.uint8 channels_last bilinear (224, 224) -> (64, 64) aa=False       |                        |         52.369 (+-0.253)        |           55.004 (+-0.281)           |    1.050 (+-0.000)
      3 torch.uint8 channels_last bilinear (270, 268) -> (224, 224) aa=False     |                        |        154.574 (+-1.276)        |          157.365 (+-1.387)           |    1.018 (+-0.000)
      3 torch.uint8 channels_last bilinear (1024, 1024) -> (256, 256) aa=False   |                        |        504.248 (+-2.860)        |          514.742 (+-2.953)           |    1.021 (+-0.000)
      4 torch.uint8 channels_last bilinear (256, 256) -> (32, 32) aa=True        |                        |         50.977 (+-0.424)        |           53.730 (+-0.327)           |    1.054 (+-0.000)
      4 torch.uint8 channels_last bilinear (256, 256) -> (32, 32) aa=False       |                        |         34.035 (+-0.233)        |           35.727 (+-0.214)           |    1.050 (+-0.000)
      4 torch.uint8 channels_last bilinear (256, 256) -> (224, 224) aa=True      |                        |        143.296 (+-3.653)        |          144.514 (+-3.717)           |    1.008 (+-0.000)
      4 torch.uint8 channels_last bilinear (256, 256) -> (224, 224) aa=False     |                        |        129.871 (+-1.173)        |          132.511 (+-1.074)           |    1.020 (+-0.000)
      4 torch.uint8 channels_last bilinear (256, 256) -> (320, 320) aa=True      |                        |        195.042 (+-1.738)        |          195.944 (+-1.508)           |    1.005 (+-0.000)
      4 torch.uint8 channels_last bilinear (256, 256) -> (320, 320) aa=False     |                        |        192.720 (+-1.484)        |          193.987 (+-1.456)           |    1.007 (+-0.000)
      4 torch.uint8 channels_last bilinear (520, 520) -> (32, 32) aa=True        |                        |        124.796 (+-1.789)        |          126.201 (+-1.249)           |    1.011 (+-0.000)
      4 torch.uint8 channels_last bilinear (520, 520) -> (32, 32) aa=False       |                        |         55.017 (+-0.333)        |           56.744 (+-0.326)           |    1.031 (+-0.000)
      4 torch.uint8 channels_last bilinear (520, 520) -> (224, 224) aa=True      |                        |        284.556 (+-2.414)        |          284.632 (+-2.139)           |    1.000 (+-0.000)
      4 torch.uint8 channels_last bilinear (520, 520) -> (224, 224) aa=False     |                        |        211.785 (+-1.665)        |          214.173 (+-1.795)           |    1.011 (+-0.000)
      4 torch.uint8 channels_last bilinear (712, 712) -> (32, 32) aa=True        |                        |        194.718 (+-1.955)        |          199.366 (+-1.836)           |    1.024 (+-0.000)
      4 torch.uint8 channels_last bilinear (712, 712) -> (32, 32) aa=False       |                        |         72.417 (+-0.269)        |           74.401 (+-0.310)           |    1.027 (+-0.000)
      4 torch.uint8 channels_last bilinear (712, 712) -> (224, 224) aa=True      |                        |        421.183 (+-3.402)        |          422.855 (+-3.890)           |    1.004 (+-0.000)
      4 torch.uint8 channels_last bilinear (712, 712) -> (224, 224) aa=False     |                        |        271.536 (+-2.119)        |          272.966 (+-2.087)           |    1.005 (+-0.000)
      4 torch.uint8 channels_last bilinear (64, 64) -> (224, 224) aa=True        |                        |         76.416 (+-0.377)        |           77.384 (+-0.535)           |    1.013 (+-0.000)
      4 torch.uint8 channels_last bilinear (224, 224) -> (270, 268) aa=True      |                        |        150.794 (+-1.088)        |          152.439 (+-1.350)           |    1.011 (+-0.000)
      4 torch.uint8 channels_last bilinear (256, 256) -> (1024, 1024) aa=True    |                        |        901.858 (+-4.775)        |          904.248 (+-4.676)           |    1.003 (+-0.000)
      4 torch.uint8 channels_last bilinear (224, 224) -> (64, 64) aa=True        |                        |         66.494 (+-0.316)        |           68.214 (+-0.346)           |    1.026 (+-0.000)
      4 torch.uint8 channels_last bilinear (270, 268) -> (224, 224) aa=True      |                        |        154.071 (+-1.935)        |          154.883 (+-1.471)           |    1.005 (+-0.000)
      4 torch.uint8 channels_last bilinear (1024, 1024) -> (256, 256) aa=True    |                        |        659.168 (+-3.579)        |          664.032 (+-3.436)           |    1.007 (+-0.000)
      4 torch.uint8 channels_last bilinear (64, 64) -> (224, 224) aa=False       |                        |         74.414 (+-0.380)        |           75.318 (+-0.525)           |    1.012 (+-0.000)
      4 torch.uint8 channels_last bilinear (224, 224) -> (270, 268) aa=False     |                        |        149.470 (+-1.235)        |          150.153 (+-1.298)           |    1.005 (+-0.000)
      4 torch.uint8 channels_last bilinear (256, 256) -> (1024, 1024) aa=False   |                        |        895.427 (+-4.726)        |          906.816 (+-4.077)           |    1.013 (+-0.000)
      4 torch.uint8 channels_last bilinear (224, 224) -> (64, 64) aa=False       |                        |         48.237 (+-0.350)        |           49.417 (+-0.281)           |    1.024 (+-0.000)
      4 torch.uint8 channels_last bilinear (270, 268) -> (224, 224) aa=False     |                        |        136.212 (+-1.515)        |          137.329 (+-1.608)           |    1.008 (+-0.000)
      4 torch.uint8 channels_last bilinear (1024, 1024) -> (256, 256) aa=False   |                        |        420.455 (+-3.340)        |          421.594 (+-2.311)           |    1.003 (+-0.000)
      3 torch.uint8 channels_first bilinear (256, 256) -> (32, 32) aa=True       |    38.656 (+-0.339)    |        128.539 (+-1.330)        |          130.639 (+-2.058)           |    1.016 (+-0.000)
      3 torch.uint8 channels_first bilinear (256, 256) -> (32, 32) aa=False      |                        |        111.092 (+-1.053)        |          112.630 (+-1.679)           |    1.014 (+-0.000)
      3 torch.uint8 channels_first bilinear (256, 256) -> (224, 224) aa=True     |   127.885 (+-1.128)    |        295.764 (+-3.432)        |          298.171 (+-2.627)           |    1.008 (+-0.000)
      3 torch.uint8 channels_first bilinear (256, 256) -> (224, 224) aa=False    |                        |        283.122 (+-1.695)        |          285.265 (+-3.432)           |    1.008 (+-0.000)
      3 torch.uint8 channels_first bilinear (256, 256) -> (320, 320) aa=True     |   179.294 (+-4.849)    |        424.563 (+-2.258)        |          429.831 (+-4.090)           |    1.012 (+-0.000)
      3 torch.uint8 channels_first bilinear (256, 256) -> (320, 320) aa=False    |                        |        424.353 (+-2.408)        |          427.383 (+-5.505)           |    1.007 (+-0.000)
      3 torch.uint8 channels_first bilinear (520, 520) -> (32, 32) aa=True       |   112.903 (+-0.981)    |        433.906 (+-3.100)        |          435.512 (+-3.032)           |    1.004 (+-0.000)
      3 torch.uint8 channels_first bilinear (520, 520) -> (32, 32) aa=False      |                        |        364.496 (+-1.905)        |          365.127 (+-1.969)           |    1.002 (+-0.000)
      3 torch.uint8 channels_first bilinear (520, 520) -> (224, 224) aa=True     |   281.352 (+-1.943)    |        668.437 (+-3.347)        |          667.571 (+-3.065)           |    0.999 (+-0.000)
      3 torch.uint8 channels_first bilinear (520, 520) -> (224, 224) aa=False    |                        |        595.843 (+-3.568)        |          603.290 (+-4.234)           |    1.012 (+-0.000)
      3 torch.uint8 channels_first bilinear (712, 712) -> (32, 32) aa=True       |   185.683 (+-1.547)    |        769.756 (+-6.105)        |          775.356 (+-2.959)           |    1.007 (+-0.000)
      3 torch.uint8 channels_first bilinear (712, 712) -> (32, 32) aa=False      |                        |        650.958 (+-3.361)        |          650.517 (+-2.077)           |    0.999 (+-0.000)
      3 torch.uint8 channels_first bilinear (712, 712) -> (224, 224) aa=True     |   409.223 (+-2.599)    |       1067.830 (+-10.038)       |          1074.082 (+-9.313)          |    1.006 (+-0.000)
      3 torch.uint8 channels_first bilinear (712, 712) -> (224, 224) aa=False    |                        |        926.449 (+-4.536)        |          926.195 (+-2.308)           |    1.000 (+-0.000)
      3 torch.uint8 channels_first bilinear (64, 64) -> (224, 224) aa=True       |    60.904 (+-0.268)    |        158.114 (+-1.570)        |          164.024 (+-2.757)           |    1.037 (+-0.000)
      3 torch.uint8 channels_first bilinear (224, 224) -> (270, 268) aa=True     |   132.684 (+-1.090)    |        319.140 (+-2.758)        |          324.211 (+-2.712)           |    1.016 (+-0.000)
      3 torch.uint8 channels_first bilinear (256, 256) -> (1024, 1024) aa=True   |   947.654 (+-4.551)    |       2605.035 (+-19.450)       |         2699.202 (+-78.809)          |    1.036 (+-0.000)
      3 torch.uint8 channels_first bilinear (224, 224) -> (64, 64) aa=True       |    52.404 (+-0.350)    |        130.805 (+-1.510)        |          132.066 (+-1.250)           |    1.010 (+-0.000)
      3 torch.uint8 channels_first bilinear (270, 268) -> (224, 224) aa=True     |   138.780 (+-1.140)    |        313.989 (+-2.274)        |          316.415 (+-2.128)           |    1.008 (+-0.000)
      3 torch.uint8 channels_first bilinear (1024, 1024) -> (256, 256) aa=True   |   688.540 (+-3.295)    |       1985.035 (+-19.832)       |         1981.955 (+-15.179)          |    0.998 (+-0.000)
      3 torch.uint8 channels_first bilinear (64, 64) -> (224, 224) aa=False      |                        |        157.253 (+-1.328)        |          161.693 (+-3.620)           |    1.028 (+-0.000)
      3 torch.uint8 channels_first bilinear (224, 224) -> (270, 268) aa=False    |                        |        317.683 (+-3.044)        |          320.037 (+-2.329)           |    1.007 (+-0.000)
      3 torch.uint8 channels_first bilinear (256, 256) -> (1024, 1024) aa=False  |                        |       2592.141 (+-24.378)       |         2651.740 (+-73.020)          |    1.023 (+-0.000)
      3 torch.uint8 channels_first bilinear (224, 224) -> (64, 64) aa=False      |                        |        112.432 (+-3.458)        |          113.297 (+-3.220)           |    1.008 (+-0.000)
      3 torch.uint8 channels_first bilinear (270, 268) -> (224, 224) aa=False    |                        |        296.546 (+-2.726)        |          297.417 (+-1.984)           |    1.003 (+-0.000)
      3 torch.uint8 channels_first bilinear (1024, 1024) -> (256, 256) aa=False  |                        |       1746.050 (+-18.607)       |         1732.480 (+-11.795)          |    0.992 (+-0.000)
      4 torch.uint8 channels_first bilinear (256, 256) -> (32, 32) aa=True       |                        |         69.539 (+-0.267)        |           72.113 (+-0.218)           |    1.037 (+-0.000)
      4 torch.uint8 channels_first bilinear (256, 256) -> (32, 32) aa=False      |                        |         52.555 (+-0.442)        |           54.232 (+-0.209)           |    1.032 (+-0.000)
      4 torch.uint8 channels_first bilinear (256, 256) -> (224, 224) aa=True     |                        |        249.478 (+-1.482)        |          251.820 (+-1.683)           |    1.009 (+-0.000)
      4 torch.uint8 channels_first bilinear (256, 256) -> (224, 224) aa=False    |                        |        237.123 (+-1.471)        |          239.649 (+-1.566)           |    1.011 (+-0.000)
      4 torch.uint8 channels_first bilinear (256, 256) -> (320, 320) aa=True     |                        |        395.810 (+-2.731)        |          403.046 (+-3.443)           |    1.018 (+-0.000)
      4 torch.uint8 channels_first bilinear (256, 256) -> (320, 320) aa=False    |                        |        394.395 (+-2.894)        |          397.330 (+-6.638)           |    1.007 (+-0.000)
      4 torch.uint8 channels_first bilinear (520, 520) -> (32, 32) aa=True       |                        |        190.018 (+-1.513)        |          195.124 (+-1.318)           |    1.027 (+-0.000)
      4 torch.uint8 channels_first bilinear (520, 520) -> (32, 32) aa=False      |                        |        123.088 (+-0.966)        |          124.789 (+-0.803)           |    1.014 (+-0.000)
      4 torch.uint8 channels_first bilinear (520, 520) -> (224, 224) aa=True     |                        |        440.479 (+-2.668)        |          442.443 (+-2.497)           |    1.004 (+-0.000)
      4 torch.uint8 channels_first bilinear (520, 520) -> (224, 224) aa=False    |                        |        368.226 (+-2.666)        |          371.279 (+-2.509)           |    1.008 (+-0.000)
      4 torch.uint8 channels_first bilinear (712, 712) -> (32, 32) aa=True       |                        |        319.776 (+-2.506)        |          326.666 (+-1.988)           |    1.022 (+-0.000)
      4 torch.uint8 channels_first bilinear (712, 712) -> (32, 32) aa=False      |                        |        198.271 (+-1.259)        |          199.871 (+-1.087)           |    1.008 (+-0.000)
      4 torch.uint8 channels_first bilinear (712, 712) -> (224, 224) aa=True     |                        |        635.960 (+-3.177)        |          639.415 (+-3.285)           |    1.005 (+-0.000)
      4 torch.uint8 channels_first bilinear (712, 712) -> (224, 224) aa=False    |                        |        487.434 (+-3.889)        |          489.071 (+-3.623)           |    1.003 (+-0.000)
      4 torch.uint8 channels_first bilinear (64, 64) -> (224, 224) aa=True       |                        |        168.073 (+-1.494)        |          171.692 (+-1.802)           |    1.022 (+-0.000)
      4 torch.uint8 channels_first bilinear (224, 224) -> (270, 268) aa=True     |                        |        294.307 (+-3.311)        |          296.748 (+-2.385)           |    1.008 (+-0.000)
      4 torch.uint8 channels_first bilinear (256, 256) -> (1024, 1024) aa=True   |                        |       2939.508 (+-28.157)       |         2990.687 (+-93.632)          |    1.017 (+-0.000)
      4 torch.uint8 channels_first bilinear (224, 224) -> (64, 64) aa=True       |                        |         86.968 (+-0.509)        |           88.210 (+-0.225)           |    1.014 (+-0.000)
      4 torch.uint8 channels_first bilinear (270, 268) -> (224, 224) aa=True     |                        |        261.816 (+-2.841)        |          265.063 (+-2.127)           |    1.012 (+-0.000)
      4 torch.uint8 channels_first bilinear (1024, 1024) -> (256, 256) aa=True   |                        |       1217.327 (+-34.133)       |         1234.903 (+-30.407)          |    1.014 (+-0.000)
      4 torch.uint8 channels_first bilinear (64, 64) -> (224, 224) aa=False      |                        |        166.244 (+-1.462)        |          170.141 (+-1.672)           |    1.023 (+-0.000)
      4 torch.uint8 channels_first bilinear (224, 224) -> (270, 268) aa=False    |                        |        291.790 (+-2.356)        |          293.382 (+-2.198)           |    1.005 (+-0.000)
      4 torch.uint8 channels_first bilinear (256, 256) -> (1024, 1024) aa=False  |                        |       2904.426 (+-24.337)       |         2990.875 (+-93.999)          |    1.030 (+-0.000)
      4 torch.uint8 channels_first bilinear (224, 224) -> (64, 64) aa=False      |                        |         67.990 (+-0.376)        |           69.233 (+-0.182)           |    1.018 (+-0.000)
      4 torch.uint8 channels_first bilinear (270, 268) -> (224, 224) aa=False    |                        |        244.322 (+-1.754)        |          245.252 (+-1.804)           |    1.004 (+-0.000)
      4 torch.uint8 channels_first bilinear (1024, 1024) -> (256, 256) aa=False  |                        |        983.071 (+-34.580)       |         1026.373 (+-45.300)          |    1.044 (+-0.000)

Times are in microseconds (us).
