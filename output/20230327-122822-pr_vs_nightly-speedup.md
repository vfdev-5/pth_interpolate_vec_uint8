Description:
- 20230327-114043-pr
Torch version: 2.1.0a0+git90861e5
Torch config: PyTorch built with:
  - GCC 9.4
  - C++ Version: 201703
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - CPU capability usage: AVX2
  - Build settings: BUILD_TYPE=Release, CXX_COMPILER=/usr/bin/c++, CXX_FLAGS= -D_GLIBCXX_USE_CXX11_ABI=1 -Wno-deprecated -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -DNDEBUG -DUSE_KINETO -DLIBKINETO_NOCUPTI -DLIBKINETO_NOROCTRACER -DUSE_PYTORCH_QNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -O2 -fPIC -Wall -Wextra -Werror=return-type -Werror=non-virtual-dtor -Werror=bool-operation -Wnarrowing -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-unused-parameter -Wno-unused-function -Wno-unused-result -Wno-strict-overflow -Wno-strict-aliasing -Wno-error=deprecated-declarations -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=old-style-cast -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Werror=cast-function-type -Wno-stringop-overflow, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, TORCH_DISABLE_GPU_ASSERTS=ON, TORCH_VERSION=2.1.0, USE_CUDA=0, USE_CUDNN=OFF, USE_EIGEN_FOR_BLAS=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=OFF, USE_MKLDNN=0, USE_MPI=OFF, USE_NCCL=OFF, USE_NNPACK=0, USE_OPENMP=ON, USE_ROCM=OFF, 


- 20230327-111746-nightly
Torch version: 2.1.0a0+git2b75955
Torch config: PyTorch built with:
  - GCC 9.4
  - C++ Version: 201703
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - CPU capability usage: AVX2
  - Build settings: BUILD_TYPE=Release, CXX_COMPILER=/usr/bin/c++, CXX_FLAGS= -D_GLIBCXX_USE_CXX11_ABI=1 -Wno-deprecated -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -DNDEBUG -DUSE_KINETO -DLIBKINETO_NOCUPTI -DLIBKINETO_NOROCTRACER -DUSE_PYTORCH_QNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -O2 -fPIC -Wall -Wextra -Werror=return-type -Werror=non-virtual-dtor -Werror=bool-operation -Wnarrowing -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-unused-parameter -Wno-unused-function -Wno-unused-result -Wno-strict-overflow -Wno-strict-aliasing -Wno-error=deprecated-declarations -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=old-style-cast -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Werror=cast-function-type -Wno-stringop-overflow, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, TORCH_DISABLE_GPU_ASSERTS=ON, TORCH_VERSION=2.1.0, USE_CUDA=0, USE_CUDNN=OFF, USE_EIGEN_FOR_BLAS=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=OFF, USE_MKLDNN=0, USE_MPI=OFF, USE_NCCL=OFF, USE_NNPACK=0, USE_OPENMP=ON, USE_ROCM=OFF, 



[-------------------------------------------------------------------------------------------------- Resize -------------------------------------------------------------------------------------------------]
                                                                                 |  Pillow (9.0.0.post1)  |  torch (2.1.0a0+git90861e5) PR  |  torch (2.1.0a0+git2b75955) nightly  |  Speed-up: PR vs nightly
1 threads: --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      3 torch.uint8 channels_last bilinear (256, 256) -> (32, 32) aa=True        |    39.151 (+-0.365)    |         56.306 (+-0.420)        |          131.033 (+-1.448)           |      2.327 (+-0.000)    
      3 torch.uint8 channels_last bilinear (256, 256) -> (32, 32) aa=False       |                        |         37.943 (+-0.284)        |          113.911 (+-1.736)           |      3.002 (+-0.000)    
      3 torch.uint8 channels_last bilinear (256, 256) -> (224, 224) aa=True      |   129.252 (+-1.502)    |        154.751 (+-1.156)        |          299.679 (+-2.099)           |      1.937 (+-0.000)    
      3 torch.uint8 channels_last bilinear (256, 256) -> (224, 224) aa=False     |                        |        145.128 (+-1.554)        |          285.331 (+-1.919)           |      1.966 (+-0.000)    
      3 torch.uint8 channels_last bilinear (256, 256) -> (320, 320) aa=True      |   180.307 (+-2.087)    |        213.160 (+-1.333)        |          431.057 (+-3.536)           |      2.022 (+-0.000)    
      3 torch.uint8 channels_last bilinear (256, 256) -> (320, 320) aa=False     |                        |        211.575 (+-1.210)        |          429.410 (+-3.620)           |      2.030 (+-0.000)    
      3 torch.uint8 channels_last bilinear (520, 520) -> (32, 32) aa=True        |   113.838 (+-1.177)    |        128.885 (+-1.427)        |          459.610 (+-13.322)          |      3.566 (+-0.000)    
      3 torch.uint8 channels_last bilinear (520, 520) -> (32, 32) aa=False       |                        |         58.355 (+-0.215)        |          400.015 (+-11.815)          |      6.855 (+-0.000)    
      3 torch.uint8 channels_last bilinear (520, 520) -> (224, 224) aa=True      |   283.189 (+-2.671)    |        322.833 (+-2.479)        |          683.555 (+-4.466)           |      2.117 (+-0.000)    
      3 torch.uint8 channels_last bilinear (520, 520) -> (224, 224) aa=False     |                        |        238.694 (+-2.093)        |          603.545 (+-2.644)           |      2.529 (+-0.000)    
      3 torch.uint8 channels_last bilinear (712, 712) -> (32, 32) aa=True        |   187.305 (+-2.106)    |        200.055 (+-1.396)        |          860.867 (+-21.763)          |      4.303 (+-0.000)    
      3 torch.uint8 channels_last bilinear (712, 712) -> (32, 32) aa=False       |                        |         76.647 (+-0.304)        |          703.019 (+-25.805)          |      9.172 (+-0.000)    
      3 torch.uint8 channels_last bilinear (712, 712) -> (224, 224) aa=True      |   412.629 (+-4.249)    |        442.694 (+-3.566)        |         1101.673 (+-49.299)          |      2.489 (+-0.000)    
      3 torch.uint8 channels_last bilinear (712, 712) -> (224, 224) aa=False     |                        |        311.538 (+-1.594)        |          941.062 (+-5.549)           |      3.021 (+-0.000)    
      3 torch.uint8 channels_last bilinear (64, 64) -> (224, 224) aa=True        |    61.431 (+-0.805)    |         78.779 (+-0.249)        |          160.853 (+-1.506)           |      2.042 (+-0.000)    
      3 torch.uint8 channels_last bilinear (224, 224) -> (270, 268) aa=True      |   134.805 (+-2.875)    |        163.804 (+-1.285)        |          327.343 (+-2.846)           |      1.998 (+-0.000)    
      3 torch.uint8 channels_last bilinear (256, 256) -> (1024, 1024) aa=True    |   957.278 (+-10.604)   |        903.323 (+-2.566)        |         2603.360 (+-20.530)          |      2.882 (+-0.000)    
      3 torch.uint8 channels_last bilinear (224, 224) -> (64, 64) aa=True        |    52.654 (+-0.511)    |         71.405 (+-0.256)        |          131.829 (+-1.333)           |      1.846 (+-0.000)    
      3 torch.uint8 channels_last bilinear (270, 268) -> (224, 224) aa=True      |   140.364 (+-1.781)    |        167.082 (+-1.289)        |          320.063 (+-2.562)           |      1.916 (+-0.000)    
      3 torch.uint8 channels_last bilinear (1024, 1024) -> (256, 256) aa=True    |   695.635 (+-7.241)    |        751.325 (+-2.741)        |         2036.860 (+-36.109)          |      2.711 (+-0.000)    
      3 torch.uint8 channels_last bilinear (64, 64) -> (224, 224) aa=False       |                        |         77.626 (+-0.299)        |          158.479 (+-1.702)           |      2.042 (+-0.000)    
      3 torch.uint8 channels_last bilinear (224, 224) -> (270, 268) aa=False     |                        |        160.625 (+-1.694)        |          322.104 (+-2.764)           |      2.005 (+-0.000)    
      3 torch.uint8 channels_last bilinear (256, 256) -> (1024, 1024) aa=False   |                        |        889.531 (+-4.113)        |         2611.388 (+-29.917)          |      2.936 (+-0.000)    
      3 torch.uint8 channels_last bilinear (224, 224) -> (64, 64) aa=False       |                        |         53.151 (+-0.250)        |          113.869 (+-1.243)           |      2.142 (+-0.000)    
      3 torch.uint8 channels_last bilinear (270, 268) -> (224, 224) aa=False     |                        |        150.666 (+-1.272)        |          299.861 (+-2.710)           |      1.990 (+-0.000)    
      3 torch.uint8 channels_last bilinear (1024, 1024) -> (256, 256) aa=False   |                        |        477.855 (+-2.241)        |         1776.796 (+-19.660)          |      3.718 (+-0.000)    
      4 torch.uint8 channels_last bilinear (256, 256) -> (32, 32) aa=True        |                        |         52.865 (+-0.227)        |           51.626 (+-0.364)           |      0.977 (+-0.000)    
      4 torch.uint8 channels_last bilinear (256, 256) -> (32, 32) aa=False       |                        |         35.937 (+-0.186)        |           34.793 (+-0.720)           |      0.968 (+-0.000)    
      4 torch.uint8 channels_last bilinear (256, 256) -> (224, 224) aa=True      |                        |        147.889 (+-1.708)        |          145.787 (+-1.424)           |      0.986 (+-0.000)    
      4 torch.uint8 channels_last bilinear (256, 256) -> (224, 224) aa=False     |                        |        135.850 (+-0.907)        |          131.900 (+-1.170)           |      0.971 (+-0.000)    
      4 torch.uint8 channels_last bilinear (256, 256) -> (320, 320) aa=True      |                        |        204.832 (+-1.742)        |          195.768 (+-1.758)           |      0.956 (+-0.000)    
      4 torch.uint8 channels_last bilinear (256, 256) -> (320, 320) aa=False     |                        |        202.354 (+-1.521)        |          193.152 (+-1.448)           |      0.955 (+-0.000)    
      4 torch.uint8 channels_last bilinear (520, 520) -> (32, 32) aa=True        |                        |        126.433 (+-0.852)        |          124.296 (+-1.251)           |      0.983 (+-0.000)    
      4 torch.uint8 channels_last bilinear (520, 520) -> (32, 32) aa=False       |                        |         57.178 (+-0.239)        |           56.383 (+-0.298)           |      0.986 (+-0.000)    
      4 torch.uint8 channels_last bilinear (520, 520) -> (224, 224) aa=True      |                        |        285.895 (+-1.626)        |          290.807 (+-2.869)           |      1.017 (+-0.000)    
      4 torch.uint8 channels_last bilinear (520, 520) -> (224, 224) aa=False     |                        |        219.550 (+-1.408)        |          214.602 (+-1.619)           |      0.977 (+-0.000)    
      4 torch.uint8 channels_last bilinear (712, 712) -> (32, 32) aa=True        |                        |        195.556 (+-2.115)        |          197.220 (+-1.500)           |      1.009 (+-0.000)    
      4 torch.uint8 channels_last bilinear (712, 712) -> (32, 32) aa=False       |                        |         74.533 (+-0.203)        |           73.541 (+-0.226)           |      0.987 (+-0.000)    
      4 torch.uint8 channels_last bilinear (712, 712) -> (224, 224) aa=True      |                        |        422.692 (+-2.293)        |          428.137 (+-4.839)           |      1.013 (+-0.000)    
      4 torch.uint8 channels_last bilinear (712, 712) -> (224, 224) aa=False     |                        |        280.382 (+-1.602)        |          276.558 (+-1.940)           |      0.986 (+-0.000)    
      4 torch.uint8 channels_last bilinear (64, 64) -> (224, 224) aa=True        |                        |         78.632 (+-1.676)        |           75.166 (+-0.391)           |      0.956 (+-0.000)    
      4 torch.uint8 channels_last bilinear (224, 224) -> (270, 268) aa=True      |                        |        158.326 (+-1.024)        |          152.750 (+-1.371)           |      0.965 (+-0.000)    
      4 torch.uint8 channels_last bilinear (256, 256) -> (1024, 1024) aa=True    |                        |        932.036 (+-4.899)        |          906.450 (+-5.702)           |      0.973 (+-0.000)    
      4 torch.uint8 channels_last bilinear (224, 224) -> (64, 64) aa=True        |                        |         68.557 (+-0.332)        |           67.724 (+-0.451)           |      0.988 (+-0.000)    
      4 torch.uint8 channels_last bilinear (270, 268) -> (224, 224) aa=True      |                        |        156.454 (+-1.265)        |          156.827 (+-1.646)           |      1.002 (+-0.000)    
      4 torch.uint8 channels_last bilinear (1024, 1024) -> (256, 256) aa=True    |                        |        660.145 (+-2.704)        |          680.733 (+-4.411)           |      1.031 (+-0.000)    
      4 torch.uint8 channels_last bilinear (64, 64) -> (224, 224) aa=False       |                        |         76.420 (+-0.352)        |           73.980 (+-0.507)           |      0.968 (+-0.000)    
      4 torch.uint8 channels_last bilinear (224, 224) -> (270, 268) aa=False     |                        |        155.974 (+-1.197)        |          150.360 (+-1.209)           |      0.964 (+-0.000)    
      4 torch.uint8 channels_last bilinear (256, 256) -> (1024, 1024) aa=False   |                        |        920.969 (+-8.523)        |          888.575 (+-6.037)           |      0.965 (+-0.000)    
      4 torch.uint8 channels_last bilinear (224, 224) -> (64, 64) aa=False       |                        |         49.699 (+-0.249)        |           47.945 (+-0.264)           |      0.965 (+-0.000)    
      4 torch.uint8 channels_last bilinear (270, 268) -> (224, 224) aa=False     |                        |        140.824 (+-1.297)        |          134.807 (+-0.937)           |      0.957 (+-0.000)    
      4 torch.uint8 channels_last bilinear (1024, 1024) -> (256, 256) aa=False   |                        |        432.930 (+-2.847)        |          421.084 (+-2.218)           |      0.973 (+-0.000)    
      3 torch.uint8 channels_first bilinear (256, 256) -> (32, 32) aa=True       |    38.506 (+-0.288)    |        130.222 (+-0.971)        |          129.264 (+-1.054)           |      0.993 (+-0.000)    
      3 torch.uint8 channels_first bilinear (256, 256) -> (32, 32) aa=False      |                        |        112.703 (+-0.996)        |          111.873 (+-1.004)           |      0.993 (+-0.000)    
      3 torch.uint8 channels_first bilinear (256, 256) -> (224, 224) aa=True     |   127.474 (+-1.095)    |        299.799 (+-1.650)        |          317.713 (+-3.049)           |      1.060 (+-0.000)    
      3 torch.uint8 channels_first bilinear (256, 256) -> (224, 224) aa=False    |                        |        288.153 (+-2.180)        |          294.468 (+-8.713)           |      1.022 (+-0.000)    
      3 torch.uint8 channels_first bilinear (256, 256) -> (320, 320) aa=True     |   178.655 (+-1.528)    |        434.897 (+-2.770)        |          471.941 (+-5.183)           |      1.085 (+-0.000)    
      3 torch.uint8 channels_first bilinear (256, 256) -> (320, 320) aa=False    |                        |        434.404 (+-2.845)        |          463.577 (+-6.143)           |      1.067 (+-0.000)    
      3 torch.uint8 channels_first bilinear (520, 520) -> (32, 32) aa=True       |   112.817 (+-1.483)    |        435.090 (+-2.517)        |          433.944 (+-2.470)           |      0.997 (+-0.000)    
      3 torch.uint8 channels_first bilinear (520, 520) -> (32, 32) aa=False      |                        |        366.099 (+-1.537)        |          366.145 (+-2.277)           |      1.000 (+-0.000)    
      3 torch.uint8 channels_first bilinear (520, 520) -> (224, 224) aa=True     |   281.223 (+-1.866)    |        668.993 (+-3.353)        |          695.277 (+-3.210)           |      1.039 (+-0.000)    
      3 torch.uint8 channels_first bilinear (520, 520) -> (224, 224) aa=False    |                        |        603.973 (+-3.316)        |          610.557 (+-3.784)           |      1.011 (+-0.000)    
      3 torch.uint8 channels_first bilinear (712, 712) -> (32, 32) aa=True       |   185.812 (+-1.516)    |        771.992 (+-4.548)        |          773.219 (+-3.421)           |      1.002 (+-0.000)    
      3 torch.uint8 channels_first bilinear (712, 712) -> (32, 32) aa=False      |                        |        652.606 (+-2.563)        |          651.833 (+-3.869)           |      0.999 (+-0.000)    
      3 torch.uint8 channels_first bilinear (712, 712) -> (224, 224) aa=True     |   410.762 (+-7.320)    |        1071.152 (+-8.860)       |         1097.446 (+-14.107)          |      1.025 (+-0.000)    
      3 torch.uint8 channels_first bilinear (712, 712) -> (224, 224) aa=False    |                        |        932.623 (+-4.465)        |          942.783 (+-4.818)           |      1.011 (+-0.000)    
      3 torch.uint8 channels_first bilinear (64, 64) -> (224, 224) aa=True       |    61.081 (+-0.486)    |        160.946 (+-1.150)        |          183.886 (+-3.023)           |      1.143 (+-0.000)    
      3 torch.uint8 channels_first bilinear (224, 224) -> (270, 268) aa=True     |   132.842 (+-1.040)    |        326.660 (+-4.164)        |          346.158 (+-3.725)           |      1.060 (+-0.000)    
      3 torch.uint8 channels_first bilinear (256, 256) -> (1024, 1024) aa=True   |   944.319 (+-4.021)    |       2641.139 (+-22.075)       |         2942.516 (+-116.401)         |      1.114 (+-0.000)    
      3 torch.uint8 channels_first bilinear (224, 224) -> (64, 64) aa=True       |    52.201 (+-0.576)    |        132.650 (+-1.043)        |          132.548 (+-0.931)           |      0.999 (+-0.000)    
      3 torch.uint8 channels_first bilinear (270, 268) -> (224, 224) aa=True     |   139.106 (+-1.284)    |        316.437 (+-1.687)        |          338.116 (+-2.309)           |      1.069 (+-0.000)    
      3 torch.uint8 channels_first bilinear (1024, 1024) -> (256, 256) aa=True   |   699.076 (+-3.907)    |       1963.708 (+-16.905)       |         2020.990 (+-16.370)          |      1.029 (+-0.000)    
      3 torch.uint8 channels_first bilinear (64, 64) -> (224, 224) aa=False      |                        |        159.395 (+-1.234)        |          181.866 (+-2.786)           |      1.141 (+-0.000)    
      3 torch.uint8 channels_first bilinear (224, 224) -> (270, 268) aa=False    |                        |        324.432 (+-1.922)        |          349.501 (+-4.249)           |      1.077 (+-0.000)    
      3 torch.uint8 channels_first bilinear (256, 256) -> (1024, 1024) aa=False  |                        |       2641.326 (+-22.937)       |         2988.936 (+-146.450)         |      1.132 (+-0.000)    
      3 torch.uint8 channels_first bilinear (224, 224) -> (64, 64) aa=False      |                        |        113.960 (+-1.272)        |          113.022 (+-0.837)           |      0.992 (+-0.000)    
      3 torch.uint8 channels_first bilinear (270, 268) -> (224, 224) aa=False    |                        |        301.588 (+-2.490)        |          309.067 (+-3.466)           |      1.025 (+-0.000)    
      3 torch.uint8 channels_first bilinear (1024, 1024) -> (256, 256) aa=False  |                        |       1740.477 (+-17.844)       |         1760.707 (+-18.867)          |      1.012 (+-0.000)    
      4 torch.uint8 channels_first bilinear (256, 256) -> (32, 32) aa=True       |                        |         71.521 (+-0.419)        |           70.344 (+-0.248)           |      0.984 (+-0.000)    
      4 torch.uint8 channels_first bilinear (256, 256) -> (32, 32) aa=False      |                        |         54.372 (+-0.278)        |           53.607 (+-0.192)           |      0.986 (+-0.000)    
      4 torch.uint8 channels_first bilinear (256, 256) -> (224, 224) aa=True     |                        |        254.658 (+-1.448)        |          262.699 (+-3.071)           |      1.032 (+-0.000)    
      4 torch.uint8 channels_first bilinear (256, 256) -> (224, 224) aa=False    |                        |        242.352 (+-1.244)        |          253.042 (+-3.699)           |      1.044 (+-0.000)    
      4 torch.uint8 channels_first bilinear (256, 256) -> (320, 320) aa=True     |                        |        409.021 (+-2.870)        |          429.776 (+-7.453)           |      1.051 (+-0.000)    
      4 torch.uint8 channels_first bilinear (256, 256) -> (320, 320) aa=False    |                        |        407.703 (+-2.925)        |          424.774 (+-9.492)           |      1.042 (+-0.000)    
      4 torch.uint8 channels_first bilinear (520, 520) -> (32, 32) aa=True       |                        |        193.466 (+-1.258)        |          192.311 (+-1.715)           |      0.994 (+-0.000)    
      4 torch.uint8 channels_first bilinear (520, 520) -> (32, 32) aa=False      |                        |        125.037 (+-0.708)        |          124.377 (+-0.689)           |      0.995 (+-0.000)    
      4 torch.uint8 channels_first bilinear (520, 520) -> (224, 224) aa=True     |                        |        443.091 (+-2.849)        |          455.736 (+-2.886)           |      1.029 (+-0.000)    
      4 torch.uint8 channels_first bilinear (520, 520) -> (224, 224) aa=False    |                        |        377.826 (+-10.416)       |          384.434 (+-3.405)           |      1.017 (+-0.000)    
      4 torch.uint8 channels_first bilinear (712, 712) -> (32, 32) aa=True       |                        |        319.166 (+-2.438)        |          324.199 (+-2.901)           |      1.016 (+-0.000)    
      4 torch.uint8 channels_first bilinear (712, 712) -> (32, 32) aa=False      |                        |        200.002 (+-1.009)        |          200.237 (+-1.403)           |      1.001 (+-0.000)    
      4 torch.uint8 channels_first bilinear (712, 712) -> (224, 224) aa=True     |                        |        635.153 (+-2.833)        |          672.493 (+-3.950)           |      1.059 (+-0.000)    
      4 torch.uint8 channels_first bilinear (712, 712) -> (224, 224) aa=False    |                        |        496.850 (+-1.966)        |          524.986 (+-5.093)           |      1.057 (+-0.000)    
      4 torch.uint8 channels_first bilinear (64, 64) -> (224, 224) aa=True       |                        |        170.813 (+-1.150)        |          180.047 (+-4.952)           |      1.054 (+-0.000)    
      4 torch.uint8 channels_first bilinear (224, 224) -> (270, 268) aa=True     |                        |        300.916 (+-1.478)        |          314.463 (+-4.386)           |      1.045 (+-0.000)    
      4 torch.uint8 channels_first bilinear (256, 256) -> (1024, 1024) aa=True   |                        |       2944.516 (+-23.691)       |         3196.146 (+-101.697)         |      1.085 (+-0.000)    
      4 torch.uint8 channels_first bilinear (224, 224) -> (64, 64) aa=True       |                        |         88.943 (+-0.364)        |           88.856 (+-0.530)           |      0.999 (+-0.000)    
      4 torch.uint8 channels_first bilinear (270, 268) -> (224, 224) aa=True     |                        |        265.157 (+-1.509)        |          277.140 (+-2.539)           |      1.045 (+-0.000)    
      4 torch.uint8 channels_first bilinear (1024, 1024) -> (256, 256) aa=True   |                        |       1189.442 (+-25.499)       |         1347.051 (+-11.137)          |      1.133 (+-0.000)    
      4 torch.uint8 channels_first bilinear (64, 64) -> (224, 224) aa=False      |                        |        169.597 (+-2.039)        |          185.902 (+-3.013)           |      1.096 (+-0.000)    
      4 torch.uint8 channels_first bilinear (224, 224) -> (270, 268) aa=False    |                        |        299.538 (+-1.940)        |          319.375 (+-3.580)           |      1.066 (+-0.000)    
      4 torch.uint8 channels_first bilinear (256, 256) -> (1024, 1024) aa=False  |                        |       2922.224 (+-22.677)       |         3245.043 (+-113.882)         |      1.110 (+-0.000)    
      4 torch.uint8 channels_first bilinear (224, 224) -> (64, 64) aa=False      |                        |         70.313 (+-0.343)        |           69.877 (+-0.367)           |      0.994 (+-0.000)    
      4 torch.uint8 channels_first bilinear (270, 268) -> (224, 224) aa=False    |                        |        249.066 (+-1.428)        |          259.398 (+-3.671)           |      1.041 (+-0.000)    
      4 torch.uint8 channels_first bilinear (1024, 1024) -> (256, 256) aa=False  |                        |        872.716 (+-7.046)        |         1067.531 (+-30.590)          |      1.223 (+-0.000)    

Times are in microseconds (us).