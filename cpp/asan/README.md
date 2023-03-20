
```
mkdir build && cd build

CC=clang CXX=clang++ LDSHARED='clang --shared' cmake ..

unset LD_PRELOAD && make

export LD_PRELOAD=/usr/lib/clang/10.0.0/lib/linux/libclang_rt.asan-x86_64.so && ./check_mem_boundaries_with_asan

LD_PRELOAD=/usr/lib/clang/10.0.0/lib/linux/libclang_rt.asan-x86_64.so ./check_mem_boundaries_with_asan
```