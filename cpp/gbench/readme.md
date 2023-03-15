- Install Google Benchmark

```
# Check out the library.
git clone https://github.com/google/benchmark.git /tmp/gbenchmark
cd /tmp/gbenchmark && \
cmake -E make_directory "build" && \
cmake -DCMAKE_BUILD_TYPE=Release -DBENCHMARK_DOWNLOAD_DEPENDENCIES=on -S . -B "build" && \
cmake --build "build" --config Release

cmake --build "build" --config Release --target install
```

- Run benchmarks

```
make uint32_vs_memcpy && ./uint32_vs_memcpy
```