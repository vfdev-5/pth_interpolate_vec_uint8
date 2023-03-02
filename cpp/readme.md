# Build and run

```bash
mkdir build && cd build
# cmake -DCMAKE_PREFIX_PATH="/pytorch/torch/" -DCMAKE_BUILD_TYPE=Debug ..
cmake -DCMAKE_PREFIX_PATH="/pytorch/torch/" -DCMAKE_BUILD_TYPE=Release ..
make -j4 && ./check
```
