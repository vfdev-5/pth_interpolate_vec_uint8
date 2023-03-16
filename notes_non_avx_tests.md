

```bash
git clone git@github.com:vfdev-5/pth_interpolate_vec_uint8.git

docker run --rm -it -v $PWD:/ws -w /ws --ipc=host python:3.9-slim-bullseye /bin/bash

pip install torch --index-url https://download.pytorch.org/whl/cpu

python -c "import torch; print(torch.__config__.show())"

pip install Pillow numpy fire

cd pth_interpolate_vec_uint8

# Quick test:
python -u run_bench_interp2.py output/test.log --min_run_time=0.1 --with_torchvision

# Benchmarks:
export fileprefix=$(date "+%Y%m%d-%H%M%S") && python -u run_bench_interp2.py output/${fileprefix}-PT2.0.pkl --tag=PT2.0 --with_torchvision &> output/${fileprefix}-PT2.0.log
```

