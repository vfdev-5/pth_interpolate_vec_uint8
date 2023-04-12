# Check and debug OpenCV code

```
docker run -it \
    --gpus=all \
    -v $PWD:/opencv \
    -w /opencv \
    -v $PWD/../tmp/pth:/tmp/pth \
    --name=opencv-debug \
    --cap-add=SYS_ADMIN \
    --cap-add=SYS_PTRACE \
    --network=host --security-opt seccomp:unconfined --ipc=host \
    nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04 \
    /bin/bash
```

```
apt-get update && ln -fs /usr/share/zoneinfo/Europe/Paris /etc/localtime && \
    apt-get install -y tzdata && \
    dpkg-reconfigure --frontend noninteractive tzdata && \
    apt-get install -y git cmake python3 python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    pip install numpy
```


```
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Debug ..
```

