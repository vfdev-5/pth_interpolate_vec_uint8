# Check and debug OpenCV code

```
docker run -it \
    --gpus=all \
    -v $PWD:/opencv \
    -w /opencv \
    -v $PWD/../pth:/tmp/pth \
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
    pip install numpy && \
    apt-get install -y gdb python3.8-gdb
```


```
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Debug ..
make
```


## Debug cv2.resize

```
# check execution
cd /tmp/pth/interpolate_vec_uint8 && PYTHONPATH="/usr/local/lib/python3.8/site-packages/:$PYTHONPATH" python -u debug_opencv_resize.py
```

```
PYTHONPATH="/usr/local/lib/python3.8/site-packages/:$PYTHONPATH" gdb --args python -u debug_opencv_resize.py


(gdb) info files


(gdb) b /opencv/modules/imgproc/src/resize.cpp:4058
(gdb) b /opencv/modules/imgproc/src/resize.cpp:3685
(gdb) b /opencv/modules/imgproc/src/resize.cpp:3819
(gdb) b /opencv/modules/core/src/parallel.cpp:510
(gdb) b /opencv/modules/core/src/parallel.cpp:549
(gdb) run
```


Notes:
```
cv::resize (_src=..., _dst=..., dsize=..., inv_scale_x=0, inv_scale_y=0, interpolation=2) at /opencv/modules/imgproc/src/resize.cpp:4058

-> Does not enter into CV_OCL_RUN as non activated
-> hal::resize()
    -> Does not enter into CALL_HAL
    -> ipp_resize
        -> ipp_resizeParallel: virtual void operator() (const Range& range) const CV_OVERRIDE

        #0  iwiResize_Process (pSpec=0xdac300, pSrcImage=0x7fffffffc6b0, pDstImage=0x7fffffffc720, border=ippBorderRepl, pBorderVal=0x7fffffffc580, pTile=0x7fffffffc3a0) at /opencv/build/3rdparty/ippicv/ippicv_lnx/iw/src/iw_image_transform_resize.c:128
        #1  0x00007fffd3efa62e in ipp::IwiResize::operator() (this=0x7fffffffc668, srcImage=..., dstImage=..., border=..., tile=...) at /opencv/build/3rdparty/ippicv/ippicv_lnx/iw/include/iw++/iw_image_transform.hpp:330
        #2  0x00007fffd3f0a3fc in cv::ipp_resizeParallel::operator() (this=0x7fffffffc650,c range=...) at /opencv/modules/imgproc/src/resize.cpp:3514
```

They define a buffer:

Buffer, size = (dst_width * cn + dst_height) * (sizeof(int) + ksize * sizeof(float))
- ksize = 4
- cn, dst_width, dst_height = (3, 128, 128)

