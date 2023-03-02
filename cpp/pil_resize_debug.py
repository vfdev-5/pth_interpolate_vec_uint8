"""
# apt-get install cgdb

OMP_NUM_THREADS=1 cgdb --args python pil_resize_debug.py

(gdb) b pillow-simd/src/libImaging/Resample.c:511
(gdb) b pillow-simd/src/libImaging/Resample.c:579
(gdb) b pillow-simd/src/libImaging/Resample.c:312
(gdb) b pillow-simd/src/libImaging/ResampleSIMDHorizontalConv.c:24
(gdb) b pillow-simd/src/_imaging.c:1787


506│ ImagingResample(Imaging imIn, int xsize, int ysize, int filter, float box[4])
->
562│ ImagingResampleInner

(gdb) p ksize_horiz
$35 = 9
(gdb) p {kk_horiz[0], kk_horiz[1], kk_horiz[2], kk_horiz[3], kk_horiz[4], kk_horiz[5], kk_horiz[6], kk_horiz[7], kk_horiz[8]}
$32 = {0.17857142857142858, 0.25, 0.25, 0.17857142857142858, 0.10714285714285714, 0.035714285714285712, 0, 0, 0}
(gdb) p {kk_horiz[9], kk_horiz[10], kk_horiz[11], kk_horiz[12], kk_horiz[13], kk_horiz[14], kk_horiz[15], kk_horiz[16], kk_horiz[17]}
$33 = {0.035714285714285712, 0.10714285714285714, 0.17857142857142858, 0.25, 0.25, 0.17857142857142858, 0, 0, 0}

(gdb) p ksize_vert
$34 = 9
(gdb) p {kk_vert[0], kk_vert[1], kk_vert[2], kk_vert[3], kk_vert[4], kk_vert[5], kk_vert[6], kk_vert[7], kk_vert[8]}
$38 = {0.17857142857142858, 0.25, 0.25, 0.17857142857142858, 0.10714285714285714, 0.035714285714285712, 0, 0, 0}
(gdb) p {kk_vert[9], kk_vert[10], kk_vert[11], kk_vert[12], kk_vert[13], kk_vert[14], kk_vert[15], kk_vert[16], kk_vert[17]}
$39 = {0.035714285714285712, 0.10714285714285714, 0.17857142857142858, 0.25, 0.25, 0.17857142857142858, 0, 0, 0}

->
311│ ImagingResampleHorizontal_8bpc(

(gdb) p maxkk
$8 = 0.25

(gdb) p coefs_precision
$12 = 16
(gdb) p {kk[0], kk[1], kk[2], kk[3], kk[4], kk[5], kk[6], kk[7], kk[8]}
$14 = {11703, 16384, 16384, 11703, 7022, 2341, 0, 0, 0}
(gdb) p {kk[9], kk[10], kk[11], kk[12], kk[13], kk[14], kk[15], kk[16], kk[17]}
$15 = {2341, 7022, 11703, 16384, 16384, 11703, 0, 0, 0}

->
337│     } else if (imIn->type == IMAGING_TYPE_UINT8) {
338│         yy = 0;
339│         for (; yy < imOut->ysize - 3; yy += 4) {
340├───────────> ImagingResampleHorizontalConvolution8u4x(


"""

#### Vertical pass

# import numpy as np
# from PIL import Image

# i = np.arange(4 * 20 * 3, dtype="uint8").reshape(4, 20, 3)
# # print(i[:, :, 0])
# # print(i[:, :, 1])
# # print(i[:, :, 2])
# print(i[0, ...].ravel())
# # print(i[1, ...].ravel())
# # print(i[2, ...].ravel())

# i = Image.fromarray(i)
# print(i.size)

# o = i.resize([20, 2], 2)
# print(o.size)
# oo = np.asarray(o)

# print(oo[0, :, :].tolist())
# print(oo[1, :, :].tolist())


#### Horizontal pass

import numpy as np
from PIL import Image

i = np.arange(2 * 18 * 3, dtype="uint8").reshape(2, 18, 3)
print(i[:, :, 0])
print(i[:, :, 1])
print(i[:, :, 2])

i = Image.fromarray(i)
print(i.size)

o = i.resize([2, 2], 2)
print(o.size)
oo = np.asarray(o)

print(oo[:, :, :].tolist())
