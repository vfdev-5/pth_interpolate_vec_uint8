import numpy as np
import PIL.Image

import torch

try:
    import cv2
    has_cv2 = True
except ImportError as e:
    has_cv2 = False
    print(e)


resampling_map = {"bilinear": PIL.Image.BILINEAR, "nearest": PIL.Image.NEAREST, "bicubic": PIL.Image.BICUBIC}


def main():

    out_size = (24, 24)
    resample = "bicubic"
    align_corners = None if resample == "nearest" else False
    mf = "channels_first"
    # mf = "channels_last"
    antialias = False

    c = 1
    # size = 48
    # tensor_uint8 = torch.arange(c * size * size, dtype=torch.uint8).reshape(c, size, size)

    tensor_uint8 = torch.tensor([
        [ 12.,  13.,  14.,  15.,  16.,  17.,  18.,  19.],
        [ 60.,  61.,  62.,  63.,  64.,  65.,  66.,  67.],
        [108., 109., 110., 111., 112., 113., 114., 115.],
        [156., 157., 158., 159., 160., 161., 162., 163.],
        [204., 205., 206., 207., 208., 209., 210., 211.],
        [252., 253., 254., 255.,   0.,   1.,   2.,   3.],
        [ 44.,  45.,  46.,  47.,  48.,  49.,  50.,  51.],
        [ 92.,  93.,  94.,  95.,  96.,  97.,  98.,  99.]
    ], dtype=torch.uint8)[None, ...]
    print(tensor_uint8.shape)
    out_size = (4, 4)
    size = tensor_uint8.shape[-1]

    tensor_float32 = tensor_uint8.float()

    tensor_uint8 = tensor_uint8[None, ...]
    tensor_float32 = tensor_float32[None, ...]

    if mf == "channels_last":
        tensor_uint8 = tensor_uint8.contiguous(memory_format=torch.channels_last)
        tensor_float32 = tensor_float32.contiguous(memory_format=torch.channels_last)

    print("Memory format:", mf)
    print("Antialias:", antialias)
    output_uint8 = torch.nn.functional.interpolate(
        tensor_uint8, mode=resample, size=out_size, align_corners=align_corners, antialias=antialias
    )
    print("output_uint8: \n", output_uint8[0, 0, :, :])

    output_float32 = torch.nn.functional.interpolate(
        tensor_float32, mode=resample, size=out_size, align_corners=align_corners, antialias=antialias
    )
    if resample == "bicubic":
        output_float32 = output_float32.clamp(min=0, max=255).round()

    print("output_float32: \n", output_float32[0, 0, :, :])

    abs_diff = torch.abs(output_float32 - output_uint8.float())
    mae = torch.mean(abs_diff)
    max_abs_err = torch.max(abs_diff)
    print("PyTorch uint8 vs PyTorch float: Mean Absolute Error:", mae.item())
    print("PyTorch uint8 vs PyTorch float: Max Absolute Error:", max_abs_err.item())

    if has_cv2 and not antialias:
        a_uint8 = tensor_uint8[0, ...].permute(1, 2, 0).contiguous().numpy()
        a_float32 = a_uint8.astype("float32")

        output_uint8_cv2 = cv2.resize(a_uint8, dsize=out_size, interpolation=cv2.INTER_CUBIC)
        print("output_uint8_cv2: \n", output_uint8_cv2[:8, :8])

        output_float32_cv2 = cv2.resize(a_float32, dsize=out_size, interpolation=cv2.INTER_CUBIC)
        print("output_float32_cv2: \n", output_float32_cv2[:8, :8])
        if resample == "bicubic":
            output_float32_cv2 = np.clip(output_float32_cv2, 0, 255).round()

        abs_diff = np.abs(output_float32_cv2 - output_uint8_cv2.astype("float32"))
        mae = np.mean(abs_diff)
        max_abs_err = np.max(abs_diff)
        print("CV2 uint8 vs CV2 float: Mean Absolute Error:", mae.item())
        print("CV2 uint8 vs CV2 float: Max Absolute Error:", max_abs_err.item())


    # m = abs_diff > max_abs_err.item() - 1e-1
    # print("Diff f32:\n", output_float32[m])
    # print("Diff ui8:\n", output_uint8[m])

    # print("Non-matched pixels:")
    # indices = torch.nonzero(m)
    # for idx in indices:
    #     print("out index:", idx)
    #     print("out f32:", output_float32[idx[0], idx[1], max(idx[2]-2,0):idx[2]+2, max(idx[3]-2,0):idx[3]+2])
    #     print("out ui8:", output_uint8[idx[0], idx[1], max(idx[2]-2,0):idx[2]+2, max(idx[3]-2,0):idx[3]+2])

    #     scale = size / min(out_size)
    #     idx = (idx * scale).to(torch.long)
    #     print("in index:", idx)
    #     print("f32:", tensor_float32[idx[0], idx[1], max(idx[2]-4,0):idx[2]+4, max(idx[3]-4,0):idx[3]+4])

    #     break



if __name__ == "__main__":

    torch.set_num_threads(1)

    print("")
    print(f"Torch version: {torch.__version__}")
    print(f"Torch config: {torch.__config__.show()}")
    print(f"Num threads: {torch.get_num_threads()}")
    print("")
    print("PIL version: ", PIL.__version__)

    main()
