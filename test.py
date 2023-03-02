import argparse
import os
import numpy as np
import PIL
from PIL import Image
from functools import partial

import torch
import torch.utils.benchmark as benchmark


# Original image size: 906, 438
sizes = [
    (320, 196),
    (460, 220),
    (120, 96),
    (1200, 196),
    (120, 1200),
]


def pth_downsample(img, mode, size):

    align_corners = False
    if mode == "nearest":
        align_corners = None

    out = torch.nn.functional.interpolate(
        img.float(), size=size,
        mode=mode,
        align_corners=align_corners,
        antialias=True,
    )
    return out.byte()


def pth_downsample_uint8(img, mode, size):

    align_corners = False
    if mode == "nearest":
        align_corners = None

    out = torch.nn.functional.interpolate(
        img, size=size,
        mode=mode,
        align_corners=align_corners,
        antialias=True,
    )
    return out

resampling_map = {"bilinear": PIL.Image.BILINEAR, "nearest": PIL.Image.NEAREST, "bicubic": PIL.Image.BICUBIC}


def run_bench(size, mode, dtype, min_run_time=10):
    # All variables are taken from __main__ scope

    inv_size = size[::-1]
    resample = resampling_map[mode]

    mem_format = "channels_last" if t_img.is_contiguous(memory_format=torch.channels_last) else "channels_first"
    is_contiguous = "contiguous" if t_img.is_contiguous() else "non-contiguous"

    label = f"Downsampling: {t_img.shape} -> {size}"
    sub_label = f"{mem_format} {is_contiguous}"

    if dtype == "uint8":
        pth_op = pth_downsample_uint8
    else:
        pth_op = pth_downsample

    results = [
        benchmark.Timer(
            # pil_img.resize(size, resample=resample_val)
            stmt=f"img.resize(size, resample=resample_val)",
            globals={
                "img": pil_img,
                "size": size,
                "resample_val": resample,
            },
            num_threads=torch.get_num_threads(),
            label=label,
            sub_label=sub_label,
            description=f"PIL {PIL.__version__}",
        ).blocked_autorange(min_run_time=min_run_time),

        benchmark.Timer(
            # pth_downsample*(t_img, mode, size)
            stmt=f"f(x, mode, size)",
            globals={
                "x": t_img,
                "size": inv_size,
                "mode": mode,
                "f": pth_op
            },
            num_threads=torch.get_num_threads(),
            label=label,
            sub_label=sub_label,
            description=f"{torch.version.__version__}, using {dtype}",
        ).blocked_autorange(min_run_time=min_run_time),
    ]
    return results


if __name__ == "__main__":

    torch.set_num_threads(1)

    parser = argparse.ArgumentParser("Test interpolation with anti-alias option")
    parser.add_argument(
        "--mode", default="bilinear", type=str,
        choices=["bilinear", "nearest", "bicubic"],
        help="Interpolation mode"
    )
    parser.add_argument(
        "--size", type=int, nargs=2,
        help="Use the specified size for the tests"
    )

    args = parser.parse_args()

    mode = args.mode
    pil_img = Image.open("data/test.png").convert("RGB")
    # dtype = "uint8"
    dtype = "float"
    min_run_time = 10

    resample = resampling_map[mode]

    if args.size is not None:
        print(f"Use specified size: {args.size}")
        sizes = [args.size, ]

    for mf in ["channels_first", "channels_last"]:
        for size in sizes:
            inv_size = size[::-1]
            pil_img_dn = pil_img.resize(size, resample=resample)
            t_pil_img_dn = torch.from_numpy(np.asarray(pil_img_dn).copy().transpose((2, 0, 1)))
            t_pil_img_dn = t_pil_img_dn[None, ...]

            memory_format = torch.channels_last if mf == "channels_last" else torch.contiguous_format
            t_img = torch.from_numpy(np.asarray(pil_img).copy().transpose((2, 0, 1)))
            t_img = t_img[None, ...].contiguous(memory_format=memory_format)
            t_img1 = t_img.clone()

            if dtype == "uint8":
                pth_op = pth_downsample_uint8
            else:
                pth_op = pth_downsample

            print("mem_format: ", "channels_last" if t_img1.is_contiguous(memory_format=torch.channels_last) else "channels_first")
            print("is_contiguous: ", t_img1.is_contiguous())

            pth_img_dn = pth_op(t_img1, mode, inv_size)

            # pth_pil_dn = Image.fromarray(pth_img_dn.permute(1, 2, 0).numpy())
            # fname = f"data/pth_{mode}_output_{size[0]}_{size[1]}.png"
            # pth_pil_dn.save(fname)
            # print(f"Saved downsampled proto output: {fname}")

            mae = torch.mean(torch.abs(t_pil_img_dn.float() - pth_img_dn.float()))
            max_abs_err = torch.max(torch.abs(t_pil_img_dn.float() - pth_img_dn.float()))
            print("PyTorch vs PIL: Mean Absolute Error:", mae.item())
            print("PyTorch vs PIL: Max Absolute Error:", max_abs_err.item())

            if mode == "bilinear":
                assert mae.item() < 1.0, mae.item()
                assert max_abs_err.item() < 1.0 + 1e-5, max_abs_err.item()
            elif mode == "nearest":
                pass
                # assert mae.item() < 5.0
                # assert max_abs_err.item() < 1.0 + 1e-5
            elif mode == "bicubic":
                assert mae.item() < 1.0
                assert max_abs_err.item() < 20.0

    print(f"Torch version: {torch.__version__}")
    print(f"Torch config: {torch.__config__.show()}")
    print(f"Num threads: {torch.get_num_threads()}")

    all_results = []
    for s in sizes:
        all_results += run_bench(s, mode, dtype, min_run_time)
    compare = benchmark.Compare(all_results)
    compare.print()

