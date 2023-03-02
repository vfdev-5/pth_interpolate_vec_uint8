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


def pth_downsample(img, mode, size, aa=True):

    align_corners = False
    if mode == "nearest":
        align_corners = None

    out = torch.nn.functional.interpolate(
        img.float(), size=size,
        mode=mode,
        align_corners=align_corners,
        antialias=aa,
    )
    return out.to(img.dtype)


def pth_downsample_uint8(img, mode, size, aa=True):

    align_corners = False
    if mode == "nearest":
        align_corners = None

    out = torch.nn.functional.interpolate(
        img, size=size,
        mode=mode,
        align_corners=align_corners,
        antialias=aa,
    )
    return out

resampling_map = {
    "bilinear": PIL.Image.Resampling.BILINEAR,
    "nearest": PIL.Image.Resampling.NEAREST,
    "bicubic": PIL.Image.Resampling.BICUBIC,
}


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

    resample = resampling_map[mode]

    if args.size is not None:
        print(f"Use specified size: {args.size}")
        sizes = [args.size, ]

    for aa in [True, False]:
        for mem_format in ["channels_first", "channels_last"]:
            for size in sizes:
                inv_size = size[::-1]

                if aa:
                    pil_img_dn = pil_img.resize(size, resample=resample)
                    t_pil_img_dn = torch.from_numpy(np.asarray(pil_img_dn).copy().transpose((2, 0, 1)))
                    t_pil_img_dn = t_pil_img_dn[None, ...]
                else:
                    t_pil_img_dn = None

                t_img = torch.from_numpy(np.asarray(pil_img).copy().transpose((2, 0, 1)))
                t_img = t_img[None, ...]
                if mem_format == "channels_first":
                    t_img = t_img.contiguous()
                else:
                    t_img = t_img.contiguous(memory_format=torch.channels_last)
                t_img1 = t_img.clone()

                print("906, 438 -> ", size)
                print("mem_format: ", "channels_last" if t_img1.is_contiguous(memory_format=torch.channels_last) else "channels_first")
                print("is_contiguous: ", t_img1.is_contiguous())
                print("AA: ", aa)

                pth_img_dn = pth_downsample_uint8(t_img1, mode, inv_size, aa=aa)

                # pth_pil_dn = Image.fromarray(pth_img_dn.permute(1, 2, 0).numpy())
                # fname = f"data/pth_{mode}_output_{size[0]}_{size[1]}.png"
                # pth_pil_dn.save(fname)
                # print(f"Saved downsampled proto output: {fname}")

                if t_pil_img_dn is not None:
                    mae = torch.mean(torch.abs(t_pil_img_dn.float() - pth_img_dn.float()))
                    max_abs_err = torch.max(torch.abs(t_pil_img_dn.float() - pth_img_dn.float()))
                    print("PyTorch uint8 vs PIL: Mean Absolute Error:", mae.item())
                    print("PyTorch uint8 vs PIL: Max Absolute Error:", max_abs_err.item())

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

                pth_img_dn2 = pth_downsample(t_img1, mode, inv_size, aa=aa)

                mae = torch.mean(torch.abs(pth_img_dn2.float() - pth_img_dn.float()))
                max_abs_err = torch.max(torch.abs(pth_img_dn2.float() - pth_img_dn.float()))
                print("PyTorch uint8 vs PyTorch float: Mean Absolute Error:", mae.item())
                print("PyTorch uint8 vs PyTorch float: Max Absolute Error:", max_abs_err.item())

                if mode == "bilinear":
                    assert mae.item() < 1.0, (mae.item(), pth_img_dn2[0, 0, 0, :], pth_img_dn[0, 0, 0, :])
                    assert max_abs_err.item() < 1.0 + 1e-5, max_abs_err.item()
                elif mode == "nearest":
                    pass
                    # assert mae.item() < 5.0
                    # assert max_abs_err.item() < 1.0 + 1e-5
                elif mode == "bicubic":
                    assert mae.item() < 1.0
                    assert max_abs_err.item() < 20.0

                if t_pil_img_dn is not None:
                    mae = torch.mean(torch.abs(t_pil_img_dn.float() - pth_img_dn2.float()))
                    max_abs_err = torch.max(torch.abs(t_pil_img_dn.float() - pth_img_dn2.float()))
                    print("PyTorch float vs PIL: Mean Absolute Error:", mae.item())
                    print("PyTorch float vs PIL: Max Absolute Error:", max_abs_err.item())

    print(f"Torch config: {torch.__config__.show()}")
    print(f"Pillow version: {PIL.__version__}")
