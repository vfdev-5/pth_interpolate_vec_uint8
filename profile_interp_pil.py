import numpy as np
import PIL.Image

import torch
import torch.utils.benchmark as benchmark

import fire


if not hasattr(PIL.Image, "Resampling"):
    resampling_map = {
        "bilinear": PIL.Image.BILINEAR,
        "nearest": PIL.Image.NEAREST,
        "bicubic": PIL.Image.BICUBIC,
    }
else:
    resampling_map = {
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "nearest": PIL.Image.Resampling.NEAREST,
        "bicubic": PIL.Image.Resampling.BICUBIC,
    }


def main():
    torch.manual_seed(12)

    # for mf in ["channels_last", "channels_first"]:
    for mf in ["channels_last", ]:
        for c, dtype in [
            (3, torch.uint8),
            # (4, torch.uint8),
        ]:
            # for size in [256, 520, 712]:
            # for size in [256, 520]:
            for size in [256, ]:
            # for size in [270, ]:
                for osize, aa, mode in [
                    ((224, 224), True, "bilinear"),
                    # ((224, 224), False, "bilinear"),
                    # Horizontal only
                    # ((256, 224), True, "bilinear"),
                    # ((256, 224), False, "bilinear"),
                    # Vertical only
                    # ((224, 256), True, "bilinear"),
                    # ((225, 256), False, "bilinear"),

                    # ((32, 32), True, "bilinear"),
                    # ((32, 32), False, "bilinear"),
                ]:

                    if dtype == torch.bool:
                        tensor = torch.randint(0, 2, size=(c, size, size), dtype=dtype)
                    elif dtype == torch.complex64:
                        real = torch.randint(0, 256, size=(c, size, size), dtype=torch.float32)
                        imag = torch.randint(0, 256, size=(c, size, size), dtype=torch.float32)
                        tensor = torch.complex(real, imag)
                    elif dtype == torch.int8:
                        tensor = torch.randint(-127, 127, size=(c, size, size), dtype=dtype)
                    else:
                        tensor = torch.randint(0, 256, size=(c, size, size), dtype=dtype)

                    pil_img = None

                    if dtype == torch.uint8 and c == 3 and aa:
                        np_array = tensor.clone().permute(1, 2, 0).contiguous().numpy()
                        pil_img = PIL.Image.fromarray(np_array)
                        output_pil_img = pil_img.resize(osize[::-1], resample=resampling_map[mode])

                    assert pil_img is not None
                    for _ in range(30000):
                        pil_img.resize(osize[::-1], resample=resampling_map[mode])


if __name__ == "__main__":

    torch.set_num_threads(1)

    print("")
    print("PIL version: ", PIL.__version__)

    fire.Fire(main)