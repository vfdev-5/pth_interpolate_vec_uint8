# Check if it makes sense to define shuffle masks as static global constants

import PIL.Image

import torch
import torch.utils.benchmark as benchmark

import fire


def pth_downsample_i8(img, mode, size, aa=True):

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


def multiple_pth_downsample_i8(*args, **kwargs):
    n = kwargs["n"]
    del kwargs["n"]
    c = 0
    for _ in range(n):
        out = pth_downsample_i8(*args, **kwargs)
        c += out.shape[0]


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


def main(min_run_time=10.0):
    tag = "PR"
    results = []

    torch.manual_seed(12)

    n = 10000

    mf = "channels_last"
    size = 325
    osize, aa, mode = (225, 225), False, "bilinear"
    c, dtype = (3, torch.float32)

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

    memory_format = torch.channels_last if mf == "channels_last" else torch.contiguous_format
    tensor = tensor[None, ...].contiguous(memory_format=memory_format)

    output = pth_downsample_i8(tensor, mode=mode, size=osize, aa=aa)
    output = output[0, ...]

    # Tensor interp
    results.append(
        benchmark.Timer(
            # output = pth_downsample_i8(tensor, mode=mode, size=(osize, osize), aa=aa)
            stmt=f"fn(data, mode='{mode}', size={osize}, aa={aa}, n={n})",
            globals={
                "data": tensor,
                "fn": multiple_pth_downsample_i8
            },
            num_threads=torch.get_num_threads(),
            label=f"Multiple Resize, loop of {n} iters",
            sub_label=f"{c} {dtype} {mf} {mode} {size} -> {osize} aa={aa}",
            description=f"torch ({torch.__version__}) {tag}",
        ).blocked_autorange(min_run_time=min_run_time)
    )

    compare = benchmark.Compare(results)
    compare.print()


if __name__ == "__main__":

    torch.set_num_threads(1)

    print("")
    print(f"Torch version: {torch.__version__}")
    print(f"Torch config: {torch.__config__.show()}")
    print(f"Num threads: {torch.get_num_threads()}")
    print("")
    print("PIL version: ", PIL.__version__)

    fire.Fire(main)