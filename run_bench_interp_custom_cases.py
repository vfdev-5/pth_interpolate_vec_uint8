
import pickle
from pathlib import Path
import unittest.mock


import numpy as np
import PIL.Image

import torch
import torch.utils.benchmark as benchmark

import fire


from torchvision_functional_tensor import resize


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


def torchvision_resize(img, mode, size, aa=True):
    return resize(img, size=size, interpolation=mode, antialias=aa)


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


def patched_as_column_strings(self):
    concrete_results = [r for r in self._results if r is not None]
    env = f"({concrete_results[0].env})" if self._render_env else ""
    env = env.ljust(self._env_str_len + 4)
    output = ["  " + env + concrete_results[0].as_row_name]
    for m, col in zip(self._results, self._columns or ()):
        if m is None:
            output.append(col.num_to_str(None, 1, None))
        else:
            if len(m.times) == 1:
                spread = 0
            else:
                spread = float(torch.tensor(m.times, dtype=torch.float64).std(unbiased=len(m.times) > 1))
                if col._trim_significant_figures:
                    spread = benchmark.utils.common.trim_sigfig(spread, m.significant_figures)
            output.append(f"{m.median / self._time_scale:>3.3f} (+-{spread / self._time_scale:>3.3f})")
    return output


def run_benchmark(c, dtype, size, osize, aa, mode, mf="channels_first", min_run_time=10, tag="", with_torchvision=False, with_pillow=True, squeeze_unsqueeze_zero=False):
    results = []
    torch.manual_seed(12)

    if dtype == torch.bool:
        tensor = torch.randint(0, 2, size=(c, size[0], size[1]), dtype=dtype)
    elif dtype == torch.complex64:
        real = torch.randint(0, 256, size=(c, size[0], size[1]), dtype=torch.float32)
        imag = torch.randint(0, 256, size=(c, size[0], size[1]), dtype=torch.float32)
        tensor = torch.complex(real, imag)
    elif dtype == torch.int8:
        tensor = torch.randint(-127, 127, size=(c, size[0], size[1]), dtype=dtype)
    else:
        tensor = torch.randint(0, 256, size=(c, size[0], size[1]), dtype=dtype)

    expected_pil = None
    pil_img = None
    if with_pillow and dtype == torch.uint8 and c == 3 and aa:
        np_array = tensor.clone().permute(1, 2, 0).contiguous().numpy()
        pil_img = PIL.Image.fromarray(np_array)
        output_pil_img = pil_img.resize(osize[::-1], resample=resampling_map[mode])
        expected_pil = torch.from_numpy(np.asarray(output_pil_img)).clone().permute(2, 0, 1).contiguous()

    memory_format = torch.channels_last if mf == "channels_last" else torch.contiguous_format
    tensor = tensor[None, ...].contiguous(memory_format=memory_format)

    squeeze_unsqueeze_zero_label = ""
    if squeeze_unsqueeze_zero:
        squeeze_unsqueeze_zero_label = "(squeeze/unsqueeze)"
        tensor = tensor[0, ...]
        tensor = tensor[None, ...]

    # warm-up
    for _ in range(10):
        output = pth_downsample_i8(tensor, mode=mode, size=osize, aa=aa)
    output = output[0, ...]

    if expected_pil is not None:
        abs_diff = torch.abs(expected_pil.float() - output.float())
        mae = torch.mean(abs_diff)
        max_abs_err = torch.max(abs_diff)

        if mode == "bilinear":
            assert mae.item() < 1.0, mae.item()
            assert max_abs_err.item() < 2.0 + 1e-5, max_abs_err.item()
        else:
            raise RuntimeError(f"Unsupported mode: {mode}")

    # PIL
    if pil_img is not None:
        results.append(
            benchmark.Timer(
                # pil_img = pil_img.resize((osize, osize), resample=resampling_map[mode])
                stmt=f"data.resize({osize[::-1]}, resample=resample_val)",
                globals={
                    "data": pil_img,
                    "resample_val": resampling_map[mode],
                },
                num_threads=torch.get_num_threads(),
                label="Resize",
                sub_label=f"{c} {dtype} {mf} {mode} {size} -> {osize} aa={aa}",
                description=f"Pillow ({PIL.__version__})",
            ).blocked_autorange(min_run_time=min_run_time)
        )
    # Tensor interp
    results.append(
        benchmark.Timer(
            # output = pth_downsample_i8(tensor, mode=mode, size=(osize, osize), aa=aa)
            stmt=f"fn(data, mode='{mode}', size={osize}, aa={aa})",
            globals={
                "data": tensor,
                "fn": pth_downsample_i8
            },
            num_threads=torch.get_num_threads(),
            label="Resize",
            sub_label=f"{c} {dtype} {mf}{squeeze_unsqueeze_zero_label} {mode} {size} -> {osize} aa={aa}",
            description=f"torch ({torch.__version__}) {tag}",
        ).blocked_autorange(min_run_time=min_run_time)
    )
    # Torchvision resize
    if with_torchvision:
        results.append(
            benchmark.Timer(
                # output = torchvision_resize(tensor, mode=mode, size=(osize, osize), aa=aa)
                stmt=f"fn(data, mode='{mode}', size={osize}, aa={aa})",
                globals={
                    "data": tensor,
                    "fn": torchvision_resize
                },
                num_threads=torch.get_num_threads(),
                label="Resize",
                sub_label=f"{c} {dtype} {mf}{squeeze_unsqueeze_zero_label} {mode} {size} -> {osize} aa={aa}",
                description=f"torchvision resize",
            ).blocked_autorange(min_run_time=min_run_time)
        )

    return results


def main(
    output_folder: str,
    min_run_time: float = 10.0,
    tag: str = "",
    display: bool = True,
    with_torchvision: bool = False,
    with_pillow: bool = True,
    extended_test_cases=False,
    num_threads=1,
    squeeze_unsqueeze_zero=False
):
    torch.set_num_threads(num_threads)
    from datetime import datetime

    now = datetime.now().strftime('%Y%m%d-%H%M%S')
    output_filepath = Path(output_folder) / f"{now}-upsample-{tag}.pkl"

    print(f"Output filepath: {str(output_filepath)}")
    print(f"Torch version: {torch.__version__}")
    print(f"Torch config: {torch.__config__.show()}")
    print(f"Num threads: {torch.get_num_threads()}")
    print("")
    print("PIL version: ", PIL.__version__)

    test_results = []
    for mf in ["channels_first", "channels_last"]:
    # for mf in ["channels_first", ]:
    # for mf in ["channels_last", ]:
        for c, dtype in [
            (3, torch.uint8),
            # (3, torch.float32),
            # (4, torch.uint8),
        ]:
            # for size in [256, 520, 712]:
            for size in [400, ]:
                if isinstance(size, int):
                    size = (size, size)

                osize_aa_mode_list = [
                    # (32, True, "bilinear"),
                    # (32, False, "bilinear"),
                    # (32, False, "bicubic"),
                    # (224, True, "bilinear"),
                    (224, False, "bilinear"),
                    (224, False, "bicubic"),

                    (700, False, "bilinear"),
                    (700, False, "bicubic"),
                ]

                if size == (256, 256):
                    osize_aa_mode_list += [
                        # (320, True, "bilinear"),
                        (320, False, "bilinear"),
                        (320, False, "bicubic"),
                    ]

                for osize, aa, mode in osize_aa_mode_list:
                    if isinstance(osize, int):
                        osize = (osize, osize)

                    test_results += run_benchmark(
                        c=c, dtype=dtype, size=size,
                        osize=osize, aa=aa, mode=mode, mf=mf,
                        min_run_time=min_run_time, tag=tag,
                        with_torchvision=with_torchvision, with_pillow=with_pillow,
                        squeeze_unsqueeze_zero=squeeze_unsqueeze_zero,
                    )

            if not extended_test_cases:
                continue

            for aa in [True, False]:
            # for aa in [False, ]:
                mode = "bilinear"

                size_osize_list = [
                    (64, 224),
                    (224, (270, 268)),
                    (256, (1024, 1024)),
                    (224, 64),
                    ((270, 268), 224),
                    (256, 224),
                    (1024, 256),
                ]

                for size, osize in size_osize_list:
                    if isinstance(size, int):
                        size = (size, size)

                    if isinstance(osize, int):
                        osize = (osize, osize)

                    test_results += run_benchmark(
                        c=c, dtype=dtype, size=size,
                        osize=osize, aa=aa, mode=mode, mf=mf,
                        min_run_time=min_run_time, tag=tag,
                        with_torchvision=with_torchvision, with_pillow=with_pillow,
                        squeeze_unsqueeze_zero=squeeze_unsqueeze_zero,
                    )

    with open(output_filepath, "wb") as handler:
        output = {
            "filepath": str(output_filepath),
            "torch_version": torch.__version__,
            "torch_config": torch.__config__.show(),
            "num_threads": torch.get_num_threads(),
            "pil_version": PIL.__version__,
            "test_results": test_results,
        }
        pickle.dump(output, handler)

    if display:
        with unittest.mock.patch(
            "torch.utils.benchmark.utils.compare._Row.as_column_strings", patched_as_column_strings
        ):
            print()
            compare = benchmark.Compare(test_results)
            compare.print()


if __name__ == "__main__":
    fire.Fire(main)
