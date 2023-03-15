import numpy as np
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


def main(min_run_time=5.0, debug=False, use_perf=False, check_consistency=True):
    tag = "PR"
    results = []

    torch.manual_seed(12)

    for mf in ["channels_last", "channels_first"]:
    # for mf in ["channels_last", ]:
        for size in [256, 520, 712]:
        # for size in [256, 520]:
        # for size in [256, ]:
        # for size in [700, 520, 256]:
        # for size in [256, ]:
        # for size in [270, ]:
            for osize, aa, mode in [
                # ((224, 224), True, "bilinear"),
                # ((224, 224), False, "bilinear"),
                # ((32, 32), True, "bilinear"),
                # # ((32, 32), False, "bilinear"),

                # Horizontal only
                # ((size, int(size * 224 / 256)), True, "bilinear"),
                # ((size, 224), True, "bilinear"),
                # ((size, 227), True, "bilinear"),
                # ((size, 224), False, "bilinear"),
                # Vertical only
                ((int(size * 224 / 256), size), True, "bilinear"),
                ((227, size), True, "bilinear"),
                ((224, size), False, "bilinear"),
                ((32, size), False, "bilinear"),
            ]:
                for c, dtype in [
                    (3, torch.uint8),
                    (4, torch.uint8),
                ]:
                    if debug:
                        print("")
                        print(mf, c, dtype, size, osize, aa, mode)

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

                    expected_pil = None
                    pil_img = None


                    if dtype == torch.uint8 and c == 3 and aa and not use_perf:
                        np_array = tensor.clone().permute(1, 2, 0).contiguous().numpy()
                        pil_img = PIL.Image.fromarray(np_array)
                        output_pil_img = pil_img.resize(osize[::-1], resample=resampling_map[mode])
                        expected_pil = torch.from_numpy(np.asarray(output_pil_img)).clone().permute(2, 0, 1).contiguous()

                    memory_format = torch.channels_last if mf == "channels_last" else torch.contiguous_format
                    tensor = tensor[None, ...].contiguous(memory_format=memory_format)

                    if use_perf:
                        output = pth_downsample_i8(tensor, mode=mode, size=osize, aa=aa)
                        continue
                    else:
                        output = pth_downsample_i8(tensor, mode=mode, size=osize, aa=aa)
                        output = output[0, ...]

                    if debug:
                        continue

                    if check_consistency and expected_pil is not None:
                        abs_diff = torch.abs(expected_pil.float() - output.float())
                        mae = torch.mean(abs_diff)
                        max_abs_err = torch.max(abs_diff)

                        if mode == "bilinear":
                            assert mae.item() < 1.0, mae.item()
                            assert max_abs_err.item() < 1.0 + 1e-5, max_abs_err.item()
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
                            sub_label=f"{c} {dtype} {mf} {mode} {size} -> {osize} aa={aa}",
                            description=f"torch ({torch.__version__}) {tag}",
                        ).blocked_autorange(min_run_time=min_run_time)
                    )
                    # # Tensor interp via float32
                    # results.append(
                    #     benchmark.Timer(
                    #         # expected_ten = pth_downsample(tensor, mode, osize, aa)
                    #         stmt=f"fn(data, mode='{mode}', size=({osize}, {osize}), aa={aa})",
                    #         globals={
                    #             "data": tensor,
                    #             "fn": pth_downsample
                    #         },
                    #         num_threads=torch.get_num_threads(),
                    #         label="Resize",
                    #         sub_label=f"{c} {dtype} {mf} {mode} {size} -> {osize} aa={aa}",
                    #         description=f"torch ({torch.__version__}) {tag} (float)",
                    #     ).blocked_autorange(min_run_time=min_run_time)
                    # )


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