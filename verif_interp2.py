from pathlib import Path
import numpy as np
import PIL.Image

import torch
import torch.utils.benchmark as benchmark

import fire


def pth_downsample(img, mode, size, aa=True, ac=False):

    align_corners = ac
    if mode == "nearest":
        align_corners = None

    out = torch.nn.functional.interpolate(
        img, size=size,
        mode=mode,
        align_corners=align_corners,
        antialias=aa,
    )
    return out


def pth_downsample_force_float(img, mode, size, aa=True, ac=False):

    align_corners = ac
    if mode == "nearest":
        align_corners = None

    out = torch.nn.functional.interpolate(
        img.float(), size=size,
        mode=mode,
        align_corners=align_corners,
        antialias=aa,
    )

    if out.dtype in (torch.uint8, torch.int8, torch.int32, torch.long):
        out = out.round()

    return out.to(img.dtype)


def store_expected(tensor, output, output_path, seed, mf, c, dtype, size, mode, osize, aa, ac):
    output_path = Path(output_path)
    if not output_path.exists():
        output_path.mkdir(parents=True)
    filepath = output_path / f"{seed}_{mf}_{c}_{dtype}_{size[0]}_{size[1]}_{mode}_{osize[0]}_{aa}_{ac}.pt"
    torch.save(
        {"input": tensor, "output": output, "torch_version": torch.__version__},
        filepath
    )


def get_expected(tensor, output_path, seed, mf, c, dtype, size, mode, osize, aa, ac):
    output_path = Path(output_path)
    filepath = output_path / f"{seed}_{mf}_{c}_{dtype}_{size[0]}_{size[1]}_{mode}_{osize[0]}_{aa}_{ac}.pt"
    obj = torch.load(filepath)
    inpt = obj["input"]
    torch.testing.assert_close(inpt, tensor)
    return obj["output"]


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


def main(output_path: str, is_ref: bool):

    for ac in [True, False]:
        for mf in ["channels_last", "channels_first"]:
            for c, dtype in [
                (3, torch.uint8),
                (1, torch.uint8),
                (2, torch.uint8),
                (4, torch.uint8),

                (3, torch.float32),
                (1, torch.float32),
                (2, torch.float32),
                (4, torch.float32),
            ]:
                for size in [256, (256, 299)]:
                    if isinstance(size, int):
                        size = [size, size]

                    for osize, aa, mode in [
                        (32, True, "bilinear"),
                        (32, False, "bilinear"),
                        (224, True, "bilinear"),
                        (224, False, "bilinear"),
                        (320, True, "bilinear"),
                        (320, False, "bilinear"),
                    ]:
                        osize = (osize, osize + 1)
                        print("mf/size/dtype/c/osize/aa/mode/ac : ", mf, size, dtype, c, osize, aa, mode, ac, end=" ")

                        seed = 115
                        torch.manual_seed(seed)

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
                        if dtype == torch.uint8 and c == 3 and aa:
                            np_array = tensor.clone().permute(1, 2, 0).contiguous().numpy()
                            pil_img = PIL.Image.fromarray(np_array)
                            pil_img = pil_img.resize(osize[::-1], resample=resampling_map[mode])
                            expected_pil = torch.from_numpy(np.asarray(pil_img)).clone().permute(2, 0, 1).contiguous()

                        memory_format = torch.channels_last if mf == "channels_last" else torch.contiguous_format
                        tensor = tensor[None, ...].contiguous(memory_format=memory_format)

                        # Tested op -> output
                        if is_ref:
                            if dtype == torch.float:
                                output = pth_downsample(tensor, mode, osize, aa, ac)
                            else:
                                output = pth_downsample_force_float(tensor, mode, osize, aa, ac)
                        else:
                            output = pth_downsample(tensor, mode, osize, aa, ac)

                        # Expected result:
                        if is_ref:
                            if output is not None:
                                print(" -> store output")
                                store_expected(tensor, output, output_path, seed, mf, c, dtype, size, mode, osize, aa, ac)
                            else:
                                print("")
                            continue

                        print(" -> get expected from file")
                        expected_ten = get_expected(tensor, output_path, seed, mf, c, dtype, size, mode, osize, aa, ac)

                        if not ac and expected_pil is not None:
                            abs_diff = torch.abs(expected_pil.float() - output.float())
                            mae = torch.mean(abs_diff)
                            max_abs_err = torch.max(abs_diff)

                            if mode == "bilinear":
                                assert mae.item() < 1.0, mae.item()
                                assert max_abs_err.item() < 1.0 + 1e-5, max_abs_err.item()

                        abs_diff = torch.abs(expected_ten.float() - output.float())
                        mae = torch.mean(abs_diff)
                        max_abs_err = torch.max(abs_diff)

                        if mode == "bilinear":
                            assert mae.item() < 1.0, mae.item()
                            m = abs_diff > 1.5
                            assert max_abs_err.item() < 1.0 + 1e-5, (max_abs_err.item(), expected_ten.float()[m], output.float()[m])

                            # if not (mae.item() < 1.0 and max_abs_err.item() < 1.0 + 1e-5):
                            #     print("FAILED")
                            # else:
                            #     print("PASSED")


if __name__ == "__main__":

    import os
    if not ("OMP_NUM_THREADS" in os.environ):
        torch.set_num_threads(1)

    print("")
    print(f"Torch version: {torch.__version__}")
    print(f"Torch config: {torch.__config__.show()}")
    print(f"Num threads: {torch.get_num_threads()}")
    print("")
    print("PIL version: ", PIL.__version__)

    fire.Fire(main)
