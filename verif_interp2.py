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


def store_expected(tensor, output, output_path, seed, mf, c, dtype, size, mode, osize, aa, ac, non_contig):
    if not output_path.exists():
        output_path.mkdir(parents=True)
    bs = len(tensor)
    filepath = output_path / f"{seed}_{bs}_{mf}_{c}_{dtype}_{size[0]}_{size[1]}_{mode}_{osize[0]}_{osize[1]}_{aa}_{ac}_{non_contig}.pt"
    torch.save(
        {"input": tensor, "output": output, "torch_version": torch.__version__},
        filepath
    )


def get_expected(tensor, output_path, seed, mf, c, dtype, size, mode, osize, aa, ac, non_contig):
    bs = len(tensor)
    filepath = output_path / f"{seed}_{bs}_{mf}_{c}_{dtype}_{size[0]}_{size[1]}_{mode}_{osize[0]}_{osize[1]}_{aa}_{ac}_{non_contig}.pt"
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


def test_consistency_or_record(
    expected_pil, tensor, c, size, mf, dtype, mode, osize, aa, ac, is_ref, output_path, seed,
    exact_match=True,
    record_path={torch.float32: "native", torch.uint8: "native"},
    non_contig=False,
):
    # Tested op -> output
    if is_ref:

        if non_contig is not False:
            assert tensor.ndim == 4, tensor.ndim
            if mf == "channels_first":
                tensor = tensor.contiguous()
            elif mf == "channels_last":
                tensor = tensor.contiguous(memory_format=torch.channels_last)
            else:
                raise RuntimeError(
                    "Unknown mf:", mf, " | ",
                    c, size, mf, dtype, mode, osize, aa, ac, is_ref, output_path, seed
                )

        # When there is no reference code, we can use float32 intermediate dtype
        code_path = record_path.get(dtype, "force_float")
        if code_path == "native":
            print("take 'native' code path", end=" ")
            output = pth_downsample(tensor, mode, osize, aa, ac)
        elif code_path == "force_float":
            print("take 'force_float' code path", end=" ")
            output = pth_downsample_force_float(tensor, mode, osize, aa, ac)
        else:
            raise ValueError(f"Unknown value for record_path on {code_path}, record_path={record_path}")
    else:
        output = pth_downsample(tensor, mode, osize, aa, ac)

    # Expected result:
    if is_ref:
        if output is not None:
            print(" -> store output")
            store_expected(tensor, output, output_path, seed, mf, c, dtype, size, mode, osize, aa, ac, non_contig)
        else:
            print("")
        return

    print(" -> get expected from file")
    expected_ten = get_expected(tensor, output_path, seed, mf, c, dtype, size, mode, osize, aa, ac, non_contig)
    print("---")
    if not ac and expected_pil is not None:
        abs_diff = torch.abs(expected_pil.float() - output.float())
        mae = torch.mean(abs_diff)
        max_abs_err = torch.max(abs_diff)

        if mode == "bilinear":
            assert mae.item() < 1.0, mae.item()
            assert max_abs_err.item() < 1.0 + 1e-5, max_abs_err.item()

    expected_mf = torch.channels_last if expected_ten.is_contiguous(memory_format=torch.channels_last) else torch.contiguous_format
    output_mf = torch.channels_last if output.is_contiguous(memory_format=torch.channels_last) else torch.contiguous_format
    assert expected_mf == output_mf, (expected_mf, output_mf)

    abs_diff = torch.abs(expected_ten.float() - output.float())
    mae = torch.mean(abs_diff)
    max_abs_err = torch.max(abs_diff)

    if mode == "bilinear":

        if exact_match:
            torch.testing.assert_close(expected_ten, output)
        else:
            assert mae.item() < 1.0, mae.item()
            max_abs_err_tol = 2.0
            m = abs_diff > 1.5
            assert max_abs_err.item() < max_abs_err_tol + 1e-5, \
                (max_abs_err.item(), expected_ten.float()[m], output.float()[m])


def main(output_path: str, is_ref: bool = False):

    output_path = Path(output_path)

    if is_ref and output_path.exists():
        raise RuntimeError("Please provide non-exising folder if --is_ref flag is used")


    for batch_size in [1, 5]:
        for non_contig in [False, "sliced", "restrided"]:
            for ac in [True, False]:
                for mf in ["channels_last", "channels_first", ]:
                    for c, dtype in [
                        (1, torch.uint8),
                        (2, torch.uint8),
                        (3, torch.uint8),
                        (4, torch.uint8),

                        (1, torch.float32),
                        (2, torch.float32),
                        (3, torch.float32),
                        (4, torch.float32),
                    ]:
                        for size in [256, (256, 299), (299, 321)]:
                            if isinstance(size, int):
                                size = [size, size]

                            if non_contig is not False:
                                if non_contig == "sliced":
                                    size = [size[0] + 50, size[1] + 50]
                                elif non_contig == "restrided":
                                    size = [size[0] * 2, size[1] * 2]
                                else:
                                    raise ValueError("Unknown non_contig value '{non_contig}'")

                            for osize, aa, mode in [
                                (32, True, "bilinear"),
                                (32, False, "bilinear"),
                                ((35, 38), True, "bilinear"),
                                ((35, 38), False, "bilinear"),
                                (224, True, "bilinear"),
                                (224, False, "bilinear"),
                                ((227, 231), True, "bilinear"),
                                ((227, 231), False, "bilinear"),
                                (320, True, "bilinear"),
                                (320, False, "bilinear"),
                                ((323, 327), True, "bilinear"),
                                ((323, 327), False, "bilinear"),
                            ]:
                                if isinstance(osize, int):
                                    osize = [osize, osize + 1]

                                print("batch_size/non_contig/mf/size/dtype/c/osize/aa/mode/ac : ", batch_size, non_contig, mf, size, dtype, c, osize, aa, mode, ac, end=" ")

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

                                if non_contig is not False:
                                    if non_contig == "sliced":
                                        tensor = tensor[:, 25:-25, 25:-25]
                                    elif non_contig == "restrided":
                                        tensor = tensor[:, ::2, ::2]
                                    else:
                                        raise ValueError("Unknown non_contig value '{non_contig}'")

                                expected_pil = None
                                if dtype == torch.uint8 and c == 3 and aa:
                                    np_array = tensor.clone().permute(1, 2, 0).contiguous().numpy()
                                    pil_img = PIL.Image.fromarray(np_array)
                                    pil_img = pil_img.resize(osize[::-1], resample=resampling_map[mode])
                                    expected_pil = torch.from_numpy(np.asarray(pil_img)).clone().permute(2, 0, 1).contiguous()

                                memory_format = torch.channels_last if mf == "channels_last" else torch.contiguous_format

                                if batch_size == 1:
                                    tensor = tensor[None, ...].contiguous(memory_format=memory_format)
                                else:
                                    new_shape = (batch_size, ) + tensor.shape
                                    tensor = tensor[None, ...].expand(new_shape).contiguous(memory_format=memory_format)

                                print(".", end=" ")
                                test_consistency_or_record(
                                    expected_pil, tensor, c, size, mf, dtype, mode, osize, aa, ac, is_ref, output_path, seed,
                                    exact_match=True, non_contig=non_contig
                                )

                                if batch_size == 1:
                                    # Check specifically squeeze/unsqueeze on batch dimension
                                    # There is an inconsistency with interpolate on output mem format
                                    # if input is unsqueezed 3D CL tensor, output is 4D CF tensor
                                    tensor = tensor[0, ...]
                                    tensor = tensor[None, ...]

                                    print("..", end=" ")
                                    # We override record_path as native code path for uint8 is buggy for nightly before
                                    # https://github.com/pytorch/pytorch/pull/100258
                                    record_path = {torch.float32: "native", torch.uint8: "force_float"}
                                    test_consistency_or_record(
                                        expected_pil, tensor, c, size, mf, dtype, mode, osize, aa, ac, is_ref, output_path / "sq_unsq", seed,
                                        exact_match=False, record_path=record_path,
                                        non_contig=non_contig
                                    )


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
