import numpy as np
import PIL.Image

import torch
import torch.utils.benchmark as benchmark


resampling_map = {"bilinear": PIL.Image.BILINEAR, "nearest": PIL.Image.NEAREST, "bicubic": PIL.Image.BICUBIC}


def main():

    out_size = 2
    resample = "bilinear"
    align_corners = None if resample == "nearest" else False

    torch.manual_seed(12)

    # for mf in ["channels_first", "channels_last"]:
    for mf in ["channels_last", ]:
    # for mf in ["channels_first", ]:
        for c, dtype in [(3, torch.uint8), ]:
        # for c, dtype in [(4, torch.uint8), ]:
        # for c, dtype in [(3, torch.float), ]:
            for size in [8, ]:

                tensor = torch.arange(c * (size + 1) * size, dtype=dtype).reshape(c, size + 1, size)
                if dtype == torch.uint8 and c == 3:
                    np_array = tensor.clone().permute(1, 2, 0).contiguous().numpy()
                elif dtype in (torch.float32, torch.int):
                    np_array = tensor[0, ...].clone().numpy()
                else:
                    np_array = None

                pil_img = None
                if np_array is not None:
                    pil_img = PIL.Image.fromarray(np_array)

                tensor = tensor[None, ...]

                if mf == "channels_last":
                    tensor = tensor.contiguous(memory_format=torch.channels_last)

                print("Memory format:", mf)
                output = torch.nn.functional.interpolate(
                    tensor, mode=resample, size=(out_size + 2, out_size), align_corners=align_corners, antialias=True
                )
                output = output[0, ...]
                print("output: \n", output)

                if pil_img is not None:
                    expected = pil_img.resize((out_size, out_size + 2), resample=resampling_map[resample])
                    expected_pil = torch.from_numpy(np.asarray(expected)).clone()
                    if dtype == torch.uint8:
                        expected_pil = expected_pil.permute(2, 0, 1)
                    elif dtype in (torch.float32, torch.int):
                        expected_pil = expected_pil[None, ...]

                    if expected_pil is not None:
                        expected_pil = expected_pil.contiguous()

                    print("expected: \n", expected_pil)


if __name__ == "__main__":

    torch.set_num_threads(1)

    print("")
    print(f"Torch version: {torch.__version__}")
    print(f"Torch config: {torch.__config__.show()}")
    print(f"Num threads: {torch.get_num_threads()}")
    print("")
    print("PIL version: ", PIL.__version__)

    main()
