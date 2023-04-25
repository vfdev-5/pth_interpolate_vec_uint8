import numpy as np
import PIL.Image

import torch

resampling_map = {"bilinear": PIL.Image.BILINEAR, "nearest": PIL.Image.NEAREST, "bicubic": PIL.Image.BICUBIC}


def resize(x: torch.Tensor, oh: int, ow: int):
    return torch.nn.functional.interpolate(x, (oh, ow), mode="bilinear", antialias=True)


def main():

    h, w, c = 256, 256, 3
    s = w * c
    rgb = list(range(h * s))

    oh, ow = 224, 224

    compiled_resize = torch.compile(resize)

    t_input = torch.tensor(rgb, dtype=torch.float32).reshape(1, h, w, 3).permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last)
    print(t_input.shape, t_input.is_contiguous(memory_format=torch.channels_last))

    t_output = compiled_resize(t_input, oh, ow)
    print(t_output.shape, t_output.is_contiguous(memory_format=torch.channels_last))


if __name__ == "__main__":

    torch.set_num_threads(1)

    print("")
    print(f"Torch version: {torch.__version__}")
    print(f"Torch config: {torch.__config__.show()}")
    print(f"Num threads: {torch.get_num_threads()}")
    print("")
    print("PIL version: ", PIL.__version__)

    main()
