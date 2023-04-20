# Debug failing case:
# mf/size/dtype/c/osize/aa/mode/ac :  channels_last 256 torch.uint8 4 32 True bilinear True  -> get expected from file
# Traceback (most recent call last):
#   File "verif_interp2.py", line 181, in <module>
#     fire.Fire(main)
#   File "/usr/local/lib/python3.8/dist-packages/fire/core.py", line 141, in Fire
#     component_trace = _Fire(component, args, parsed_flag_args, context, name)
#   File "/usr/local/lib/python3.8/dist-packages/fire/core.py", line 475, in _Fire
#     component, remaining_args = _CallAndUpdateTrace(
#   File "/usr/local/lib/python3.8/dist-packages/fire/core.py", line 691, in _CallAndUpdateTrace
#     component = fn(*varargs, **kwargs)
#   File "verif_interp2.py", line 160, in main
#     assert mae.item() < 1.0, mae.item()
# AssertionError: 1.1787109375


import numpy as np
import PIL.Image

import torch

resampling_map = {"bilinear": PIL.Image.BILINEAR, "nearest": PIL.Image.NEAREST, "bicubic": PIL.Image.BICUBIC}


def main():

    # h, w, c = 256, 256, 3
    # h, w, c = 8, 28, 3
    # h, w, c = 4, 20, 3
    h, w, c = 2, 18, 3
    s = w * c
    rgb = list(range(h * s))

    # oh, ow = 224, 224
    # oh, ow = h, 24
    # oh, ow = 2, w
    oh, ow = h, 5
    # oh, ow = 10, w
    # for oh in range(2, h):
    # for ow in range(2, w):
    # for ow in [7, ]:
    for _ in [1, ]:

        t_input = torch.tensor(rgb, dtype=torch.uint8).reshape(1, h, w, 3).permute(0, 3, 1, 2).contiguous(memory_format=torch.contiguous_format)
        print(t_input.shape, t_input.dtype, t_input.is_contiguous())

        t_output = torch.nn.functional.interpolate(t_input, (oh, ow), mode="bilinear", antialias=True)
        print(t_output.shape, t_output.dtype, t_output.is_contiguous())
        output = t_output[0, ...]

        print("Compare:")
        # print(expected[0, :10, :].ravel().tolist())
        print(output[:, :, :].ravel().tolist())
        print("")
        # np.testing.assert_allclose(expected, output[:, :, :3])


if __name__ == "__main__":

    torch.set_num_threads(1)

    print("")
    print(f"Torch version: {torch.__version__}")
    print(f"Torch config: {torch.__config__.show()}")
    print(f"Num threads: {torch.get_num_threads()}")
    print("")
    print("PIL version: ", PIL.__version__)

    main()
