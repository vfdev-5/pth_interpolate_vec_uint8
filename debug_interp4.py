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

    check_rgb = True

    # h, w, c = 256, 256, 3
    h, w, c = 8, 28, 3
    s = w * c
    rgb = list(range(h * s))
    rgba = []
    for i in range(h):
        for j, v in enumerate(rgb[s * i:s * (i + 1)]):
            if j > 0 and j % c == 0:
                rgba.append(255)
                rgba.append(v)
            else:
                rgba.append(v)
        rgba.append(255)

    # oh, ow = 224, 224
    oh, ow = h, 24
    # oh, ow = 10, w
    # for oh in range(2, h):
    # for ow in range(2, w):
    # for ow in [7, ]:
    for _ in [1, ]:
        pil_input = PIL.Image.fromarray(np.array(rgb, dtype="uint8").reshape(h, w, 3))
        print(pil_input.size)
        pil_output = pil_input.resize([ow, oh], 2)
        print(pil_output.size)
        np_output = np.asarray(pil_output)
        expected = np_output

        # use RGB data
        if check_rgb:
            t_input = torch.tensor(rgb, dtype=torch.uint8).reshape(1, h, w, 3).permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last)
            print(t_input.shape, t_input.is_contiguous(memory_format=torch.channels_last))

            t_output = torch.nn.functional.interpolate(t_input, (oh, ow), mode="bilinear", antialias=True)
            print(t_output.shape, t_output.is_contiguous(memory_format=torch.channels_last))
            output = t_output[0, ...].permute(1, 2, 0)

            print("Compare:")
            print(expected[0, :10, :].ravel().tolist())
            print(output[0, :10, :3].ravel().tolist())
            print("")
            # np.testing.assert_allclose(expected, output[:, :, :3])

        else:
            # use RGBA data
            t_input = torch.tensor(rgba, dtype=torch.uint8).reshape(1, h, w, 4).permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last)
            print(t_input.shape, t_input.is_contiguous(memory_format=torch.channels_last))

            t_output = torch.nn.functional.interpolate(t_input, (oh, ow), mode="bilinear", antialias=True)
            print(t_output.shape, t_output.is_contiguous(memory_format=torch.channels_last))
            output = t_output[0, ...].permute(1, 2, 0)

            print("Compare:")
            print(expected[0, :10, :].ravel().tolist())
            print(output[0, :10, :3].ravel().tolist())
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
