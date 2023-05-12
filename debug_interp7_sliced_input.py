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


# CHECK ALL COMBINATORICS FOR skip_unpacking AND skip_packing
# c = 3
# skip_unpacking = False
# is_out_CL = True

# for c in [1, 2, 3, 4]:
#     for in_mf in ["CF", "CL", "Other"]:
#         for out_mf in ["CF", "CL"]:
#             skip_unpacking = (c == 3 or c ==4) and in_mf == "CL"
#             skip_packing = ((c == 3 or c ==4) and out_mf == "CL") and (skip_unpacking or c != 3)

#             print(c, in_mf, out_mf, end=" -> \n")
#             buf_c = c if skip_unpacking else 4
#             print("\t skip_unpacking:", skip_unpacking, "-->", f"c={buf_c}")
#             print("\t skip_packing:", skip_packing, "-->", f"{buf_c} -> {c}")


def main():

    # b = 1
    b = 1

    # h, w, c = 256, 256, 3
    # h, w, c = 8, 28, 3
    h, w, c = 32, 32, 3
    # h += 10; w += 10
    h *= 3; w *= 3

    s = w * c
    rgb = list(range(b * h * s))

    # oh, ow = 224, 224
    oh, ow = 12, 12

    # oh, ow = h - 10, 24
    # oh, ow = 12, w - 10

    # oh, ow = 10, w
    # for oh in range(2, h):
    # for ow in range(2, w):
    # for ow in [7, ]:
    for _ in [1, ]:

        # t_input = torch.tensor(rgb, dtype=torch.uint8).reshape(b, h, w, 3).permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last)
        t_input = torch.tensor(rgb, dtype=torch.uint8).reshape(b, h, w, 3).permute(0, 3, 1, 2).contiguous()

        t_input = t_input[:, :, 5:-5, 5:-5]
        # t_input = t_input[:, :, ::3, ::3]

        t_input2 = t_input.contiguous()
        # t_input2 = t_input.contiguous(memory_format=torch.channels_last)
        print(
            t_input.shape,
            t_input.stride(),
            t_input.is_contiguous(),
            t_input.is_contiguous(memory_format=torch.channels_last),
            int(t_input.sum()),
        )
        print(
            t_input2.shape,
            t_input2.stride(),
            t_input2.is_contiguous(),
            t_input2.is_contiguous(memory_format=torch.channels_last),
            int(t_input2.sum()),
        )

        print("====")
        t_output = torch.nn.functional.interpolate(t_input, (oh, ow), mode="bilinear", antialias=True, align_corners=False)
        print("====")
        t_output2 = torch.nn.functional.interpolate(t_input2, (oh, ow), mode="bilinear", antialias=True, align_corners=False)
        print("====")
        print(
            t_output.shape,
            t_output.stride(),
            t_output.is_contiguous(),
            t_output.is_contiguous(memory_format=torch.channels_last)
        )
        print(
            t_output2.shape,
            t_output2.stride(),
            t_output2.is_contiguous(),
            t_output2.is_contiguous(memory_format=torch.channels_last)
        )

        output = t_output[0, ...]
        output2 = t_output2[0, ...]

        # print("Compare:")
        print(output[0, :2, :10].ravel().tolist())
        print(output2[0, :2, :10].ravel().tolist())
        torch.testing.assert_close(t_output, t_output2)
        print("")


if __name__ == "__main__":

    torch.set_num_threads(1)

    print("")
    print(f"Torch version: {torch.__version__}")
    print(f"Torch config: {torch.__config__.show()}")
    print(f"Num threads: {torch.get_num_threads()}")
    print("")
    print("PIL version: ", PIL.__version__)

    main()
