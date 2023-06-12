import numpy as np
import PIL.Image

import torch
import cv2

resampling_map = {"bilinear": PIL.Image.BILINEAR, "nearest": PIL.Image.NEAREST, "bicubic": PIL.Image.BICUBIC}


def main():

    # b = 1
    b = 1

    # h, w, c = 256, 256, 3
    # h, w, c = 8, 28, 3
    h, w, c = 31, 9, 3

    s = w * c
    rgb = list(range(b * h * s))

    # oh, ow = 224, 224
    oh, ow = 29, 9

    # oh, ow = h - 10, 24
    # oh, ow = 12, w - 10

    # oh, ow = 10, w
    # for oh in range(2, h):
    # for ow in range(2, w):
    # for ow in [7, ]:
    for _ in [1, ]:

        t = torch.tensor(rgb).to(dtype=torch.uint8).reshape(b, h, w, c)
        # t_input = t.permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last)
        t_input = t.permute(0, 3, 1, 2).contiguous()

        np_input = t[0, ...].numpy()

        t_input2 = t_input.double()
        # t_input2 = t_input.float()
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
        t_output = torch.nn.functional.interpolate(t_input, (oh, ow), mode="bilinear", antialias=False, align_corners=False)
        print("====")
        t_output2 = torch.nn.functional.interpolate(t_input2, (oh, ow), mode="bilinear", antialias=False, align_corners=False).round().byte()
        print("====")
        np_output3 = cv2.resize(np_input, dsize=(ow, oh), interpolation=cv2.INTER_LINEAR)

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

        print("Compare:")
        print("")
        for i in [12, 13, 14, 15]:
            print(i)
            print("uint8:", output[0, i, :])
            print("diff:", (output[0, i, :].long() - output2[0, i, :].long()))
            print("cv2:", np_output3[i, :, 0])
            print("float:", output2[0, i, :])
        print("")
        print(output[0, :, :].ravel().tolist())
        print(np_output3[:, :, 0].ravel().tolist())

        # print((output[1, :, :] - output2[1, :, :]).ravel().tolist())
        print("")
        print(output[2, :, :].ravel().tolist())
        print((output[2, :, :] - output2[2, :, :]).ravel().tolist())

        torch.testing.assert_close(t_output, t_output2, rtol=0.0, atol=1.0)
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
