import numpy as np
import PIL
from PIL import Image


print("Pillow version: ", PIL.__version__)

def reduce_image_range(min, max):
    def wrapper(x):
        return (x % (max - min)) + min
    return wrapper


for interpolation in [Image.BICUBIC, Image.BILINEAR]:
    print("\n--- Interpolation mode: ", interpolation)
    for fn in (reduce_image_range(0, 255), reduce_image_range(50, 200)):

        input = np.arange(225 * 225, dtype="float32").reshape(225, 225)
        input = fn(input)
        input_ui8 = input.astype("uint8")

        pil_ui8 = Image.fromarray(input_ui8)
        pil_f32 = Image.fromarray(input)

        print(pil_ui8.size, pil_ui8.mode, input.min(), input.max())
        print(pil_f32.size, pil_f32.mode, input_ui8.min(), input_ui8.max())

        pil_out_ui8 = pil_ui8.resize([128, 128], interpolation)
        pil_out_f32 = pil_f32.resize([128, 128], interpolation)

        out_f32 = np.asarray(pil_out_f32)
        out_ui8 = np.asarray(pil_out_ui8).astype("float32")

        abs_diff = np.abs(out_f32 - out_ui8)
        print("Max Abs Error: ", abs_diff.max())
        print("Number of non-equal values: ", (abs_diff > 0).sum())
