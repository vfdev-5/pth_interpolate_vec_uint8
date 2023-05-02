import PIL.Image

import torch
import torchvision
import torchvision.transforms as T

img = PIL.Image.open("test/assets/encode_jpeg/grace_hopper_517x606.jpg")
tensor_from_pil = T.PILToTensor()(img)[None]
assert tensor_from_pil.is_contiguous(memory_format=torch.channels_last)