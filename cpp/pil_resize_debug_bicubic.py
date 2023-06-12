import numpy as np
from PIL import Image

i = np.arange(3 * 225 * 225, dtype="float32").reshape(3, 225, 225)
i = i.transpose((1, 2, 0))
# i = (i % 256).astype("uint8")
i = ((i % 50) + 100).astype("uint8")

ii = Image.fromarray(i)
print(ii.size)
print(i.min(), i.max())

o = ii.resize([128, 128], Image.BICUBIC)

print(o.size)
oo = np.asarray(o)
print(oo[:7, :7, 0])
print(oo[:7, :7, 1])
print(oo[:7, :7, 2])

o2 = ii.resize([128, 128], Image.BILINEAR)
oo2 = np.asarray(o2)

# To tensor and save into files
import torch
t_i = torch.from_numpy(i).float().permute(2, 0, 1).unsqueeze(0).contiguous()
t_o1 = torch.from_numpy(oo).float().permute(2, 0, 1).unsqueeze(0).contiguous()
t_o2 = torch.from_numpy(oo2).float().permute(2, 0, 1).unsqueeze(0).contiguous()

print("tensor input:", t_i.shape, t_i.dtype)
print("tensor output bicubic:", t_o1.shape, t_o1.dtype)
print("tensor output bilinear:", t_o2.shape, t_o2.dtype)

# https://discuss.pytorch.org/t/load-tensors-saved-in-python-from-c-and-vice-versa/39435/9
from torch import nn

class TensorContainer(nn.Module):
    def __init__(self, tensor_dict):
        super().__init__()
        for key,value in tensor_dict.items():
            setattr(self, key, value)

tensor_dict = {
    'input': t_i,
    'output_bicubic': t_o1,
    'output_bilinear': t_o2,
    'info': '(3, 225, 225 | uint8) -> (3, 128, 128 | uint8) : resize bicubic/bilinear by Pillow'
}
tensors = TensorContainer(tensor_dict)
tensors = torch.jit.script(tensors)
tensors.save('expected_input_output_from_pillow.pt')
