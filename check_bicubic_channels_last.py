import torch

i = torch.rand(1, 3, 32, 32).contiguous(memory_format=torch.channels_last)
assert i.is_contiguous(memory_format=torch.channels_last)
o = torch.nn.functional.interpolate(i, size=(4, 4), mode="bicubic")
assert o.is_contiguous(memory_format=torch.channels_last), f"Should be channels last but given channels first ({o.is_contiguous(memory_format=torch.contiguous_format)})"
