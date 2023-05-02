# python -u check_compile_interp_sq_unsq_CL.py
#
# TORCH_COMPILE_DEBUG=1 python -u check_compile_interp_sq_unsq_CL.py
#

import torch
print(torch.__version__)


@torch.compile()
def fn(x, mode, aa, size):
    out = torch.nn.functional.interpolate(x, size=size, mode=mode, antialias=aa)
    return out


aa = False
mode = "bilinear"
dtype = torch.uint8
mf = torch.channels_last
device = "cpu"
size = 224

print("-", size, aa, mode, mf, device, dtype)
x = torch.randint(0, 256, size=(1, 3, 256, 256), dtype=dtype, device=device).contiguous(memory_format=mf)

x = x[0, ...]
x = x[None, ...]

y = fn(x, mode, aa, size)

input_mem_format = "CL" if x.is_contiguous(memory_format=torch.channels_last) else "CF"
if input_mem_format == "CF":
    assert x.is_contiguous(memory_format=torch.contiguous_format)

output_mem_format = "CL" if y.is_contiguous(memory_format=torch.channels_last) else "CF"
if output_mem_format == "CF":
    assert y.is_contiguous(memory_format=torch.contiguous_format)

if input_mem_format != output_mem_format:
    print(f"- {mf}, {device}, {dtype}: {output_mem_format} != {input_mem_format}\n")
