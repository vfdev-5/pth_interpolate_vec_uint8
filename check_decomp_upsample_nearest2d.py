import torch
# from torch._subclasses.fake_tensor import FakeTensorMode


# device = "cpu"
device = "cuda"


x = torch.ones(2, 5, 32, 32, dtype=torch.float32, device=device).contiguous(memory_format=torch.channels_last)
print("\n-", type(x), x.shape, x.stride(), x.device, x.dtype)
expected_output = torch._C._nn.upsample_nearest2d(x, [12, 12])
print(
    "=>",
    type(expected_output),
    expected_output.shape,
    expected_output.is_contiguous(),
    expected_output.is_contiguous(memory_format=torch.channels_last),
)

# with FakeTensorMode():
#     x = torch.empty_strided(x.shape, x.stride(), dtype=torch.float32, device=device)
#     print("\n-", type(x), x.shape, x.stride(), x.device, x.dtype)
#     decomp_output = torch.ops.aten.upsample_nearest2d.default(x, [12, 12])
#     print(
#         "=>",
#         type(decomp_output),
#         decomp_output.shape,
#         decomp_output.is_contiguous(),
#         decomp_output.is_contiguous(memory_format=torch.channels_last),
#     )


x = torch.empty_strided(x.shape, x.stride(), dtype=torch.float32, device=device)
print("\n-", type(x), x.shape, x.stride(), x.device, x.dtype)

fn = torch.compile(torch.ops.aten.upsample_nearest2d.default)
decomp_output = fn(x, [12, 12])
print(
    "=>",
    type(decomp_output),
    decomp_output.shape,
    decomp_output.is_contiguous(),
    decomp_output.is_contiguous(memory_format=torch.channels_last),
)


print()

#####

x = torch.ones(2, 3, 32, 32, dtype=torch.float32, device=device).contiguous(memory_format=torch.channels_last)
print("\n-", type(x), x.shape, x.stride(), x.device, x.dtype)
expected_output = torch._C._nn.upsample_nearest2d(x, [12, 12])
print(
    "=>",
    type(expected_output),
    expected_output.shape,
    expected_output.is_contiguous(),
    expected_output.is_contiguous(memory_format=torch.channels_last),
)

# with FakeTensorMode():
#     x = torch.empty_strided(x.shape, x.stride(), dtype=torch.float32, device=device)
#     print("\n-", type(x), x.shape, x.stride(), x.device, x.dtype)
#     decomp_output = torch.ops.aten.upsample_nearest2d.default(x, [12, 12])
#     print(
#         "=>",
#         type(decomp_output),
#         decomp_output.shape,
#         decomp_output.is_contiguous(),
#         decomp_output.is_contiguous(memory_format=torch.channels_last),
#     )

x = torch.empty_strided(x.shape, x.stride(), dtype=torch.float32, device=device)
print("\n-", type(x), x.shape, x.stride(), x.device, x.dtype)

fn = torch.compile(torch.ops.aten.upsample_nearest2d.default)
decomp_output = fn(x, [12, 12])
print(
    "=>",
    type(decomp_output),
    decomp_output.shape,
    decomp_output.is_contiguous(),
    decomp_output.is_contiguous(memory_format=torch.channels_last),
)

print()
