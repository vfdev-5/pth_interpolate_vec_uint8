import torch

device = "cuda"

aten_upsample_nearest2d = torch._C._nn.upsample_nearest2d
dec_upsample_nearest2d = torch.compile(torch.ops.aten.upsample_nearest2d.default, backend="cudagraphs")

aten_upsample_bilinear2d = torch._C._nn.upsample_bilinear2d
dec_upsample_bilinear2d = torch.compile(torch.ops.aten.upsample_bilinear2d.default, backend="cudagraphs")


for x in [
    torch.ones(2, 5, 32, 32, dtype=torch.float32, device=device).contiguous(memory_format=torch.channels_last),
    torch.ones(2, 3, 32, 32, dtype=torch.float32, device=device).contiguous(memory_format=torch.channels_last),
    torch.ones(2, 5, 32, 32, dtype=torch.float32, device=device),
    torch.ones(2, 3, 32, 32, dtype=torch.float32, device=device),
]:

    for aten_fn, decomp_fn in [
        (aten_upsample_nearest2d, dec_upsample_nearest2d),
        (aten_upsample_bilinear2d, dec_upsample_bilinear2d),
    ]:

        args = ()
        if aten_fn == aten_upsample_bilinear2d:
            args += (True, )

        print("Input:", x.shape, x.stride(), x.device, x.dtype)
        expected_output = aten_fn(x, [12, 12], *args)
        print(
            "Output, Aten:",
            aten_fn,
            expected_output.shape,
            expected_output.is_contiguous(),
            expected_output.is_contiguous(memory_format=torch.channels_last),
        )

        decomp_output = decomp_fn(x, [12, 12], *args)
        print(
            "Output, Decomposition:",
            decomp_fn,
            decomp_output.shape,
            decomp_output.is_contiguous(),
            decomp_output.is_contiguous(memory_format=torch.channels_last),
        )

        print()
        assert expected_output.is_contiguous() == decomp_output.is_contiguous()
        assert expected_output.is_contiguous(memory_format=torch.channels_last) == decomp_output.is_contiguous(memory_format=torch.channels_last)
