import torch
print(torch.__version__)


def t1(x, mode, aa):
    do_squeeze = False
    if x.dim() == 3:
        do_squeeze = True
        x = x[None, ...]
    out = torch.nn.functional.interpolate(
        x, size=224, mode=mode, antialias=aa
    )
    if do_squeeze:
        out = out[0, ...]
    return out

def t2(x, mode):
    do_squeeze = False
    if x.dim() == 4:
        do_squeeze = True
        x = x[None, ...]
    out = torch.nn.functional.interpolate(
        x, size=224, mode=mode
    )
    if do_squeeze:
        out = out[0, ...]
    return out


c_t1 = torch.compile(t1)


# for aa in [False, True]:
for aa in [True, False]:
    for mode in ["bilinear", "nearest", "nearest-exact", "bicubic"]:

        if "nearest" in mode and aa:
            continue

        for dtype in [torch.uint8, torch.float32, torch.float64]:

            if mode == "bicubic" and dtype == torch.uint8:
                continue

            for mf in [torch.channels_last, torch.contiguous_format]:

                for device in ["cpu", "cuda"]:

                    if device == "cuda" and not torch.cuda.is_available():
                        continue

                    if device == "cuda" and dtype == torch.uint8:
                        continue

                    print("-", aa, mode, mf, device, dtype)
                    x = torch.randint(0, 256, size=(1, 3, 256, 256), dtype=dtype, device=device).contiguous(memory_format=mf)

                    y_ref = t1(x, mode, aa)

                    _ = c_t1(x, mode, aa)

                    input_mem_format = "CL" if x.is_contiguous(memory_format=torch.channels_last) else "CF"
                    if input_mem_format == "CF":
                        assert x.is_contiguous(memory_format=torch.contiguous_format)

                    output_mem_format = "CL" if y_ref.is_contiguous(memory_format=torch.channels_last) else "CF"
                    if output_mem_format == "CF":
                        assert y_ref.is_contiguous(memory_format=torch.contiguous_format)

                    # assert input_mem_format == output_mem_format, f"1 {mf}, {device}, {dtype}: {output_mem_format} != {input_mem_format}"
                    if input_mem_format != output_mem_format:
                        print(f"1 {mf}, {device}, {dtype}: {output_mem_format} != {input_mem_format}\n")

                    x = x[0, ...]
                    x = x[None, ...]

                    # # Restride x
                    # shape = x.shape
                    # strides = list(x.stride())
                    # strides[0] = shape[1] * shape[2] * shape[3]
                    # x = x.as_strided(shape, strides)

                    y = t1(x, mode, aa)

                    input_mem_format = "CL" if x.is_contiguous(memory_format=torch.channels_last) else "CF"
                    if input_mem_format == "CF":
                        assert x.is_contiguous(memory_format=torch.contiguous_format)

                    output_mem_format = "CL" if y.is_contiguous(memory_format=torch.channels_last) else "CF"
                    if output_mem_format == "CF":
                        assert y.is_contiguous(memory_format=torch.contiguous_format)

                    # assert input_mem_format == output_mem_format, f"2 {mf}, {device}, {dtype}: {output_mem_format} != {input_mem_format}"
                    if input_mem_format != output_mem_format:
                        print(f"2 {mf}, {device}, {dtype}: {output_mem_format} != {input_mem_format}\n")

                    # torch.testing.assert_close(y_ref, y)

                    y = c_t1(x, mode, aa)

                    input_mem_format = "CL" if x.is_contiguous(memory_format=torch.channels_last) else "CF"
                    if input_mem_format == "CF":
                        assert x.is_contiguous(memory_format=torch.contiguous_format)

                    output_mem_format = "CL" if y.is_contiguous(memory_format=torch.channels_last) else "CF"
                    if output_mem_format == "CF":
                        assert y.is_contiguous(memory_format=torch.contiguous_format)

                    # assert input_mem_format == output_mem_format, f"2 {mf}, {device}, {dtype}: {output_mem_format} != {input_mem_format}"
                    if input_mem_format != output_mem_format:
                        print(f"3 {mf}, {device}, {dtype}: {output_mem_format} != {input_mem_format}\n")



for mode in ["trilinear", "nearest", "nearest-exact", ]:

    for dtype in [torch.float32, torch.float64]:

        for mf in [torch.channels_last_3d, torch.contiguous_format]:

            for device in ["cpu", "cuda"]:

                if device == "cuda" and not torch.cuda.is_available():
                    continue

                if device == "cuda" and dtype == torch.uint8:
                    continue

                print("-", mode, mf, device, dtype)
                x = torch.randint(0, 256, size=(1, 2, 3, 256, 256), dtype=dtype, device=device).contiguous(memory_format=mf)

                y_ref = t2(x, mode)

                input_mem_format = "CL" if x.is_contiguous(memory_format=torch.channels_last_3d) else "CF"
                if input_mem_format == "CF":
                    assert x.is_contiguous(memory_format=torch.contiguous_format)

                output_mem_format = "CL" if y_ref.is_contiguous(memory_format=torch.channels_last_3d) else "CF"
                if output_mem_format == "CF":
                    assert y_ref.is_contiguous(memory_format=torch.contiguous_format)

                # assert input_mem_format == output_mem_format, f"1 {mf}, {device}, {dtype}: {output_mem_format} != {input_mem_format}"
                if input_mem_format != output_mem_format:
                    print(f"1 {mf}, {device}, {dtype}: {output_mem_format} != {input_mem_format}\n")

                x = x[0, ...]
                x = x[None, ...]

                y = t2(x, mode)

                input_mem_format = "CL" if x.is_contiguous(memory_format=torch.channels_last_3d) else "CF"
                if input_mem_format == "CF":
                    assert x.is_contiguous(memory_format=torch.contiguous_format)

                output_mem_format = "CL" if y.is_contiguous(memory_format=torch.channels_last_3d) else "CF"
                if output_mem_format == "CF":
                    assert y.is_contiguous(memory_format=torch.contiguous_format)

                # assert input_mem_format == output_mem_format, f"2 {mf}, {device}, {dtype}: {output_mem_format} != {input_mem_format}"
                if input_mem_format != output_mem_format:
                    print(f"2 {mf}, {device}, {dtype}: {output_mem_format} != {input_mem_format}\n")

                # torch.testing.assert_close(y_ref, y)
