import torch
import pytest
from PIL import Image
import numpy as np

def check(img, Ho, Wo, mode):

    img = img.clone()
    img_fallback = img.clone()

    img[0, 0, 0, 0] = 1  # Will be set to 0 internally
    img_fallback[0, 0, 0, 0] = 0

    out_avx = torch.nn.functional.interpolate(img, [Ho, Wo], mode=mode, antialias=True)
    out_fallback = torch.nn.functional.interpolate(img_fallback, [Ho, Wo], mode=mode, antialias=True)
    out_float = torch.nn.functional.interpolate(img_fallback.to(torch.float), [Ho, Wo], mode=mode, antialias=True).clamp(min=0, max=255).round().to(torch.uint8)

    # assert exact match between AVX and fallback implementations
    torch.testing.assert_allclose(out_avx, out_fallback, rtol=0, atol=0)
    if mode == "linear":
        # Assert no pixel value differs by more than 1 when comparing against float implem
        torch.testing.assert_allclose(out_avx, out_float, rtol=0, atol=1)
    else:
        # assert less than "percent_threshold"% of the pixel differ by more than "max_pix_diff"
        # we also add an absolute check for very small tenosrs
        max_pix_diff = 2
        percent_threshold = 10
        abs_threshold = 2
        diff = ((out_avx.float() - out_float.float()).abs() > max_pix_diff).float()
        percent_cond = diff.mean() <= percent_threshold / 100
        abs_cond = diff.sum() <= abs_threshold
        assert percent_cond or abs_cond

    if img.shape[0] == 1 and img.shape[1] == 3:
        img_pil = Image.fromarray(img[0].permute(1, 2, 0).numpy())
        mode = {"bilinear": Image.BILINEAR, "bicubic": Image.BICUBIC}[mode]
        out_pil = torch.tensor(np.array(img_pil.resize((Wo, Ho), mode))).permute(2, 0, 1)
        torch.testing.assert_allclose(out_avx[0], out_pil, rtol=0, atol=0)

@pytest.mark.parametrize("Hi, Wi, Ho, Wo", (
    (271, 268, 224, 224),
    (256, 128, 512, 256),
    (68, 49, 1549, 2890),
    (10, 15, 512, 320),
    (4, 8, 8, 4),
    (2, 2, 4, 4),
    (10, 15, 20, 15),
    (10, 15, 10, 20),
))
@pytest.mark.parametrize("batch_size", (1, 4))
@pytest.mark.parametrize("C", (range(1, 5)))
@pytest.mark.parametrize("mode", ("bilinear", "bicubic"))
@pytest.mark.parametrize("reverse", (False, True))
@pytest.mark.parametrize("channels_last", (True, False))
def test_lol(Hi, Wi, Ho, Wo, batch_size, C, mode, reverse, channels_last):
    if reverse:
        Hi, Wi, Ho, Wo = Ho, Wo, Hi, Wi

    if Hi == Ho or Wi == Wo:
        pytest.xfail("Segfault lololol")

    img = torch.randint(0, 256, (batch_size, C, Hi, Wi), dtype=torch.uint8)
    if channels_last:
        img = img.contiguous(memory_format=torch.channels_last)

    check(img, Hi, Wo, mode)  # horizontal
    check(img, Ho, Wi, mode)  # vertical
    check(img, Ho, Wo, mode)  # both