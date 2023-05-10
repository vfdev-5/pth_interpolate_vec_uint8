import torch
from torch.utils.benchmark import Timer, Compare
from torchvision.transforms import functional as F_stable
from torchvision.transforms.v2 import functional as F_v2
from itertools import product
from functools import partial


make_arg_int = partial(torch.randint, 0, 256, dtype=torch.uint8)
shapes = (
    (3, 400, 400),
    (16, 3, 400, 400)
)
modes = [
    F_stable.InterpolationMode.NEAREST,
    F_stable.InterpolationMode.BILINEAR,
    # F_stable.InterpolationMode.BICUBIC,
]

makers = (make_arg_int, )
devices = ("cpu", "cuda")
fns = ["resize", ]
threads = (1, )

for make, shape, device, fn_name, threads, mode in product(makers, shapes, devices, fns, threads, modes):
    t1 = make(shape, device=device)

    args = ([64,],  )
    kwargs = dict(interpolation=mode, antialias=True)

    fn = getattr(F_v2, fn_name)
    sfn = torch.jit.script(fn)

    out = sfn(t1, *args, **kwargs)
    ref = fn(t1, *args, **kwargs)
    torch.testing.assert_close(ref, out, atol=1, rtol=0)

    fn = getattr(F_stable, fn_name)
    sfn = torch.jit.script(fn)

    out = sfn(t1, *args, **kwargs)
    ref = fn(t1, *args, **kwargs)
    torch.testing.assert_close(ref, out, atol=1, rtol=0)
