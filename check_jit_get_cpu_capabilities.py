import torch


torch.jit.script(torch.backends.cpu.get_cpu_capability)


def func() -> str:
    return torch.backends.cpu.get_cpu_capability()


sfunc = torch.jit.script(func)

print(sfunc())


cfunc = torch.compile(func)

print(cfunc())


