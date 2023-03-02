import operator_benchmark as op_bench
import torch

"""Microbenchmarks for interpolate operator."""


class InterpolateBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, input_size, output_size, channels_last=False, mode='linear', antialias=False, dtype=torch.float):

        input_image = torch.randint(0, 256, size=input_size, dtype=torch.uint8, device='cpu')

        if channels_last:
            input_image = input_image.contiguous(memory_format=torch.channels_last)

        self.inputs = {
            "input_image": input_image,
            "output_size": output_size,
            "mode": mode,
            "antialias": antialias,
            "dtype":dtype,
        }

        self.set_module_name("interpolate")

    def forward(self, input_image, output_size, mode, antialias, dtype):
        if dtype == torch.float:
            input_image = input_image.float()

        out = torch.nn.functional.interpolate(input_image, size=output_size, mode=mode, align_corners=False, antialias=antialias)
        if dtype == torch.float:
            out = out.round().clamp(min=0, max=256).to(torch.uint8)


def make_config():
    sizes = (
        ((224, 224), (64, 64)),
        ((270, 268), (224, 224)),
        ((256, 256), (1024, 1024)),
    )

    attrs = []
    for (HW1, HW2) in sizes:
        attrs.append([(1, 3, *HW1), HW2])  # 3 channels
        # attrs.append([(1, 1, *HW1), HW2])  # 1 channel

        attrs.append([(1, 3, *HW2), HW1])  # 3 channels
        # attrs.append([(1, 1, *HW2), HW1])  # 1 channel

    config = op_bench.config_list(
        attr_names=["input_size", "output_size"],
        attrs=attrs,
        cross_product_configs={
            # 'channels_last': [True, False],
            'channels_last': [True, ],
            # 'mode': ["bilinear", "bicubic"],
            'mode': ["bilinear", ],
            'antialias': [True, ],
            # 'antialias': [True, False],
            # 'dtype': [torch.float, torch.uint8]
            'dtype': [torch.uint8]
            # 'dtype': [torch.float]
        },
        tags=["short"],
    )

    return config

config = make_config()
op_bench.generate_pt_test(config, InterpolateBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
