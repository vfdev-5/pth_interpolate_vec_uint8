import argparse
import pickle
from pathlib import Path
import unittest.mock

import torch
import torch.utils.benchmark as benchmark


def patched_as_column_strings(self):
    concrete_results = [r for r in self._results if r is not None]
    env = f"({concrete_results[0].env})" if self._render_env else ""
    env = env.ljust(self._env_str_len + 4)
    output = ["  " + env + concrete_results[0].as_row_name]
    for m, col in zip(self._results, self._columns or ()):
        if m is None:
            output.append(col.num_to_str(None, 1, None))
        else:
            if len(m.times) == 1:
                spread = 0
            else:
                spread = float(torch.tensor(m.times, dtype=torch.float64).std(unbiased=len(m.times) > 1))
                if col._trim_significant_figures:
                    spread = benchmark.utils.common.trim_sigfig(spread, m.significant_figures)
            output.append(f"{m.median / self._time_scale:>3.3f} (+-{spread / self._time_scale:>3.3f})")
    return output


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Tool to create a table with results as markdown file")
    parser.add_argument("output", default=str, help="Output filename, e.g. output.md")
    parser.add_argument("inputs", default=str, nargs='+', help="Input pickle files")

    args = parser.parse_args()
    assert not Path(args.output).exists(), f"{args.output} should not exist"

    ab_results = []
    ab_configs = []
    for in_filepath in args.inputs:
        assert Path(in_filepath).exists(), f"{in_filepath} is not found"
        with open(in_filepath, "rb") as handler:
            output = pickle.load(handler)
            ab_configs.append(
                f"Torch version: {output['torch_version']}\n"
                f"Torch config: {output['torch_config']}\n"
            )
            ab_results.extend(output["test_results"])

    assert len(ab_configs) == len(args.inputs), (len(ab_configs), len(args.inputs))
    compare = benchmark.Compare(ab_results)

    with open(args.output, "w") as handler:
        with unittest.mock.patch(
            "torch.utils.benchmark.utils.compare._Row.as_column_strings", patched_as_column_strings
        ):
            handler.write(f"Description:\n")
            for in_filepath, config in zip(args.inputs, ab_configs):
                handler.write(f"- {Path(in_filepath).stem}\n")
                handler.write(f"{config}\n")

            handler.write(f"\n")
            handler.write(str(compare))
