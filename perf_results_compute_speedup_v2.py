import pickle
from pathlib import Path
from typing import List, Optional
import unittest.mock

import torch
import torch.utils.benchmark as benchmark
from torch.utils.benchmark.utils import common
from torch.utils.benchmark.utils.compare import Table


import typer


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


class Value(common.Measurement): pass


class CustomizedTable(Table):

    def __init__(self, results, colorize, trim_significant_figures, highlight_warnings):
        assert len(set(r.label for r in results)) == 1

        self.results = results
        self._colorize = colorize
        self._trim_significant_figures = trim_significant_figures
        self._highlight_warnings = highlight_warnings
        self.label = results[0].label
        self.time_unit, self.time_scale = common.select_unit(
            min(r.median for r in results if not isinstance(r, Value))
        )

        self.row_keys = common.ordered_unique([self.row_fn(i) for i in results])
        self.row_keys.sort(key=lambda args: args[:2])  # preserve stmt order
        self.column_keys = common.ordered_unique([self.col_fn(i) for i in results])
        self.rows, self.columns = self.populate_rows_and_columns()


def get_new_table(compare, list_col1_col2_desc, debug):
    results = common.Measurement.merge(compare._results)
    grouped_results = compare._group_by_label(results)
    assert len(grouped_results.values()) == 1, grouped_results.values()
    groups_iter = iter(grouped_results.values())
    group = next(groups_iter)

    # Add speed-up column into results:
    updated_group = []
    sub_label = None
    v1 = None
    v2 = None
    r = None
    _, scale = common.select_unit(min([r.median for r in group]))

    for col1, col2, description in list_col1_col2_desc:
        for measurement in group:
            if debug:
                print("measurement.task_spec.description:", measurement.task_spec.description)

            if measurement.task_spec.description == col1:
                v1 = measurement.median
                sub_label = measurement.task_spec.sub_label
                if debug:
                    print("Matched col1:", col1, v1, sub_label)

            measurement2 = None
            for m2 in group:
                d2 = m2.task_spec.description
                sl2 = m2.task_spec.sub_label
                if d2 == col2 and sl2 == sub_label:
                    v2 = m2.median
                    if debug:
                        print("Matched col2:", col2, v2)
                    measurement2 = m2
                    break

            if measurement not in updated_group:
                updated_group.append(measurement)
            if v1 is not None and v2 is not None:
                if measurement2 not in updated_group:
                    updated_group.append(measurement2)
                r = v2 / v1 * scale
                if debug:
                    print("ratio is: ", r)
                v1 = None
                v2 = None
                sub_label = None
                speedup_task = common.TaskSpec(
                    "",
                    setup="",
                    label=measurement.label,
                    sub_label=measurement.sub_label,
                    num_threads=measurement.num_threads,
                    env=measurement.env,
                    description=description
                )
                speedup_measurement = Value(1, [r, ], speedup_task)
                r = None
                updated_group.append(speedup_measurement)

    assert len(updated_group) > len(group), "Seems like nothing was added. Run with --debug"

    table = CustomizedTable(
        updated_group,
        compare._colorize,
        compare._trim_significant_figures,
        compare._highlight_warnings
    )
    return table


def main(
    output_filepath: str,
    perf_files: List[str],
    compare: Optional[List[str]] = None,
    debug: bool = False,
):
    output_filepath = Path(output_filepath)
    if output_filepath.exists():
        raise FileExistsError(f"Output file '{output_filepath}' exists. Please provide a path to non-existing file")

    if debug:
        print("output_filepath:", output_filepath)
        print("perf_files:", perf_files, type(perf_files))
        print("compare:", compare)
        print()

    ab_results = []
    ab_configs = []
    for perf_filepath in perf_files:
        assert Path(perf_filepath).exists(), f"{perf_filepath} is not found"
        with open(perf_filepath, "rb") as handler:
            output = pickle.load(handler)
            ab_configs.append(
                f"Torch version: {output['torch_version']}\n"
                f"Torch config: {output['torch_config']}\n"
            )
            ab_results.extend(output["test_results"])

    assert len(ab_configs) == len(perf_files), (len(ab_configs), len(perf_files))
    compare_obj = benchmark.Compare(ab_results)

    list_col1_col2_desc = []
    for ccd in compare:
        list_ccd = ccd.split(";")
        if len(list_ccd) not in (2, 3):
            raise ValueError(
                "compare argument should encode columns and description with ';'. "
                "For example, --compare 'col one;colum two;speed-up' --compare 'col one;col two'"
            )
        if len(list_ccd) == 2:
            col1, col2 = list_ccd
            description = f"Speed-up: {col1} vs {col2}"
            list_ccd.append(description)

        col1, col2, description = list_ccd
        list_col1_col2_desc.append((col1, col2, description))

    table = get_new_table(compare_obj, list_col1_col2_desc, debug=debug)

    if debug:
        print(table.render())

    with output_filepath.open("w") as handler:
        handler.write(f"Description:\n")
        with unittest.mock.patch(
            "torch.utils.benchmark.utils.compare._Row.as_column_strings", patched_as_column_strings
        ):
            for in_filepath, config in zip(perf_files, ab_configs):
                handler.write(f"- {Path(in_filepath).stem}\n")
                handler.write(f"{config}\n")

            handler.write(f"\n")
            handler.write(table.render())


if __name__ == "__main__":
    typer.run(main)
