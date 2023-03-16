import pickle
from pathlib import Path
from typing import List, Optional

import torch.utils.benchmark as benchmark
from torch.utils.benchmark.utils import common
from torch.utils.benchmark.utils.compare import Table


import fire


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


def main(
    output_filepath: str,
    perf_files: List[str],
    *,
    col1: str,
    col2: str,
    description: Optional[str] = None,
    debug: bool = False
):
    output_filepath = Path(output_filepath)
    if output_filepath.exists():
        raise FileExistsError(f"Output file '{output_filepath}' exists. Please path to non-existing file")

    if debug:
        print("output_filepath:", output_filepath)
        print("perf_files:", perf_files, type(perf_files))
        print("col1:", col1, type(col1))
        print("col1:", col2, type(col2))
        print("description:", description, type(description))

    ab_results = []
    for perf_filepath in perf_files:
        assert Path(perf_filepath).exists(), f"{perf_filepath} is not found"
        with open(perf_filepath, "rb") as handler:
            output = pickle.load(handler)
            ab_results.extend(output["test_results"])

    compare = benchmark.Compare(ab_results)

    results = common.Measurement.merge(compare._results)
    grouped_results = compare._group_by_label(results)
    assert len(grouped_results.values()) == 1, grouped_results.values()
    groups_iter = iter(grouped_results.values())
    group = next(groups_iter)

    col1 = "torch (2.1.0a0+git0968a5d) PR"
    col2 = "torch (2.1.0a0+git5309c44) nightly"
    if description is None:
        description = f"Speed-up: {col1} vs {col2}"

    # Add speed-up column into results:
    updated_group = []
    sub_label = None
    v1 = None
    v2 = None
    r = None
    _, scale = common.select_unit(min([r.median for r in group]))

    for measurement in group:
        if debug:
            print(measurement.task_spec.description)

        if measurement.task_spec.description == col1:
            v1 = measurement.median
            sub_label = measurement.task_spec.sub_label
            if debug:
                print(col1, v1, sub_label)

        measurement2 = None
        for m2 in group:
            d2 = m2.task_spec.description
            sl2 = m2.task_spec.sub_label
            if d2 == col2 and sl2 == sub_label:
                v2 = m2.median
                if debug:
                    print(col2, v2)
                measurement2 = m2
                break

        if measurement not in updated_group:
            updated_group.append(measurement)
        if v1 is not None and v2 is not None:
            if measurement2 not in updated_group:
                updated_group.append(measurement2)
            r = v2 / v1 * scale
            if debug:
                print("->", r)
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

    table = CustomizedTable(
        updated_group,
        compare._colorize,
        compare._trim_significant_figures,
        compare._highlight_warnings
    )

    with output_filepath.open("w") as h:
        h.write(table.render())


if __name__ == "__main__":
    fire.Fire(main)
