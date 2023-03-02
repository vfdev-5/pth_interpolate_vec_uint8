import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("f1", nargs="?", default="main")
parser.add_argument("f2", nargs="?", default="new")
args = parser.parse_args()

with open(args.f1) as f:
    main = f.readlines()
with open(args.f2) as f:
    new = f.readlines()

out = []

for main_line, new_line in zip(main, new):
    # num_threads=1  # TODO: remove
    if main_line.startswith("num_threads="):
        num_threads = int(main_line.split("=")[-1])
    if main_line.startswith("# Input"):
        deets = f"{main_line.strip()}, {num_threads=}"
    if main_line.startswith("Forward"):
        main_time = float(main_line.split()[-1])
        new_time = float(new_line.split()[-1])
        ratio = main_time / new_time
        fmt = ".1f" if ratio < 3 else ".0f"
        improv = f"{ratio:{fmt}}X"
        time_fmt = ",.3f" if new_time < 100 else ",.1f"
        deets = deets.strip().replace("# Input: ", "")
        deets = deets.replace(": ", "=")
        deets = deets.replace("input_size=", "")
        deets = deets.replace(", output_size=", " -> ")
        deets = deets.replace("dtype=torch.", "")
        deets = deets.replace("mode=", "")
        deets = deets.replace("antialias=", "")
        deets = deets.replace("channels_last=", "")
        # deets = deets.replace("channels_last=True, ", "")
        split = deets.split(",")

        # size = ','.join(split[:-3])
        # mode, dtype, threads = split[-3:]
        # deets = f"{size:<30} {mode:<15} {dtype:<10} {threads:<15}"

        size = ','.join(split[:-5])
        channels_last, mode, antialias, dtype, threads= split[-5:]
        deets = f"{size:<33} {channels_last:<7} {antialias:<7} {mode:<10} {threads:<15}"

        l = f"{deets}  {improv:<5} {main_time / 1000:{time_fmt}}ms vs {new_time / 1000:{time_fmt}}ms"
        out.append(l)


def key(s):
    # s = ''.join(s.split()[1:]) # remove "N.nX" part
    num_threads = (int(re.findall(r"num_threads=(\d+)", s)[0]),)

    input_shape, output_shape = re.findall("\(.*?\)", s)
    input_shape = input_shape[1:-1]  # remove parenthesis
    input_HW = tuple(int(x) for x in input_shape.split(",")[-2:])
    input_C = (-int(input_shape.split(",")[1]),)

    output_HW = tuple(int(x) for x in output_shape[1:-1].split(","))
    is_downsample = (output_HW[0] < input_HW[0],)
    if "linear" in s:
        mode = "linear"
    elif "nearest-exact" in s:
        mode = "nearest-exact"
    else:
        # assert "nearest" in s
        mode = "nearest"
    mode = (mode,)
    return is_downsample + input_HW + output_HW + num_threads + input_C + mode

for i, l in enumerate(sorted(out, key=key)):
    if i % 8 == 0:
        print()
    # if i % 10 == 0 and i % 40 != 0:
    #     print()
    # if i % 40 == 0:
    #     print("-" * 100)
    print(l)
