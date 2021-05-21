#!/usr/bin/python3
from __future__ import print_function

import sys
import math
import lzma

if len(sys.argv) < 3:
    print("Usage: %s <column> [file...]" % sys.argv[0])
    sys.exit(0)

# Open all files
def actual_open(filename):
    if filename.endswith('.xz'):
        return lzma.open(filename, 'rt')
    else:
        return open(filename, 'r')

col = int(sys.argv[1])
files = [actual_open(f) for f in sys.argv[2:]]

# Read and average each files
N = float(len(files))
i = 0
running_mean = None
running_err = None
elems = [0.0] * len(files)
timesteps = [0] * len(files)
elements = [None] * len(files)

running_mean = None
running_err = 1.0
running_coeff = 0.6

while True:
    # Read a line from every file
    ok = True

    for j, f in enumerate(files):
        elements[j] = f.readline().strip().replace(',', ' ')

        if len(elements[j]) == 0:
            ok = False

    if not ok:
        # No more file
        break

    try:
        # Plot lines
        i += 1
        N = 0

        for j in range(len(files)):
            if len(elements[j]) > 0:
                parts = elements[j].split()

                elems[j] = float(parts[col])
                try:
                    timesteps[j] = int(parts[2])
                except:
                    timesteps[j] = 0

                N += 1
            else:
                elems[j] = 0.0
                timesteps[j] = 0

        mean = sum(elems) / N
        var = sum([(e - mean)**2 for e in elems if e != 0.0])
        std = math.sqrt(var)
        err = std / N

        if running_mean is None:
            running_mean = mean
        else:
            running_mean = running_coeff * running_mean + (1.0 - running_coeff) * mean

        running_err = running_coeff * running_err + (1.0 - running_coeff) * err

        if i % 32 == 0:
            print(i / 1000., running_mean, running_mean + running_err, running_mean - running_err, sum(timesteps) / N)
    except Exception:
        pass
