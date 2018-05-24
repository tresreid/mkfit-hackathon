#!/usr/bin/env python

from __future__ import print_function

metrics = [ "gld_efficiency",
            "gst_efficiency",
            "flop_sp_efficiency",
            "l2_tex_read_hit_rate",
            "l2_tex_write_hit_rate",
            ]

cmd_parts = [
    "nvprof",
    "--csv",
    "--metrics " + ",".join(metrics),
    
    "./multorture",
    ]

import time
now = time.time()

# timestamp for spreadsheet
timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
output_fname = time.strftime("prof-%Y-%m-%d-%H%M%S.csv", time.localtime(now))

# TODO: use subprocess instead
import commands

# get git hash (does not check for local modifications though...)
git_hash = commands.getoutput("git rev-parse HEAD")

# check if there were modified files
status, dummy = commands.getstatusoutput("git diff --quiet")
assert status in (0,256), "status was " + str(status)

if status == 256:
    git_hash += "+modif"

# run the profiler
output = commands.getoutput(" ".join(cmd_parts))

lines = output.splitlines()

import re

while lines:
    line = lines.pop(0)
    if re.match("==\d+== Metric result:$", line):
        break


fout = open(output_fname, "w")
is_first = True
while lines:
    line = lines.pop(0)

    if is_first:
        print('"time","git_hash",%s' % line, file = fout)
        is_first = False
    else:
        print('"%s","%s",%s' % (timestamp, git_hash,line), file = fout)


print("wrote",output_fname)
