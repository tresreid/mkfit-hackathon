#!/usr/bin/env python

from __future__ import print_function

metrics = [
    "flop_count_sp",
    "flop_sp_efficiency",
    "shared_store_transactions",
    "shared_load_transactions",
    "local_load_transactions",
    "local_store_transactions",
    "gld_transactions",
    "gst_transactions",
    "gld_throughput",
    "gst_throughput",
    "gld_requested_throughput",
    "gld_efficiency",
    "gst_requested_throughput",
    "gst_efficiency",
    "l2_read_transactions",
    "l2_write_transactions",
    "l2_utilization",
    "l1_cache_global_hit_rate",
    "l1_shared_utilization",
    "l2_l1_read_hit_rate"
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

# keep a copy of the CSV values to read from them afterwards
csv_buffer = ""

while lines:
    line = lines.pop(0)

    csv_buffer += line + "\n"

    if is_first:
        print('"time","git_hash",%s' % line, file = fout)
        is_first = False
    else:
        print('"%s","%s",%s' % (timestamp, git_hash,line), file = fout)


#----------
# parse CSV to produce some ASCII tables 
#----------
import cStringIO as StringIO
import csv
csv_buffer = StringIO.StringIO(csv_buffer)
reader = csv.DictReader(csv_buffer)

# first index is metric name
# second index is kernel name
# value is average value of metric
data = {}

for line in reader:
    # TODO: we could move this up
    kernel_name = line['Kernel']
    kernel_name = re.match("^(void )?([a-zA-Z0-9_]+)", kernel_name).group(2)

    data.setdefault(line["Metric Name"], {})[kernel_name] = line['Avg']

# print kernels ordered by selected metric

for selected_metric in [ "gld_throughput", "flop_sp_efficiency"]:
    print(selected_metric + ":")

    def parse_func(item):
        item = re.sub("[^0-9\.]","", item)
        return float(item)

    kernels = data[selected_metric].keys()

    for kernel in sorted(kernels, key = lambda kernel: parse_func(data[selected_metric][kernel]), reverse = True):
        print("  %-40s: %s" % (kernel, data[selected_metric][kernel]))


    print()

print("wrote",output_fname)

