#!/usr/bin/env python

from __future__ import print_function

#metrics = [
#    "gld_transactions",
#    "flop_count_sp"
#]
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
    
    "./multorture 1",
    ]

import time
now = time.time()

# timestamp for spreadsheet
timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
output_fname = time.strftime("prof-%Y-%m-%d-%H%M%S.csv", time.localtime(now))

# TODO: use subprocess instead
#import commands
import subprocess
# get git hash (does not check for local modifications though...)
#git_hash = commands.getoutput("git rev-parse HEAD")
git_hash = subprocess.getstatusoutput("git rev-parse HEAD")[1]

# check if there were modified files
#status, dummy = commands.getstatusoutput("git diff --quiet")
status, dummy = subprocess.getstatusoutput("git diff --quiet")
assert status in (0,256,1), "status was " + str(status)

if status == 256 or status == 1:
    git_hash += "+modif"

# run the profiler
#output = commands.getoutput(" ".join(cmd_parts))
output1 = subprocess.Popen("nvprof --csv ./multorture ",shell=True,stderr=subprocess.PIPE, stdout=subprocess.PIPE, stdin=subprocess.PIPE,universal_newlines=True).communicate()[1]
print(output1)
lines1 = output1.splitlines()
output = subprocess.Popen(" ".join(cmd_parts),shell=True,stderr=subprocess.PIPE, stdout=subprocess.PIPE, stdin=subprocess.PIPE,universal_newlines=True).communicate()[1]
print(output)
lines = output.splitlines()

import re

while lines1:
    line = lines1.pop(0)
    if re.match("==\d+== Profiling result:$", line):
        break

fout1 = open("profile_test.csv", "w")
is_first = True
csv_buffer = ""
skip=False

while lines1:
    line = lines1.pop(0)
    if "API calls" in line:
        break
    csv_buffer += line + "\n"
    
   
    if skip:
        skip = False
        continue
    if is_first:
        print('"time","git_hash","Metric Name",%s' % line.replace("Name","Kernel").replace("Calls","Invocations"), file = fout1)
        is_first = False
        skip = True
    else:
        print('"%s","%s","Avg Time",%s' % (timestamp, git_hash,line), file = fout1)


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
#import cStringIO as StringIO
import io
import csv
#csv_buffer = StringIO.StringIO(csv_buffer)
csv_buffer = io.StringIO(csv_buffer)
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

for selected_metric in metrics:#[ "gld_throughput", "flop_sp_efficiency"]:
    print(selected_metric + ":")

    def parse_func(item):
        item = re.sub("[^0-9\.]","", item)
        return float(item)

    kernels = data[selected_metric].keys()

    for kernel in sorted(kernels, key = lambda kernel: parse_func(data[selected_metric][kernel]), reverse = True):
        print("  %-40s: %s" % (kernel, data[selected_metric][kernel]))


    print()

print("wrote",output_fname)
fout.close()
fout1.close()
import pandas as pd
csv_dataframe1 = pd.read_csv(output_fname)
prof_dataframe1 = pd.read_csv("profile_test.csv")
#new_dataframe = pd.merge(csv_dataframe,prof_dataframe,on='Kernel')
#new_dataframe.to_csv("data_"+output_fname)
#new_table = new_dataframe[new_dataframe['Invocations']==1000].pivot(index="Metric Name", columns='Kernel',values='Avg')

csv_dataframe = csv_dataframe1[["Kernel","Invocations","Metric Name", "Avg","Max","Min"]]
prof_dataframe = prof_dataframe1[["Kernel","Invocations","Metric Name", "Avg","Max","Min"]]
new_dataframe = pd.concat([csv_dataframe,prof_dataframe])

new_dataframe["Kernel"] = new_dataframe["Kernel"].apply(lambda x: x.split("(")[0].split("<")[0])

#csv_dataframe.to_csv("csv_test.csv")
#prof_dataframe.to_csv("prof_test.csv")
#new_dataframe.to_csv("new_test.csv")

new_table = new_dataframe[new_dataframe['Invocations']==1000].pivot(index="Metric Name", columns='Kernel',values='Avg')
#csv_table.to_csv("csv_"+output_fname)
#prof_table = prof_dataframe[prof_dataframe['Calls']==1000].pivot(index="Metric Name",columns='Kernel',values='Avg')
#prof_table.to_csv("prof_"+output_fname)
##new_table = pd.merge(csv_table,prof_table,on='Kernel')
#new_table = pd.merge(csv_table,prof_table)
new_table.to_csv("new_"+output_fname)
