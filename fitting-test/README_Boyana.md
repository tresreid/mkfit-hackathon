# Instructions

It should work similarly to the x86 version.
```bash
tar xzf mictest_gpu.tar.gz
```

Download/clone https://github.com/NVlabs/cub
```
export CUBROOT=[cub directory path]
```

We need to change some variables in the Makefile.config so it works for GPUs
```
python gpu_scripts/run_setup_only.py
```
Note that it replaces icc with icpc.

```
mkdir mkFit/ptxs  # or remove the --keep --keep-dir ptxs  option in mkFit/Makefile
make
```

If you are only interested in kernel time, the following should be good:
```
mpirun -np 1 ./mkFit/mkFit --build-ce --seed-based
```

If you are only interested in kernel time and you are able to track multiple kernels at once:
```
mpirun -np 4 ./mkFit/mkFit --build-ce --seed-based  # 4 or whatever
```

However the last command is not very representative because of the time required to simulate events. It is better to write simulations to file and read them afterwards.

```
mpirun -np 1 ./mkFit/mkFit --build-ce --seed-based --write --file-name data_with_mpi.bin --num-events 200

mpirun -np 1 ./mkFit/mkFit --build-ce --seed-based --read --file-name data_with_mpi.bin
mpirun -np 10 ./mkFit/mkFit --build-ce --seed-based --read --file-name data_with_mpi.bin
```

This is faster and allows to somewhat overlap computations and communications.

In the previous examples, the parallelization over events is done with MPI. Even though NVIDIA warns about mutex issues, it is possible to run a multithreaded version of the code. The MPI and multithreaded versions use slightly different file formats (to make it easier for MPI-IO to read the events).
To run the multithreaded version, first comment USE_MPI :=yes on line 65 of Makefile.config
Then:
```
make
./mkFit/mkFit --build-ce --seed-based --write --file-name data_no_mpi.bin --num-events 200
./mkFit/mkFit --build-ce --seed-based --read --file-name data_no_mpi.bin --num-thr 10 --num-thr-ev 10
```


Note:
The environment I have been running on:
Currently Loaded Modulefiles:
    1) rh/devtoolset/4
    2) cudatoolkit/9.0
    3) intel-mkl/2017.4/5/64
    4) intel/17.0/64/17.0.5.239
    5) openmpi/intel-17.0/2.1.0/64
    6) anaconda3/5.0.1
    Python version is 2.7
