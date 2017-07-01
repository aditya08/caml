srun -n 24 --cpu-freq=2400000 ./cabcd ../../../data/a9a.txt 32561 123 .001 65536 1e-3 10 512 1 1 1 2>&1 | tee a9a_24_nominal.log
srun -n 24 ./cabcd ../../../data/a9a.txt 32561 123 .001 65536 1e-3 10 512 1 1 1 2>&1 | tee a9a_24_TB.log
srun -n 24 --cpu-freq=3200000 ./cabcd ../../../data/a9a.txt 32561 123 .001 65536 1e-3 10 512 1 1 1 2>&1 | tee a9a_24_peak.log

#!/bin/bash

#printf "Number of processors: "
#read P
#
#printf "Filename: "
#read fname
#
#printf "number of samples: "
#read n
#
#printf "number of features: "
#read m
#
#printf "lambda: "
#read lambda
#
#printf "iterations: "
#read maxit
#
#
#printf "tolerance: "
#read tol
#
#printf "random seed: "
#read seed
#
#printf "frequency of plotting: "
#read freq
#
#printf "block size: "
#read blk
#
#
#printf "loop blocking parameter: "
#read s
#
#
#printf "number of benchmark iterations: "
#read niters

#OUT=`srun -n $P --cpu-freq=2200000 ./cabcd $fname  $n $m $lambda $maxit $tol $seed $freq $blk $s $niters`
