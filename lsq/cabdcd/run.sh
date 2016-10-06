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

srun -n $1 -N ${13} --cpu-freq=2200000 ./cabdcd $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12}
#sed 's/.*: \(.*\)\n/\1/g' << $OUT
#echo $OUT
