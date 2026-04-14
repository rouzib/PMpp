#!/bin/bash

SEED=0  # You can modify this or expose as a script argument

for SNAPSHOT in 0 1 2 3
do
    echo "Running snapshot=$SNAPSHOT with seed=$SEED"
    srun python pmpp_1024_per_snap.py --seed $SEED --snapshot $SNAPSHOT
done