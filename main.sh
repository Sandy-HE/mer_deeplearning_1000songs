#!/bin/bash
start=0
end=4
export model=base
export CUDA_VISIBLE_DEVICES=1
for i in $(eval echo {$start..$end}); do
export fold=$i
python train.py
done

