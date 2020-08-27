#!/bin/bash
start=0
end=9
export model=my
export CUDA_VISIBLE_DEVICES=0
for i in $(eval echo {$start..$end}); do
export fold=$i
python train.py
done

