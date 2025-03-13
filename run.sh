#!/bin/bash

for grid_size in {25..50}; do
    python simple_custom_taxi_env.py \
        --grid_size $grid_size \
        --do_training \
        --pre_trained
done
