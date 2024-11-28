#!/bin/bash

# Save this script as run_multiple.sh and make it executable with: chmod +x run_multiple.sh

for num_classes in {2..10}
do
    echo "Running script with num_classes=${num_classes}"
    python mt_supervised_merge_2.py --num_classes ${num_classes}
done
