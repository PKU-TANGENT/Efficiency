#!/bin/bash
cleanup_tasks=( mrpc rte wnli stsb cola )
for i in "${cleanup_tasks[@]}"; do
    find . -type d -name "*${i}" -exec rm -rf {} \;
done
