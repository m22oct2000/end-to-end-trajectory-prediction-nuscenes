#!/bin/bash
set -e
echo "Running evaluation..."
CHECKPOINT=${1:-"./outputs/checkpoints/checkpoint_best.pt"}
python training/evaluate.py --checkpoint $CHECKPOINT
echo "Evaluation complete."
