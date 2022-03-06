# level1-image-classification-level1-cv-06
level1-image-classification-level1-cv-06 created by GitHub Classroom

### Guide

python train.py --epochs 10 --resize 128 96 --lr 0.001

python inference.py --model_dir ./model/exp

tensorboard --logdir=./model --bind_all
