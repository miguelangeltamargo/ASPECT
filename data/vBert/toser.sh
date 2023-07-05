#!/bin/bash

docker run -itd --gpus all -m 2g --cpus 0.45 -v /home/ubuntu/Miguel:/aspect aspect
