#!/bin/bash
#line
# docker build -t aspect .
# docker run -it -d --gpus all --name aspect -v /home/miguelt:/aspect aspect
docker run -it -d --gpus all --shm-size 8G --name aspect -v /home/miguelt:/aspect aspect
docker images
docker ps -a
docker exec -it aspect bash
cd ASPECT/data 