#!/bin/bash
#line
# docker build -t aspect .
# docker run -it -d --gpus all --name aspect -v /home/miguelt:/aspect aspect
# docker run -it -e SHELL=/bin/zsh -d --gpus all --shm-size 13G --name aspect -v /home/miguelt:/aspect aspect
docker run -e TERM -e COLORTERM -e LC_ALL=C.UTF-8 -it -d --gpus all --shm-size 13G --name aspect -v /home/mtamargo/ASPECT:/home -w /home aspect
docker ps -a
docker exec -it aspect zsh
