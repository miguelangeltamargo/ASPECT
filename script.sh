#!/bin/bash
#line
# Remove any existing container with the name 'aspect'
if [ "$(docker ps -aq -f name=aspect)" ]; then
    echo "Removing existing container 'aspect'..."
    docker rm -f aspect
fi

docker build -t aspect .
# docker run -it -d --gpus all --name aspect ./:/home/aspect aspect
# docker run -it -e SHELL=/bin/zsh -d --gpus all --shm-size 13G --name aspect ./:/home/aspect aspect

docker run -e TERM -e COLORTERM -e LC_ALL=C.UTF-8 -it -d --gpus all --shm-size 8G --name aspect -v $(pwd):/home/aspect -w /home/aspect aspect
docker ps -a
docker exec -it aspect zsh
# docker exec -it aspect ls /home/aspect
