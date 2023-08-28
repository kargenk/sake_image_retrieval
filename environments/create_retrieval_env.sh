IMAGE_NAME=$(whoami)_retrieval:latest
CONTAINER_NAME="kg_retrieval"

# dockerイメージのビルド
docker build ./ \
    --tag ${IMAGE_NAME} \
    --build-arg USERNAME=$USER \
    --build-arg GROUPNAME=$USER \
    --build-arg UID=$(id -u $USER) \
    --build-arg GID=$(id -g $USER) \
    --build-arg PASSWORD=testpass

# dockerコンテナの立ち上げ
docker run \
    --name ${CONTAINER_NAME} \
    --gpus all \
    --net=host \
    -h `hostname` \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v /home/$(whoami)/04_competition/nishika/sake_image_retrieval:/home/$USER/sake_image_retrieval \
    -it -d --shm-size=32gb ${IMAGE_NAME} /bin/zsh