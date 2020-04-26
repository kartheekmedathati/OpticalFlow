docker run -it \
    --runtime=nvidia \
    --shm-size=16GB \
    -e NVIDIA_VISIBLE_DEVICES=0 \
    --env="DISPLAY" \
    --volume="/etc/group:/etc/group:ro" \
    --volume="/etc/passwd:/etc/passwd:ro" \
    --volume="/etc/shadow:/etc/shadow:ro" \
    --volume="/etc/sudoers.d:/etc/sudoers.d:ro" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --volume="/media/medathati/4TBInt/MedathatiExt/:/home/4TBInt" \
    flownet2py3:latest \
    bash


    
