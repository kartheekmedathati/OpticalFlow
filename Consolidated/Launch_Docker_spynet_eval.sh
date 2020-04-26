docker run -it \
    --runtime=nvidia \
    --shm-size=16GB \
    -e NVIDIA_VISIBLE_DEVICES=1 \
    --env="DISPLAY" \
    --volume="/etc/group:/etc/group:ro" \
    --volume="/etc/passwd:/etc/passwd:ro" \
    --volume="/etc/shadow:/etc/shadow:ro" \
    --volume="/etc/sudoers.d:/etc/sudoers.d:ro" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --volume="/media/medathati/4TBInt/MedathatiExt/:/home/4TBInt" \
    medathati/pytorch:CUDA8-py27 \
    bash


    
