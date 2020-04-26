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
    --volume="/media/medathati/MedathatiExt:/home/MedathatiExt" \
    --volume="/media/medathati/2910b1ae-0b9f-4c66-b5eb-357d01a713d7/medathati/Work:/home/SDD_Work" \
    nvidia/cuda \
    bash


    
