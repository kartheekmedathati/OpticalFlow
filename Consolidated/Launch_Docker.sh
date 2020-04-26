docker run -it \
    --runtime=nvidia \
    --shm-size=12GB \
    --env="DISPLAY" \
    --volume="/etc/group:/etc/group:ro" \
    --volume="/etc/passwd:/etc/passwd:ro" \
    --volume="/etc/shadow:/etc/shadow:ro" \
    --volume="/etc/sudoers.d:/etc/sudoers.d:ro" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --volume="/run/media/medathati/MedathatiExt:/home/MedathatiExt" \
    --volume="/home/medathati/Centos/home/medathati/Work/OpticalFlowEstimation:/home/OFCode"\
    --volume="/home/medathati/Work:/home/SDD_Work" \
    ubuntu:pytorch \
    bash


    