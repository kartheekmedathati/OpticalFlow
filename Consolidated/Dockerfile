#This is a sample Image 
FROM nvidia/cudagl:10.0-devel-ubuntu16.04

RUN apt-get update
RUN apt-get install -y python python-pip python-dev
RUN apt-get install -y git
RUN apt-get install -y wget
RUN pip install numpy scipy 
RUN pip install networkx==2.2 matplotlib==2.2.4 scikit-image==0.14.2
RUN pip install torch torchvision
RUN pip install tensorboardX colorama tqdm setproctitle 
CMD [“echo”,”Image created”] 
