#!/bin/bash
cd ./networks/correlation_package
python3 setup.py install --user --force
cd ../resample2d_package 
python3 setup.py install --user --force
cd ../channelnorm_package 
python3 setup.py install --user --force
cd ..
