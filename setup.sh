#!/usr/bin/env bash

sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt -y install python3.8
sudo apt -y install python3-pip

python3.8 -m pip install pip
python3.8 -m pip install -r requirements.txt

