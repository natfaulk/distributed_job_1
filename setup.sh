#!/usr/bin/env bash

sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install python3.8

python3.8 - m pip install -r requirements.txt

