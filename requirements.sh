#!/bin/bash
sudo apt update && sudo apt upgrade
sudo apt install git libzbar0
git clone https://github.com/mindee/doctr.git
python3 -m pip install -r requirements.txt
sudo python3 -m pip install -e doctr/.[tf]
cd doctr && sudo python3 setup.py install
cd ..

