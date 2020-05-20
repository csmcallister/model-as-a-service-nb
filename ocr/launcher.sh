#!/usr/bin/env bash

pip3 install -r requirements.txt

/usr/bin/python3 detect.py --model_dir $SM_MODULE_DIR