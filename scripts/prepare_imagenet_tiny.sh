#!/bin/bash
set -e

SAVE_PATH="../datasets"
cd ${SAVE_PATH}
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip
rm tiny-imagenet-200.zip
cd -
