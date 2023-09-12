#!/bin/bash
SAVE_PATH="./datasets"

set -e

mkdir -p ${SAVE_PATH}
cd ${SAVE_PATH}

wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip
rm tiny-imagenet-200.zip

cd -
