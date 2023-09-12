#!/bin/bash
SAVE_PATH="./datasets"

set -e
mkdir -p ${SAVE_PATH}
cd ${SAVE_PATH}

wget https://datashare.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz
tar -xvzf CINIC-10.tar.gz
rm CINIC-10.tar.gz

cd -
