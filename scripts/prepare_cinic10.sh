#!/bin/bash
set -e

SAVE_PATH="../datasets"
cd ${SAVE_PATH}
wget https://datashare.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz
tar -xvzf CINIC-10.tar.gz
rm CINIC-10.tar.gz
cd -
