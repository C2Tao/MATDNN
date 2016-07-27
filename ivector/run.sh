#!/bin/bash

./script/extract-mfcc.sh
./script/train_diag_ubm.sh
./script/train_full_ubm.sh
./script/ivector-train.sh
./script/ivector-extract.sh
