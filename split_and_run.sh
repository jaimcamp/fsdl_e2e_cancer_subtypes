#! /usr/bin/env bash

MANIFEST_FILE=$1
WORKERS=$2
OUT_DIRL=$3

tail -n +2 $MANIFEST_FILE | split -l 1000 - clinical_split_
if [ $? -eq 0 ]; then
    for file in clinical_split_*; do head -n 1 $MANIFEST_FILE > tmp_file; cat $file >> tmp_file; mv -f tmp_file $file; done
    if [ $? -eq 0 ]; then
        find . -name "clinical_split_*" |parallel -I% --max-args 1 -j $WORKERS gdc-client download --manifest % --dir $OUT_DIRL
        rm clinical_split_*
    else
        echo FAILURE
    fi
else
    echo FAILURE
fi
