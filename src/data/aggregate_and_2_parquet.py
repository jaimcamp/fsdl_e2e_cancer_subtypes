#! /usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', help="Input directory with txt files from TCGA", required=True)
parser.add_argument('--output_dir', help="Output directory with txt files from TCGA", required=True)
args = parser.parse_args()

input = Path(args.input_dir)
output = Path(args.output_dir)

# Decompress
list_gz = [x.resolve() for x in input.glob('**/*.gz')]

chunks = np.array_split(list_gz, 10)
for ix, chunk in enumerate(chunks):
    output_filename = output / "interim" /f"chunk_{ix}.parquet"
    print(output_filename.resolve())
    tmp_df = pd.concat(
        (pd.read_table(filename, delimiter='\t', header=None, index_col=0, compression='gzip').transpose() for filename in chunk),
        axis=0)
    tmp_df.set_index(pd.Series([Path(x).with_suffix('').stem for  x in chunk]), inplace=True)
    tmp_df.to_parquet(output_filename.resolve())
    del tmp_df

list_chunks = [x.resolve() for x in output.glob('interim/*.parquet')]
final = pd.concat((pd.read_parquet(chunk) for chunk in list_chunks), axis=0)
final.to_parquet((output / "processed"/ "complete_dataset.parquet").resolve())
