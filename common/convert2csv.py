# /usr/bin/env python3
import os
import sys
import numpy as np
import pandas as pd
import fire

def convert2csv(testfile, tsvfile, outfile, flag_name):
    infile = tsvfile
    labels = []
    with open(infile) as f:
        for line in f.readlines():
            line = line.strip()
            if line is None or len(line) == 0:
                continue
            ss = line.split()
            arr = np.array(ss)
            label = np.argmax(arr)
            labels.append(label)
    #         print(label)
    labels = np.array(labels)
    df = pd.read_csv(testfile)
    df[flag_name] = labels
    df_result = df[['id', flag_name]]
    df_result.to_csv(os.path.join(outfile, 'result.csv'), index=False)

if __name__ == '__main__':
    fire.Fire()