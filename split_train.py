import competition_utilities as cu
import csv
import datetime
import os
import numpy as np
import pandas as pd

def main():
    print "get data"
    data = cu.get_dataframe("train.csv")
    print "sort by creation date"
    data = data.sort_index(by="PostCreationDate")
    print "cut off"
    header = cu.get_header("train.csv")
    splits = np.array_split(data, 3)
    frames = [splits[0], splits[1]]
    train_data = pd.concat(frames)
    test_data = splits[2]
    # cutoff = datetime.datetime(2012, 7, 18)
    print "write to csv"
    cu.write_sample("train_data.csv", header, train_data)
    train_data.to_csv(os.path.join(cu.data_path, "train_data.csv"), index=False, header=header)
    test_data.to_csv(os.path.join(cu.data_path, "test_data.csv"), index=False, header=header)

if __name__=="__main__":
    main()
