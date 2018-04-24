import pandas as pd

MLDATAFILEDIR = r"D:\科研\CodeQualityAnalysis\CodeAnalysis\followerExp\mldata.csv"
datadf = pd.read_csv(MLDATAFILEDIR)
print(datadf.describe())