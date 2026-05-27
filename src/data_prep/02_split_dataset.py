import pathlib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
np.random.seed(42)
META_DATA_PATH = pathlib.Path("../../data/metadata/svd_cleaned_metadata.csv")
if __name__ == "__main__":
    healthy = 0
    pathological = 0
    data = pd.read_csv(META_DATA_PATH)
    for row in data.iterrows():
        if row[1]["Status"] ==0:
            healthy += 1
        else:
            pathological += 1
    train,temp = train_test_split(data, test_size=0.3, random_state=42,stratify = data["Status"])

    val,test = train_test_split(temp, test_size=0.5, random_state=42,stratify = temp["Status"])

    train.to_csv("../../data/metadata/train.csv", index=False)

    val.to_csv("../../data/metadata/val.csv", index=False)

    test.to_csv("../../data_prep/metadata/test.csv", index=False)


