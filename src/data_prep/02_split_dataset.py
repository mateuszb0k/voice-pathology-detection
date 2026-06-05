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
    unique_speakers = data.drop_duplicates(subset='Id')[['Id','Status']]
    train_spk,temp_spk = train_test_split(unique_speakers, test_size=0.3, random_state=42,stratify = unique_speakers["Status"])
    val_spk,test_spk = train_test_split(temp_spk, test_size=0.5, random_state=42,stratify = temp_spk["Status"])
    train_data = data[data['Id'].isin(train_spk['Id'])]
    val_data = data[data['Id'].isin(val_spk['Id'])]
    test_data = data[data['Id'].isin(test_spk['Id'])]
    train_data.to_csv("../../data/metadata/train.csv", index=False)

    val_data.to_csv("../../data/metadata/val.csv", index=False)

    test_data.to_csv("../../data/metadata/test.csv", index=False)
    print(data['Status'].value_counts())

