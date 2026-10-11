import pandas as pd

FEATURES = pd.read_csv("../../../data/metadata/features.csv")
TEST_SPLIT = pd.read_csv("../../../data/metadata/test.csv")
TRAIN_SPLIT = pd.read_csv("../../../data/metadata/train.csv")
VAL_SPLIT = pd.read_csv("../../../data/metadata/val.csv")

def get_train_features():

    train_features = TRAIN_SPLIT.merge(FEATURES, on=['FilePath', 'Status'])
    train_features_x = train_features.drop(columns=['FilePath', 'Status','Id','RecordingID'])
    train_features_y = train_features['Status']

    return train_features_x, train_features_y

def get_val_features():

    val_features = VAL_SPLIT.merge(FEATURES, on=['FilePath', 'Status'])
    val_features_x = val_features.drop(columns=['FilePath', 'Status','Id','RecordingID'])
    val_features_y = val_features['Status']

    return val_features_x, val_features_y

def get_test_features():

    test_features = TEST_SPLIT.merge(FEATURES, on=['FilePath', 'Status'])
    test_features_x = test_features.drop(columns=['FilePath', 'Status', 'Id', 'RecordingID'])
    test_features_y = test_features['Status']

    return test_features_x, test_features_y