import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, recall_score, roc_auc_score


def main():
    pipeline = joblib.load("wav2vec2_base_logreg.joblib")

    X_test = np.load("../../data/processed/test_embeddings.npy")
    y_test = pd.read_csv("../../data/processed/test_manifest.csv")['Status'].to_numpy()

    test_result = pipeline.predict_proba(X_test)

    print(test_result.shape)
    print(pipeline.classes_)

    test_pos = test_result[:, 1]
    test_preds = test_pos >= 0.5

    auc_score_test = roc_auc_score(y_test, test_pos)
    sens = recall_score(y_test, test_preds, pos_label=1)
    spec = recall_score(y_test, test_preds, pos_label=0)
    uar = (sens + spec) / 2
    cm = confusion_matrix(y_test, test_preds)

    print(cm)
    print(f"AUC Score: {auc_score_test} , UAR Score: {uar}, Specificity: {spec}, Sensitivity: {sens}")


if __name__ == "__main__":
    main()
