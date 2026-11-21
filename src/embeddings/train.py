import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def main():
    X_train = np.load("../../data/processed/train_embeddings.npy")
    X_val = np.load("../../data/processed/val_embeddings.npy")

    y_train = pd.read_csv("../../data/processed/train_manifest.csv")['Status'].to_numpy()
    y_val = pd.read_csv("../../data/processed/val_manifest.csv")['Status'].to_numpy()

    pipeline = Pipeline([("scaler", StandardScaler()),("classifier", LogisticRegression(C = 1.0, class_weight = "balanced", random_state = 123, max_iter = 1000))])

    pipeline.fit(X_train, y_train)
    val_result = pipeline.predict_proba(X_val)

    print(val_result.shape)
    print(pipeline.classes_)

    val_pos = val_result[:,1]

    auc_score_val = roc_auc_score(y_val, val_pos)

    val_preds = val_pos >= 0.5
    sens = recall_score(y_val, val_preds, pos_label=1)
    spec = recall_score(y_val, val_preds, pos_label = 0 )
    uar = (sens + spec) / 2
    cm = confusion_matrix(y_val, val_preds)
    print(cm)
    print(f"AUC Score: {auc_score_val} , UAR Score: {uar}, Specificity: {spec}, Sensitivity: {sens}")

    joblib.dump(pipeline, "wav2vec2_base_logreg.joblib")

    pipeline1 = joblib.load("wav2vec2_base_logreg.joblib")

    val1_result = pipeline1.predict_proba(X_val)

    print(np.allclose(val1_result, val_result))


if __name__ == "__main__":
    main()
