import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score,f1_score,precision_score,recall_score,confusion_matrix
from src.classical_methods.load_features import get_train_features,get_test_features,get_val_features

test_features_x, test_features_y = get_test_features()
train_features_x, train_features_y = get_train_features()
val_features_x, val_features_y = get_val_features()

scaler = StandardScaler()
X_train = scaler.fit_transform(train_features_x)
X_val = scaler.transform(val_features_x)
X_test = scaler.transform(test_features_x)

print("Training data shape: ", X_train.shape)
print("Validation data shape: ", X_val.shape)
print("Test data shape: ", X_test.shape)

print("Starting Training...")

C = [0.1 , 1.0 , 10.0, 50.0, 100.0 ]
gamma = ['scale', 'auto' , 0.001, 0.005, 0.01, 0.05]

for _c in C:
    for _g in gamma:
        print(f"Starting Training: C = {_c}, gamma = {_g}")
        svm_classifier = SVC(kernel='rbf', C=_c, gamma=_g, probability=True, class_weight='balanced',
                             random_state=123)
        svm_classifier.fit(X_train,train_features_y)
        y_pred = svm_classifier.predict(X_val)
        print(f"Results C = {_c}, gamma = {_g}")
        print(f"F1 Score: {f1_score(val_features_y, y_pred, average='macro')},"
              f" Balanced Accuracy: {balanced_accuracy_score(val_features_y, y_pred)},"
              f" Precision: {precision_score(val_features_y, y_pred, average='macro')}",
              f" Recall: {recall_score(val_features_y, y_pred, average='macro')}")
        cm = confusion_matrix(val_features_y, y_pred)
        print(cm)
