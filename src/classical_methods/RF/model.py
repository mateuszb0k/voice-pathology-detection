from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score,f1_score,precision_score,recall_score,confusion_matrix
from src.classical_methods.load_features import get_train_features,get_test_features,get_val_features


train_features_x, train_features_y = get_train_features()
test_features_x, test_features_y = get_test_features()
val_features_x, val_features_y = get_val_features()

N_ESTIMATORS = [100, 200, 300, 400]
MAX_DEPTH = [None, 15, 25]
MIN_SAMPLES_SPLIT = [2, 5, 10]


for estimators in N_ESTIMATORS:
    for depth in MAX_DEPTH:
        for samples_split in MIN_SAMPLES_SPLIT:
            print(f"Starting Training with {estimators} Estimators, {depth} Depth and {samples_split} Samples Split ")
            classifier = RandomForestClassifier(
                n_estimators=estimators,
                max_depth=depth,
                min_samples_split=samples_split,
                random_state = 123,
                n_jobs=-1,
                class_weight='balanced'
            )
            classifier.fit(train_features_x, train_features_y)
            y_pred = classifier.predict(val_features_x)
            print(f"Results {estimators} Estimators, {depth} Depth and {samples_split} Samples Split")
            print(f"F1 Score: {f1_score(val_features_y, y_pred, average='macro')},"
                  f" Balanced Accuracy: {balanced_accuracy_score(val_features_y, y_pred)},"
                  f" Precision: {precision_score(val_features_y, y_pred, average='macro')}",
                  f" Recall: {recall_score(val_features_y, y_pred, average='macro')}")
            cm = confusion_matrix(val_features_y, y_pred)
            print(cm)

