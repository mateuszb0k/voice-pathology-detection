from xgboost import XGBClassifier
from sklearn.metrics import balanced_accuracy_score,f1_score,precision_score,recall_score,confusion_matrix
from src.classical_methods.load_features import get_train_features,get_test_features,get_val_features

train_features_x, train_features_y = get_train_features()
test_features_x, test_features_y = get_test_features()
val_features_x, val_features_y = get_val_features()

pos_count = train_features_y[train_features_y == 1].count()
neg_count = train_features_y[train_features_y == 0].count()


N_ESTIMATORS = [100, 200, 300]
LEARNING_RATE = [0.01, 0.05, 0.1]
MAX_DEPTH = [3, 6]

for estimators in N_ESTIMATORS:
    for depth in MAX_DEPTH:
        for lr in LEARNING_RATE:
            print(f"Starting training with {estimators} estimators, depth {depth} and learning rate {lr}.")
            model = XGBClassifier(
                learning_rate = lr,
                n_estimators = estimators,
                max_depth = depth,
                scale_pos_weight = neg_count/pos_count,
                random_state = 123,
                n_jobs = -1
            )
            model.fit(train_features_x, train_features_y)
            y_pred = model.predict(val_features_x)
            print(f"Results {estimators} Estimators, Depth {depth} and learning rate {lr}.")
            print(f"F1 Score: {f1_score(val_features_y, y_pred, average='macro')},"
                  f" Balanced Accuracy: {balanced_accuracy_score(val_features_y, y_pred)},"
                  f" Precision: {precision_score(val_features_y, y_pred, average='macro')}",
                  f" Recall: {recall_score(val_features_y, y_pred, average='macro')}")
            cm = confusion_matrix(val_features_y, y_pred)
            print(cm)

