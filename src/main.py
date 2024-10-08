import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import KFold


def unzip(path: str, dest: str):
    import zipfile

    with zipfile.ZipFile(path, "r") as zip_ref:
        zip_ref.extractall(dest)


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})
    return df

def objective(trial):
    """optunaのobjective関数

    Args:
        trial (optuna.trial.Trial): パラメータ試行のためのオブジェクト

    Returns:
        float: 精度
    """
    param = {
        "boosting_type": "gbdt",
        "objective": "binary",
        "metric": "binary_error",
        "learning_rate": trial.suggest_uniform("learning_rate", 0.01, 0.08),
        "num_leaves": trial.suggest_int("num_leaves", 10, 100),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 50),
        "min_sum_hessian_in_leaf": trial.suggest_int("min_sum_hessian_in_leaf", 5, 50),
        "lambda_l1": trial.suggest_loguniform("lambda_l1", 1e-8, 10.0),
        "lambda_l2": trial.suggest_loguniform("lambda_l2", 1e-8, 10.0),
        "bagging_fraction": trial.suggest_uniform("bagging_fraction", 0.1, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 0, 10),
        "feature_fraction": trial.suggest_uniform("feature_fraction", 0.1, 1.0),
        "random_seed": 42,
        "verbose": -1,
    }

    # データ読み込み
    train_data = pd.read_csv('data/train.csv')

    # 不要な列は削除しない（後で使うため）
    train_data = preprocess_data(train_data)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    accuracy_list = []

    for train_index, val_index in kf.split(train_data):
        train_data_fold = train_data.iloc[train_index]
        val_data_fold = train_data.iloc[val_index]

        # 前処理
        train_data_fold = preprocess_data(train_data_fold)
        val_data_fold = preprocess_data(val_data_fold)

        # 不要な列を削除（学習データと検証データで同じ処理を行う）
        features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]  # 例：残すべき特徴量
        train_X = train_data_fold[features]
        train_y = train_data_fold["Survived"]
        val_X = val_data_fold[features]
        val_y = val_data_fold["Survived"]

        # LightGBMのデータセット作成
        train_fold = lgb.Dataset(train_X, label=train_y)
        val_fold = lgb.Dataset(val_X, label=val_y, reference=train_fold)

        # モデル訓練
        gbm = lgb.train(param, train_fold, valid_sets=[val_fold])

        # 予測
        preds = gbm.predict(val_X, num_iteration=gbm.best_iteration)
        pred_labels = np.rint(preds)

        # 精度計算
        accuracy = (pred_labels == val_y).mean()
        accuracy_list.append(accuracy)
        
    # 平均精度を返す
    accuracy = np.mean(accuracy_list)
    return accuracy


def main():
    if not os.path.exists("./data/titanic.zip"):
        unzip("./titanic.zip", "./data")
    sample_submission = load_data("data/gender_submission.csv")
    test_data = load_data("data/test.csv")
    train_data = load_data("data/train.csv")

    # Sex, Embarkedを数字に変換する
    train_data = train_data.sample(frac=0.8, random_state=42)

    train_data = preprocess_data(train_data)
    test_data = preprocess_data(test_data)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # 最適なパラメータで再学習
    best_params = trial.params
    best_params["objective"] = "binary"
    best_params["metric"] = "binary_error"
    best_params["boosting_type"] = "gbdt"
    best_params["random_seed"] = 42

    train = lgb.Dataset(
        train_data.drop(["Name", "Ticket", "Cabin", "Survived"], axis=1),
        label=train_data["Survived"],
    )
    gbm = lgb.train(best_params, train, num_boost_round=100)

    y_pred = gbm.predict(test_data.drop(["Name", "Ticket", "Cabin"], axis=1))
    y_pred = np.round(y_pred)
    y_pred = y_pred.astype(int)
    y_pred = pd.DataFrame(y_pred, columns=["Survived"])
    y_pred["PassengerId"] = test_data["PassengerId"]
    y_pred = y_pred[["PassengerId", "Survived"]]
    y_pred.to_csv("submission.csv", index=False)

    # 予測精度を表示
    print("accuracy: ", (y_pred["Survived"] == sample_submission["Survived"]).mean())

if __name__ == "__main__":
    main()