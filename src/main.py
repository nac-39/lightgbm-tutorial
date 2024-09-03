import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna


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
        trial (??): パラメータ？

    Returns:
        float?: 精度
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
    train_data = pd.read_csv('data/train.csv')  # ここでtrain_dataを定義
    train_data.drop(["Name", "Ticket", "Cabin", "Survived"], axis=1),
    sample_submission = pd.read_csv('data/gender_submission.csv')
    test_data = pd.read_csv('data/test.csv')
    train_data = preprocess_data(train_data)
    test_data = preprocess_data(test_data)
    train = lgb.Dataset(
        train_data.drop(["Name", "Ticket", "Cabin", "Survived"], axis=1),
        label=train_data["Survived"],
    )
    validation = lgb.Dataset(
        test_data.drop(["Name", "Ticket", "Cabin"], axis=1),
        label=sample_submission["Survived"],
        reference=train
    )

    gbm = lgb.train(param, train, valid_sets=[validation])
    preds = gbm.predict(test_data.drop(["Name", "Ticket", "Cabin"], axis=1))
    pred_labels = np.rint(preds)
    accuracy = (pred_labels == sample_submission["Survived"]).mean()
    return accuracy


def main():
    if not os.path.exists("./data/titanic.zip"):
        unzip("./titanic.zip", "./data")
    sample_submission = load_data("data/gender_submission.csv")
    test_data = load_data("data/test.csv")
    train_data = load_data("data/train.csv")

    # Sex, Embarkedを数字に変換する
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