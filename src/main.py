import os
import pandas as pd
import numpy as np
import lightgbm as lgb


def unzip(path: str, dest: str):
    import zipfile

    with zipfile.ZipFile(path, "r") as zip_ref:
        zip_ref.extractall(dest)


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def main():
    if not os.path.exists("./data/titanic.zip"):
        unzip("./titanic.zip", "./data")
    sample_submission = load_data("data/gender_submission.csv")
    test_data = load_data("data/test.csv")
    train_data = load_data("data/train.csv")

    # Sex, Embarkedを数字に変換する
    train_data["Sex"] = train_data["Sex"].map({"male": 0, "female": 1})
    train_data["Embarked"] = train_data["Embarked"].map({"S": 0, "C": 1, "Q": 2})

    test_data["Sex"] = test_data["Sex"].map({"male": 0, "female": 1})
    test_data["Embarked"] = test_data["Embarked"].map({"S": 0, "C": 1, "Q": 2})

    train = lgb.Dataset(
        train_data.drop(["Name", "Ticket", "Cabin", "Survived"], axis=1),
        label=train_data["Survived"],
    )
    test = lgb.Dataset(
        test_data.drop(["Name", "Ticket", "Cabin"], axis=1),
        label=sample_submission["Survived"],
    )

    param = {
        "objective": "binary",
    }
    print(train)
    gbm = lgb.train(param, train, num_boost_round=100)

    y_pred = gbm.predict(test.data)
    y_pred = np.round(y_pred)
    y_pred = y_pred.astype(int)
    y_pred = pd.DataFrame(y_pred, columns=["Survived"])
    y_pred["PassengerId"] = test.data["PassengerId"]
    y_pred = y_pred[["PassengerId", "Survived"]]
    y_pred.to_csv("submission.csv", index=False)
    
    # 予測精度を表示
    print("accuracy: ", (y_pred["Survived"] == sample_submission["Survived"]).mean())


if __name__ == "__main__":
    main()
