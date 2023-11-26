import kaggle
import pandas as pd
import numpy as np
import os


def get_data():
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        "clmentbisaillon/fake-and-real-news-dataset/", path="rael_fake", unzip=True
    )


def truncate_data(filename: list):
    for name in filename:
        np.random.seed(1)
        file_path = os.path.join("real_fake", f"{name}.csv")
        out_path = os.path.join("real_fake", f"{name}_short.csv")

        df = pd.read_csv(file_path, index_col=[0]).dropna()

        indexes = np.random.randint(0, df.shape[0], 5000)
        df.iloc[indexes, :].to_csv(out_path)


def combine_datasets():
    true_path = os.path.join("real_fake", "True_short.csv")
    fake_path = os.path.join("real_fake", "Fake_short.csv")
    out_path = os.path.join("real_fake", f"Data_short.csv")

    true = pd.read_csv(true_path, index_col=[0]).assign(is_true=1)
    fake = pd.read_csv(fake_path, index_col=[0]).assign(is_true=0)

    data = pd.concat([true, fake], axis=0)

    for i, row in data.iterrows():
        try:
            row["date"] = pd.to_datetime(row["date"])
        except:
            data.drop(index=row.name, inplace=True)
    data["date"] = pd.to_datetime(data["date"], format="mixed")

    data.to_csv(out_path)
