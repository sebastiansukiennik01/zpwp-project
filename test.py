import pandas as pd
from src.download import get_data, truncate_data, combine_datasets
from src.analysis import *
from src.visualization import *

if __name__ == "__main__":
    # # DOWNLOAD
    # get_data()  # pobieranie danych do ./real_fake/Fake.csv, ./real_fake/True.csv
    # truncate_data(["Fake", "True"])  # truncate data to 5000 sample
    # combine_datasets()  # combine fake and true into one dataset

    # ADD

    data = pd.read_csv("./real_fake/Data_short.csv", parse_dates=["datetime"])
    data = add_words_count(data, k=10, column="title")
    data = add_words_count(data, k=10, column="text")
    data = add_month_of_year(data)
    data = add_day_of_week(data)
