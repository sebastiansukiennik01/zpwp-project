import pandas as pd
from src.download import get_data, truncate_data, combine_datasets

if __name__ == "__main__":
    get_data()  # pobieranie danych do ./real_fake/Fake.csv, ./real_fake/True.csv
    truncate_data(["Fake", "True"])  # truncate data to 5000 sample
    combine_datasets()  # combine fake and true into one dataset

    # TODO
    # - połączyć dane w jeden set z kolumna true/fake
    # - ile jest fake newsów w zależnośc od subject
    # - ile jest fake newsów w zależnośc od dnia tygodnia
    # - ile jest fake newsów w zależnośc od dnia tygodnia
