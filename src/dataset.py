import pandas as pd
import numpy as np
import os
from abc import ABC

from sklearn.model_selection import train_test_split


class Preprocessing(ABC):
    
    def __init__(self) -> None:
        self.data = None
                
    def add_words_count(self, k: int = 10, columns: list = ["title", "text"]) -> pd.DataFrame:
        """
        Extraxts top 'k' most frequent words in column, and adds 'k' columns,
        where each column represents one of the most frequent words and
        has number of occurrances of this word in specific row.
        We ommit most frequent prepositions/single letters occuring in English.
        Args:
            data : pandas DataFrame
            k : number of most frequent words to be selected
            column : column from which words should be selected
        Returns:
            DataFrame with additional columns
        """
        to_drop = [
            "to",
            "in",
            "that",
            "i",
            "t",
            "s",
            "he",
            "was",
            "it",
            "has",
            "be",
            "will",
            "on",
            "of",
            "for",
            "a",
            "the",
            "and",
            "is",
            "about",
            "from",
            "this",
            "they",
            "but",
            "an",
            "his",
            "by",
            "after",
            "at",
            "as",
            "over",
            "with",
        ]
        for column in columns:
            self.data[column] = self.data[column].str.lower()
            top_k_words = (
                self.data[column]
                .str.replace("(\[|\]|\(|\))", "", regex=True)
                .str.split(expand=True)
                .stack()
                .value_counts()
                .drop(to_drop, errors="ignore")
                .head(k)
            )

            for word, _ in top_k_words.items():
                self.data[f"{column}_{word}"] = self.data[column].str.count(word)

        return self

    def add_day_of_week(self):
        """
        Add day of week column which represents day of the week starting from Monday = 0
        """
        self.data["weekday"] = self.data["datetime"].dt.weekday

        return self

    def add_month_of_year(self):
        """
        Add month of year column which represents month of the year starting from Januray = 0
        """
        self.data["month"] = self.data["datetime"].dt.month

        return self

    def prepare_title_word_count(self):
        words = list(filter(lambda x: "title_" in x, self.data.columns))
        x_labels = [w.replace("title_", "") for w in words]
        x = list(range(0, len(x_labels)))
        result = self.data[words].sum()

        return result
    
    def prepare_text_word_count(self):
        words = list(filter(lambda x: f"text_" in x, self.data.columns))
        x_labels = [w.replace(f"text_", "") for w in words]
        x = list(range(0, len(x_labels)))
        result = self.data[words].sum()

        return result


    def prepare_monthly_comparison(self):
        self.data = self.data.loc[self.data["datetime"].dt.year != 2015]
        self.data["month"] = self.data["datetime"].dt.to_period("M")
        result = self.data.groupby(["month", "is_true"]).size().unstack().fillna(0)

        return result

    def prepare_daily_comparison(self):
        self.data["day_of_week"] = self.data["datetime"].dt.day_name()
        days_order = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        self.data["day_of_week"] = pd.Categorical(
            self.data["day_of_week"], categories=days_order, ordered=True
        )
        result = self.data.groupby(["day_of_week", "is_true"]).size().unstack().fillna(0)
        
        return result
    
    def prepare_category_comparision(self):
        result = self.data["subject"].unique()
        
        return result


class Dataset(Preprocessing):
    def __init__(self) -> None:
        super().__init__()
        
    def load_from_csv(self, filename: str = "Data.csv"):
        """ Loads data from real_fake/ directory to pandas DataFrame """
        dir_name = "real_fake/"
        filepath = os.path.join(dir_name, filename)
        self.data = pd.read_csv(filepath, parse_dates=['datetime'])
        
        return self

    def save_data_to_csv(self, filename: str = "Data_short.csv"):
        """ Saves data to real_fake/ under provided filename. """
        dir_name = "real_fake/"
        filepath = os.path.join(dir_name, filename)
        self.data.to_csv(filepath)
        
        return self
    
        
    def drop_missing_values(self):
        """ Drops missing observations """
        self.data = self.data.dropna()
        
        return self
        
    def truncate_data(self, k: int = 5000):
        """ Truncates data to k operations"""
        
        np.random.seed(1)
        self.data["subject"] = self.data["subject"].replace(
            {
                "News": "worldnews",
                "politics": "politicsNews",
                "Government News": "politicsNews",
                "left-news": "politicsNews",
                "Middle-east": "worldnews",
                "US_News": "politicsNews",
            }
        )
        
        indexes = np.random.randint(0, self.data.shape[0], k)
        self.data = self.data.iloc[indexes, :]
        
        return self
    
    def get_train_test(
        self,
        label: str = "is_true",
        test_size: int = 0.2,
        random: bool = True,
        random_state: int = None,
    ):
        """
        Returns shuffled dataset divided to train, test and with extraced label column
        Args:
            ...
        Return:
            X_train, X_test, y_train, y_test
        """
        X = self.data.drop(columns=[label])
        y = self.data[label]

        return train_test_split(
            X, y, test_size=test_size, shuffle=random, random_state=random_state
        )
    


            
