import pandas as pd


def add_words_count(data: pd.DataFrame, k: int, column: str) -> pd.DataFrame:
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
    data[column] = data[column].str.lower()
    top_k_words = (
        data[column]
        .str.replace("(\[|\]|\(|\))", "", regex=True)
        .str.split(expand=True)
        .stack()
        .value_counts()
        .drop(to_drop, errors="ignore")
        .head(k)
    )

    for word, _ in top_k_words.items():
        data[word] = data[column].str.count(word)

    return data


def add_day_of_week(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add day of week column which represents day of the week starting from Monday = 0
    Args:
        data : pandas DataFrame
    """
    data["weekday"] = data["datetime"].dt.weekday

    return data


def add_month_of_year(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add month of year column which represents month of the year starting from Januray = 0
    Args:
        data : pandas DataFrame
    """
    data["month"] = data["datetime"].dt.month

    return data
