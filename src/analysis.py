import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from functools import reduce


def add_words_count(
    data: pd.DataFrame, k: int = 10, columns: list = ["title", "text"]
) -> pd.DataFrame:
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
            data[f"{column}_{word}"] = data[column].str.count(word)

    return data


def add_date_info(*funcs):
    def helper(initial):
        return reduce(lambda acc, f: f(acc), reversed(funcs), initial)

    return helper


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


""" DECISSION TREE MODEL """


def get_train_test(
    data: pd.DataFrame,
    label: str = "stroke",
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
    X = data.drop(columns=[label])
    y = data[label]

    return train_test_split(
        X, y, test_size=test_size, shuffle=random, random_state=random_state
    )


def extract_metrics(real_y, pred_y):
    report = classification_report(real_y, pred_y, output_dict=True)
    accuracy = round(report["accuracy"], 3)
    precision = round(report["1"]["precision"], 3)
    recall = round(report["1"]["recall"], 3)

    return accuracy, precision, recall


def plot_metrics(x, accuracy, precision, recall, title):
    """
    Plots accuracy, precision and recall for different hiperparameter values
    """

    plt.title(title)
    plt.plot(x, accuracy, label="accuracy")
    plt.plot(x, precision, label="precision")
    plt.plot(x, recall, label="recall")
    plt.ylim(0, 1)
    plt.legend()
    plt.show()


def test_model_depths(df):
    """
    Test random forest with different number of samples taken to train each tree.
    Args:
        n_samples : (0:1] representing percentage of samples to be taken
    """
    train_x, test_x, train_y, test_y = get_train_test(df, label="is_true", random=False)

    dokladnosc = {"train": [], "test": []}
    czulosc = {"train": [], "test": []}
    precyzyjnosc = {"train": [], "test": []}

    depths = list(range(2, 20))
    for d in depths:
        # create, fit and predict model
        tree = DecisionTreeClassifier(max_depth=d)
        tree.fit(train_x, train_y)
        pred_y_train = tree.predict(train_x)  # predykcja na zbiorze train
        pred_y_test = tree.predict(test_x)  # predykcja na zbiorze test

        tr_dok, tr_prec, tr_czul = extract_metrics(
            train_y, pred_y_train
        )  # dla zbioru treningowego
        dokladnosc["train"].append(tr_dok)
        precyzyjnosc["train"].append(tr_prec)
        czulosc["train"].append(tr_czul)

        ts_dok, ts_prec, ts_czul = extract_metrics(
            test_y, pred_y_test
        )  # dla zbioru testoweg
        dokladnosc["test"].append(ts_dok)
        precyzyjnosc["test"].append(ts_prec)
        czulosc["test"].append(ts_czul)

    plot_metrics(
        x=depths,
        accuracy=dokladnosc["train"],
        precision=precyzyjnosc["train"],
        recall=czulosc["train"],
        title="Wielkość zbioru traningoweg dla zbioru traningowego",
    )
    plot_metrics(
        x=depths,
        accuracy=dokladnosc["test"],
        precision=precyzyjnosc["test"],
        recall=czulosc["test"],
        title="Wielkość zbioru traningoweg dla zbioru testowego",
    )
    print(
        f"Max precyzja (train): {max(precyzyjnosc['train'])} for idx: {precyzyjnosc['train'].index(max(precyzyjnosc['train']))}"
    )
    print(
        f"Max precyzja (test): {max(precyzyjnosc['test'])} for idx: {precyzyjnosc['test'].index(max(precyzyjnosc['test']))}"
    )
