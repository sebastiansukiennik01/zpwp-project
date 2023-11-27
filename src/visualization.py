import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def plot_histogram_of_subject(data: pd.DataFrame):
    """
    Plots histogram of subject.
    """

    subject_counts = data["subject"].value_counts()
    plt.bar(subject_counts.index, subject_counts)
    plt.xticks(rotation="vertical")
    plt.xlabel("Subject")
    plt.ylabel("Count")
    plt.title("Count of Appearances for Each Subject")
    save_plot(plt)
    plt.show()


def plot_real_fake_news_by_weekday(data: pd.DataFrame):
    """
    Plots barplot of true and fake news in each weekday.
    """
    data["day_of_week"] = data["datetime"].dt.day_name()
    days_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    data["day_of_week"] = pd.Categorical(
        data["day_of_week"], categories=days_order, ordered=True
    )
    result = data.groupby(["day_of_week", "is_true"]).size().unstack().fillna(0)
    result.plot(kind="bar", stacked=True)
    plt.title("Count of Real and Fake News for Each Day of the Week")
    plt.xlabel("Day of the Week")
    plt.ylabel("Count")
    plt.legend(["Fake", "Real"])
    save_plot(plt)
    plt.show()


def plot_real_fake_news_by_month(data: pd.DataFrame):
    """
    Plots barplot of true and fake news in each weekday.
    """

    data = data.loc[data["datetime"].dt.year != 2015]
    data["month"] = data["datetime"].dt.to_period("M")
    result = data.groupby(["month", "is_true"]).size().unstack().fillna(0)

    # Plot the data
    result.plot(kind="bar", stacked=True)
    plt.title("Count of Real and Fake News for Each Month (excluding 2015)")
    plt.xlabel("Month")
    plt.ylabel("Count")
    plt.legend(["Fake", "Real"])
    save_plot(plt)
    plt.show()


def plot_most_frequent_words_in_column(data: pd.DataFrame, column: str):
    """
    Plots how many times word has occured in column.
    """
    words = list(filter(lambda x: f"{column}_" in x, data.columns))
    x_labels = [w.replace(f"{column}_", "") for w in words]
    x = list(range(0, len(x_labels)))
    result = data[words].sum()

    result.plot(kind="bar")
    plt.title(f"Count of 10 most frequent words in {column}")
    plt.xticks(x, labels=x_labels)
    plt.xlabel("Word")
    plt.ylabel("Count")
    save_plot(plt)
    plt.show()


def plot_number_of_news_per_day(data: pd.DataFrame):
    """
    Plots how many times word has occured in column.
    """
    data = data.loc[data["datetime"].dt.year != 2015]
    data["day"] = data["datetime"].dt.to_period("D")
    result = data.groupby(["day", "is_true"]).size().unstack().fillna(0)

    # Plot the data
    result.plot(kind="line", stacked=True)
    plt.title("Count of Real and Fake News for each day")
    plt.xlabel("Month")
    plt.ylabel("Count")
    plt.legend(["Fake", "Real"])
    save_plot(plt)
    plt.show()


def plot_correlation_between_set_set_of_columns(data: pd.DataFrame, column: str):
    """
    Plots correlation between number of words in title/text columns and 'is_true' label.
    """
    words = list(filter(lambda x: f"{column}_" in x, data.columns)) + ["is_true"]

    plt.figure(figsize=(15, 12))
    plt.title(f"Correlation between '{column}' words frequency", fontsize=18)
    sns.heatmap(
        data[words].corr(method="spearman"),
        annot=True,
        annot_kws={"size": 12},
    )
    save_plot(plt)
    plt.show()


def plot_confusion_matrix(y_true, y_pred, set_type):
    """
    Plots confusion matrix based on provide true and predicted values.
    """
    cf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred)
    disp = ConfusionMatrixDisplay(cf_matrix)
    disp.plot()
    plt.title(f"Confustion matrix for {set_type} set")
    save_plot(plt)
    plt.show()


def save_plot(plt):
    """
    Function for saving pyplot figures to 'plots/' directory.
    """
    title = plt.gca().get_title().lower().replace(" ", "_")
    output_path = os.path.join("plots", f"{title}.png")
    plt.savefig(output_path)
    print(f"Saved plot to: {output_path}  :)")
