import pandas as pd
from dataset import Kaggle, Dataset
from src.visualization import LinearPlot, HistPlot, BarPlot
from decision_tree import DecisionTree

if __name__ == "__main__":
    # # DOWNLOAD
    # k = Kaggle()
    # k.authenticate().download_data()
    
    # loading    
    d = Dataset()
    d.load_from_csv().drop_missing_values().truncate_data().save_data_to_csv()
    
    # perform preprocessing
    d.add_day_of_week().add_month_of_year().add_words_count()



    # 
    # subject_counts = d.data["subject"].unique()
    # hp = HistPlot().plot(x=subject_counts, 
    #                      y=d.data["subject"], 
    #                      bins=3, 
    #                      xlabel="Kategoria",
    #                      ylabel="Liczebność")
    
    
    
    
    # d.data["day_of_week"] = d.data["datetime"].dt.day_name()
    # days_order = [
    #     "Monday",
    #     "Tuesday",
    #     "Wednesday",
    #     "Thursday",
    #     "Friday",
    #     "Saturday",
    #     "Sunday",
    # ]
    # d.data["day_of_week"] = pd.Categorical(
    #     d.data["day_of_week"], categories=days_order, ordered=True
    # )
    # result = d.data.groupby(["day_of_week", "is_true"]).size().unstack().fillna(0)

    
    bp = BarPlot()
    # bp.plot(data=result, title="Liczebność fake newsów w zależności od dnia tygodnia")
    
    
    # data = d.data.loc[d.data["datetime"].dt.year != 2015]
    # data["month"] = data["datetime"].dt.to_period("M")
    # result = data.groupby(["month", "is_true"]).size().unstack().fillna(0)

    # bp.plot(data=result, title="Liczebność fake newsów z podziełem na miesiące")
    
    
    # words = list(filter(lambda x: f"title_" in x, d.data.columns))
    # x_labels = [w.replace(f"title_", "") for w in words]
    # x = list(range(0, len(x_labels)))
    # result = d.data[words].sum()
    # bp.plot(data=result, title="Liczebność true newsów z podziełem na miesiące")


    # words = list(filter(lambda x: f"text_" in x, d.data.columns))
    # x_labels = [w.replace(f"text_", "") for w in words]
    # x = list(range(0, len(x_labels)))
    # result = d.data[words].sum()
    # bp.plot(data=result, title="Liczebność true newsów z podziełem na miesiące")

    
    # data = d.data.loc[d.data["datetime"].dt.year != 2015]
    # data["day"] = data["datetime"].dt.to_period("D")
    # result = data.groupby(["day", "is_true"]).size().unstack().fillna(0)
    
    # lp = LinearPlot()
    # lp.plot(data=result)
    
    
    
    d.data = d.data.drop(columns=['title', 'text', 'subject', 'date', 'datetime', 'weekday'])
    train_X, test_X, train_y, test_y = d.get_train_test()
    
    dt = DecisionTree()
    dt.fit(train_X, train_y)
    pred_y = dt.predict(test_X, test_y)
    DecisionTree.plot_confusion_matrix(test_y, pred_y, set_type="test")
    
    print(dt)

    
    
    