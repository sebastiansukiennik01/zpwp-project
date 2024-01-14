import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


class DecisionTree():
    
    def __init__(self, **kwargs) -> None:
        self.model = DecisionTreeClassifier(**kwargs)
        
        self.pred_y = None
        self.max_depth = 5
        self.accuracy = None
        self.precision = None
        self.recall = None
        
        
    def fit(self, train_X, train_y):
        self.model.fit(train_X, train_y)
    
    def predict(self, test_X, test_y):
        self.pred_y = self.model.predict(test_X)
        self.extract_metrics(test_y)
        
        return self.pred_y


    def extract_metrics(self, real_y):
        report = classification_report(real_y, self.pred_y, output_dict=True)
        self.accuracy = round(report["accuracy"], 3)
        self.precision = round(report["1"]["precision"], 3)
        self.recall = round(report["1"]["recall"], 3)
        
        return self.accuracy, self.precision, self.recall
    
    
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, set_type):
        """
        Plots confusion matrix based on provide true and predicted values.
        """
        cf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred)
        disp = ConfusionMatrixDisplay(cf_matrix)
        disp.plot()
        plt.title(f"Confustion matrix for {set_type} set")
        plt.show()

    def __repr__(self):
        return f"DecisionTree(accuracy: {self.accuracy}, precission: {self.precision}, recall: {self.recall})"

        
    
    
    


    
    
    