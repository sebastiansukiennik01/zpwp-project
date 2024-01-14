import os
from typing import Iterable
from abc import ABC, abstractmethod

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import datetime as dt


class Plot(ABC):
    def __init__(
        self,
        legend: bool = True,
        w: int = 10,
        h: int = 4,
        default_style: str = "default",
        **kwargs,
    ) -> None:
        self.fig, self.ax = plt.subplots(1, 1)
        self.w = w
        self.h = h
        self.legend = legend
        self.default_style = default_style
        self.title_font_dict = kwargs.pop(
            "title_font_dict", {"size": 20, "weight": "bold"}
        )
        self.labels_font_dict = kwargs.pop("labels_font_dict", {"size": 16})

    def __post_init__(self):
        plt.style.use(self.default_style)
        matplotlib.rcParams["lines.linewidth"] = 2
        self.ax.figure.set_size_inches(self.w, self.h)

    @abstractmethod
    def plot() -> plt.axes:
        ...

    def _save_figure(self, file_name: str) -> None:
        """
        Saves figure to 'images/" directory.
        """
        plt.show()
        print(f"saving: {file_name}")
        if file_name == "":
            file_name = dt.datetime.today().strftime("%Y%m%d_%H%M%S")
        file_name = file_name.lower().replace(" ", "_")
        self.fig.savefig(f"images/{file_name}.png", dpi=100)


class LinearPlot(Plot):
    def __init__(self, legend: bool = True, w: int = 10, h: int = 4) -> None:
        super().__init__(legend, w, h)

    def plot(self, data: pd.DataFrame, **kwargs) -> plt.axes:
        """
        Plots data onto provided axes. If none axes provided new one is created.
        Defualt style is set to 'fivethirtyeight' but can be changed via kwargs argument.
        args:
            x : iteravale data ploted onto X axis
            y : iteravale data ploted onto Y axis
        return : ax with ploted linear data
        """
        xlabel = kwargs.pop("xlabel", None)
        ylabel = kwargs.pop("ylabel", None)
        title = kwargs.pop("title", self.ax.get_title())
        
        try:
            self.ax.plot(
                data.index, data.values, label=kwargs.pop("label", ""), color=kwargs.pop("color", None), **kwargs
            )
        except TypeError:
            self.ax.plot(
                range(len(data.index)), data.values, label=kwargs.pop("label", ""), color=kwargs.pop("color", None), **kwargs
            )
            self.ax.set_xticklabels(data.index.strftime('%Y-%m'), rotation=90, ha='right')
            
        self.ax.set_title(title, fontdict=self.title_font_dict)
        self.ax.figure.set_size_inches(self.w, self.h)
        self.ax.set_xlabel(xlabel, fontdict=self.labels_font_dict)
        self.ax.set_ylabel(ylabel, fontdict=self.labels_font_dict)
        if self.legend:
            plt.legend()

        self._save_figure(title)

        return self.ax


class HistPlot(Plot):
    def __init__(self, legend: bool = True, w: int = 10, h: int = 4) -> None:
        super().__init__(legend, w, h)

    def plot(self, x: Iterable, y: Iterable, **kwargs) -> plt.axes:
        """
        Plots data onto provided axes. If none axes provided new one is created.
        Defualt style is set to 'fivethirtyeight' but can be changed via kwargs argument.
        args:
            y : iteravale data ploted onto Y axis
        return : ax with ploted linear data
        """
        xlabel = kwargs.pop("xlabel", None)
        ylabel = kwargs.pop("ylabel", None)
        title = kwargs.pop("title", "")
        bins = kwargs.pop("bins", 20)

        self.ax.hist(y, bins=bins)
        self.ax.set_title(title, fontdict=self.title_font_dict)
        self.ax.set_xlabel(xlabel, fontdict=self.labels_font_dict)
        self.ax.set_ylabel(ylabel, fontdict=self.labels_font_dict)

        if self.legend:
            plt.legend()

        self._save_figure(title)

        return self.ax


class BarPlot(Plot):
    def __init__(self, legend: bool = True, w: int = 10, h: int = 4) -> None:
        super().__init__(legend, w, h)

    def plot(self, data: pd.DataFrame, **kwargs) -> plt.axes:
        """
        Plots data onto provided axes. If none axes provided new one is created.
        Defualt style is set to 'fivethirtyeight' but can be changed via kwargs argument.
        args:
            x : iteravale data ploted onto X axis
            y : iteravale data ploted onto Y axis
        return : ax with ploted linear data
        """
        xlabel = kwargs.pop("xlabel", None)
        ylabel = kwargs.pop("ylabel", None)
        title = kwargs.pop("title", "")
        
        data = pd.DataFrame(data)
 
        
        # Stacked bar plot
        bottom = None
        for is_true in data.columns:
            try:
                self.ax.bar(data.index, data[is_true], label=f'is_true {is_true}', bottom=bottom)
            except TypeError:
                self.ax.bar(range(len(data.index)), data[is_true], label=f'is_true {is_true}', bottom=bottom)
                self.ax.set_xticks(range(len(data.index)))
                self.ax.set_xticklabels(data.index.strftime('%Y-%m'), rotation=90, ha='right')
                
            if bottom is None:
                bottom = data[is_true]
            else:
                bottom += data[is_true]
                
        self.ax.set_title(title, fontdict=self.title_font_dict)
        self.ax.figure.set_size_inches(self.w, self.h)
        self.ax.set_xlabel(xlabel, fontdict=self.labels_font_dict)
        self.ax.set_ylabel(ylabel, fontdict=self.labels_font_dict)

        if self.legend:
            plt.legend()

        self._save_figure(title)

        return self.ax

