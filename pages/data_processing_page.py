from PyQt5.QtWidgets import (
    QWidget,
    QComboBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QFileDialog,
    QMessageBox,
    QSplitter,
    QLabel,
    QSizePolicy,
)
from PyQt5.QtCore import Qt
from pages.style import CenterDelegate, CenteredComboBoxStyle
from pages.config import GlobalData
from scipy.ndimage import gaussian_filter1d
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import os


class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)

        super(PlotCanvas, self).__init__(self.fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def clear_plot(self):
        self.axes.clear()
        self.draw()

    def plot_comparison(self, original_df, filtered_df, column=None):
        self.axes.clear()

        if column is None and not original_df.empty:
            # If no column specified, use the first numeric column
            for col in original_df.columns:
                if pd.api.types.is_numeric_dtype(original_df[col]):
                    column = col
                    break

        if column is not None:
            x = np.arange(len(original_df))
            self.axes.plot(x, original_df[column], "b-", label="原始数据")
            self.axes.plot(x, filtered_df[column], "r-", label="处理后数据")
            self.axes.set_title(f"数据对比 - {column}")
            self.axes.set_xlabel("数据点")
            self.axes.set_ylabel("值")
            self.axes.legend()
            self.axes.grid(True)

        self.fig.tight_layout()
        self.draw()


class DataProcessingPage(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.current_plot_column = None

    def initUI(self):
        main_layout = QHBoxLayout()

        # Left panel for controls and table
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        self.filter_combo = QComboBox()
        self.filter_combo.addItems(
            ["移动平均滤波", "中值滤波", "指数平滑", "高斯滤波", "归一化"]
        )
        self.filter_combo.setItemDelegate(CenterDelegate(self.filter_combo))
        self.filter_combo.setStyle(CenteredComboBoxStyle())
        left_layout.addWidget(self.filter_combo)

        self.upload_btn = QPushButton("上传Excel文件")
        self.upload_btn.clicked.connect(self.upload_file)
        left_layout.addWidget(self.upload_btn)

        self.filter_btn = QPushButton("应用滤波算法处理")
        self.filter_btn.clicked.connect(self.apply_filter)
        left_layout.addWidget(self.filter_btn)

        self.download_btn = QPushButton("下载处理后数据")
        self.download_btn.clicked.connect(self.download_filtered_data)
        left_layout.addWidget(self.download_btn)

        self.table_widget = QTableWidget()
        self.table_widget.cellClicked.connect(self.on_cell_click)
        left_layout.addWidget(self.table_widget)

        # Right panel for plot
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        plot_label = QLabel("数据可视化")
        plot_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(plot_label)

        self.plot_canvas = PlotCanvas(self)
        right_layout.addWidget(self.plot_canvas)

        self.column_combo = QComboBox()
        self.column_combo.setItemDelegate(CenterDelegate(self.column_combo))
        self.column_combo.setStyle(CenteredComboBoxStyle())
        self.column_combo.currentTextChanged.connect(self.update_plot)
        right_layout.addWidget(self.column_combo)

        # Add both panels to main layout with splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([500, 500])  # Equal initial sizes

        main_layout.addWidget(splitter)
        self.setLayout(main_layout)

    def upload_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择Excel文件", "", "Excel 文件 (*.xlsx *.xls)"
        )
        if file_path:
            try:
                df = pd.read_excel(file_path)
                GlobalData.df = df.copy()
                GlobalData.filtered_df = df.copy()
                self.display_table(df)

                # Update column combo box for plotting
                self.update_column_combo(df)

                # Initialize the plot
                self.update_plot()

                QMessageBox.information(self, "成功", "文件上传成功！")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"文件上传失败: {str(e)}")

    def update_column_combo(self, df):
        self.column_combo.clear()
        numeric_columns = [
            col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])
        ]
        self.column_combo.addItems(numeric_columns)
        if numeric_columns:
            self.current_plot_column = numeric_columns[0]

    def update_plot(self):
        if GlobalData.df is not None and GlobalData.filtered_df is not None:
            column = self.column_combo.currentText()
            if column:
                self.current_plot_column = column
                self.plot_canvas.plot_comparison(
                    GlobalData.df, GlobalData.filtered_df, column
                )

    def on_cell_click(self, row, col):
        if GlobalData.df is not None:
            column_name = self.table_widget.horizontalHeaderItem(col).text()
            if pd.api.types.is_numeric_dtype(GlobalData.df[column_name]):
                self.column_combo.setCurrentText(column_name)

    def apply_filter(self):
        """filter algorithms"""
        if GlobalData.filtered_df is not None:
            try:
                df = GlobalData.filtered_df.copy()
                selected_filter = self.filter_combo.currentText()

                if selected_filter == "移动平均滤波":
                    df = df.rolling(window=3, min_periods=1).mean()

                elif selected_filter == "中值滤波":
                    df = df.rolling(window=3, min_periods=1).median()

                elif selected_filter == "指数平滑":
                    df = df.ewm(span=3, adjust=False).mean()

                elif selected_filter == "高斯滤波":
                    for col in df.columns:
                        if pd.api.types.is_numeric_dtype(df[col]):
                            df[col] = gaussian_filter1d(df[col].values, sigma=1)
                elif selected_filter == "归一化":
                    for col in df.columns:
                        if pd.api.types.is_numeric_dtype(df[col]):
                            df[col] = (df[col] - df[col].min()) / (
                                df[col].max() - df[col].min()
                            )

                GlobalData.filtered_df = df
                self.display_table(df)

                # Update plot with new filtered data
                self.update_plot()

                QMessageBox.information(
                    self, "成功", f"已成功应用{selected_filter}算法！"
                )
            except Exception as e:
                QMessageBox.critical(self, "错误", f"数据处理失败: {str(e)}")
        else:
            QMessageBox.warning(self, "警告", "请先上传数据文件！")

    def download_filtered_data(self):
        if GlobalData.filtered_df is not None:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "保存Excel文件", "filtered_data.xlsx", "Excel 文件 (*.xlsx)"
            )
            if file_path:
                try:
                    # Save Excel data
                    GlobalData.filtered_df.to_excel(file_path, index=False)

                    # Save plot as image
                    figure_path = os.path.splitext(file_path)[0] + ".png"
                    self.plot_canvas.fig.savefig(
                        figure_path, dpi=300, bbox_inches="tight"
                    )

                    QMessageBox.information(
                        self, "成功", f"数据已成功保存！\n图像已保存为: {figure_path}"
                    )
                except Exception as e:
                    QMessageBox.critical(self, "错误", f"保存文件失败: {str(e)}")
        else:
            QMessageBox.warning(self, "警告", "没有可下载的数据！")

    def display_table(self, df):
        self.table_widget.setRowCount(df.shape[0])
        self.table_widget.setColumnCount(df.shape[1])
        self.table_widget.setHorizontalHeaderLabels(df.columns)

        for row in range(df.shape[0]):
            for col in range(df.shape[1]):
                item = QTableWidgetItem(str(df.iloc[row, col]))
                self.table_widget.setItem(row, col, item)
