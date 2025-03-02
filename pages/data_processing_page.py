from PyQt5.QtWidgets import (
    QWidget,
    QComboBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QPushButton,
    QFileDialog,
)
from pages.style import CenterDelegate, CenteredComboBoxStyle
from pages.config import GlobalData
from scipy.ndimage import gaussian_filter1d
import pandas as pd


class DataProcessingPage(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["移动平均滤波", "中值滤波", "指数平滑", "高斯滤波"])
        self.filter_combo.setItemDelegate(CenterDelegate(self.filter_combo))
        self.filter_combo.setStyle(CenteredComboBoxStyle())
        layout.addWidget(self.filter_combo)

        self.upload_btn = QPushButton("上传Excel文件")
        self.upload_btn.clicked.connect(self.upload_file)
        layout.addWidget(self.upload_btn)

        self.filter_btn = QPushButton("应用滤波算法处理")
        self.filter_btn.clicked.connect(self.apply_filter)
        layout.addWidget(self.filter_btn)

        self.download_btn = QPushButton("下载处理后数据")
        self.download_btn.clicked.connect(self.download_filtered_data)
        layout.addWidget(self.download_btn)

        self.table_widget = QTableWidget()
        layout.addWidget(self.table_widget)

        self.setLayout(layout)

    def upload_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择Excel文件", "", "Excel 文件 (*.xlsx *.xls)"
        )
        if file_path:
            df = pd.read_excel(file_path)
            GlobalData.df = df.copy()
            GlobalData.filtered_df = df.copy()
            self.display_table(df)

    def apply_filter(self):
        """filter algorithms"""
        if GlobalData.filtered_df is not None:
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

            GlobalData.filtered_df = df
            self.display_table(df)

    def download_filtered_data(self):
        if GlobalData.filtered_df is not None:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "保存Excel文件", "filtered_data.xlsx", "Excel 文件 (*.xlsx)"
            )
            if file_path:
                GlobalData.filtered_df.to_excel(file_path, index=False)

    def display_table(self, df):
        self.table_widget.setRowCount(df.shape[0])
        self.table_widget.setColumnCount(df.shape[1])
        self.table_widget.setHorizontalHeaderLabels(df.columns)

        for row in range(df.shape[0]):
            for col in range(df.shape[1]):
                item = QTableWidgetItem(str(df.iloc[row, col]))
                self.table_widget.setItem(row, col, item)
