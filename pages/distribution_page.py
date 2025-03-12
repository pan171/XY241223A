from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
    QFileDialog,
    QLabel,
    QLineEdit,
    QApplication,
    QMessageBox,
    QHBoxLayout,
    QScrollArea,
)
from PyQt5.QtGui import QPixmap
import matplotlib.pyplot as plt
import os
import pandas as pd
import shutil

from pages.config import GlobalData, resource_path

import platform

plt.rcParams["axes.unicode_minus"] = False

if platform.system() == "Windows":
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]  # Windows
elif platform.system() == "Darwin":  # macOS
    plt.rcParams["font.sans-serif"] = ["Heiti TC"]  # Mac
else:
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC"]  # Linux


class DistributionPage(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.layout = QVBoxLayout()

        self.upload_btn = QPushButton("上传 Excel 文件")
        self.upload_btn.clicked.connect(self.upload_file)
        self.layout.addWidget(self.upload_btn)

        depth_layout = QHBoxLayout()
        self.start_depth_label = QLabel("start_depth: ")
        self.start_depth_input = QLineEdit("500")
        self.start_depth_input.setPlaceholderText("起始深度")
        self.end_depth_label = QLabel("end_depth: ")
        self.end_depth_input = QLineEdit("1000")
        self.end_depth_input.setPlaceholderText("结束深度")

        depth_layout.addWidget(self.start_depth_label)
        depth_layout.addWidget(self.start_depth_input)
        depth_layout.addWidget(self.end_depth_label)
        depth_layout.addWidget(self.end_depth_input)

        self.layout.addLayout(depth_layout)

        self.identify_btn = QPushButton("绘制录井数据分布图")
        self.identify_btn.clicked.connect(self.run_data_distribution)
        self.layout.addWidget(self.identify_btn)

        self.status_label = QLabel("状态: 等待操作")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setFixedHeight(25)
        self.status_label.setStyleSheet("font-size: 12px; padding: 2px;")
        self.layout.addWidget(self.status_label)

        # Create a scroll area for the image
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setMinimumHeight(300)  # Set minimum height

        # Create a container widget for the image
        self.image_container = QWidget()
        self.image_layout = QVBoxLayout(self.image_container)

        self.image_label = QLabel("录井数据分布图结果")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(
            580, 280
        )  # Set minimum size for the image label
        self.image_layout.addWidget(self.image_label)

        self.scroll_area.setWidget(self.image_container)
        self.layout.addWidget(self.scroll_area)

        self.current_file_path = None

        self.download_btn = QPushButton("下载图片")
        self.download_btn.clicked.connect(self.download_image)
        self.layout.addWidget(self.download_btn)

        self.setLayout(self.layout)
        self.setMinimumSize(600, 600)  # Set minimum window size

    def upload_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择 Excel 文件", "", "Excel 文件 (*.xlsx *.xls)"
        )
        if file_path:
            self.current_file_path = file_path
            df = pd.read_excel(file_path)
            GlobalData.df = df.copy()
            GlobalData.filtered_df = df.copy()
            self.status_label.setText("上传文件成功！")

    def download_image(self):
        if not hasattr(self, "image_label") or self.image_label.pixmap() is None:
            QMessageBox.warning(
                self, "警告", "没有可下载的图片，请先运行录井数据分布图。"
            )
            return

        pdf_path, _ = QFileDialog.getSaveFileName(
            self, "保存PDF文件", "", "PDF 文件 (*.pdf)"
        )

        if pdf_path:
            try:
                # 直接复制之前生成的 PDF 文件
                if hasattr(self, "current_pdf_path") and os.path.exists(
                    self.current_pdf_path
                ):
                    shutil.copy2(self.current_pdf_path, pdf_path)
                    QMessageBox.information(
                        self, "成功", f"图片已成功保存为可编辑的PDF：\n{pdf_path}"
                    )
                else:
                    QMessageBox.warning(
                        self, "警告", "找不到PDF文件，请重新运行录井数据分布图。"
                    )
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存PDF文件时出错：{str(e)}")

    def run_data_distribution(self):
        QApplication.processEvents()

        start_depth = float(self.start_depth_input.text())
        end_depth = float(self.end_depth_input.text())

        file_path = self.current_file_path if self.current_file_path else None
        image_path, pdf_path = self.plot_resistivity_log(
            file_path, start_depth, end_depth
        )

        if image_path:
            self.show_image(image_path)
            self.current_pdf_path = pdf_path
            self.status_label.setText("状态: 处理完成 ✅")
        else:
            self.status_label.setText("状态: 处理失败 ❌")

    def plot_resistivity_log(self, file_path=None, start_depth=500, end_depth=1000):
        self.status_label.setText("状态: 正在处理...")
        if file_path:
            df = pd.read_excel(file_path)
        else:
            df = GlobalData.filtered_df

        if df is None:
            self.status_label.setText("❌ 数据为空，请上传或加载数据。")
            return None, None

        required_columns = [
            "井深",
            "漏失速度",
            "漏失量",
            "塑性黏度",
            "钻速",
            "钻井液排量",
            "泵压",
            "裂缝宽度",
        ]
        if not all(col in df.columns for col in required_columns):
            QMessageBox.critical(
                self, "错误", f"输入文件必须包含列: {required_columns}"
            )
            return None, None

        df_section = df[(df["井深"] >= start_depth) & (df["井深"] < end_depth)]

        if df_section.empty:
            QMessageBox.warning(self, "警告", "所选深度范围内无数据。")
            return None, None

        # Create a figure with 2 rows - scatter plots on top and histograms below
        fig, axes = plt.subplots(nrows=2, ncols=7, figsize=(24, 16))

        # Top row for scatter plots (axes[0])
        scatter_axes = axes[0]
        for ax in scatter_axes:
            ax.set_ylim(bottom=max(df_section["井深"]), top=min(df_section["井深"]))

        # fig1: 漏失速度 - blue
        scatter_axes[0].scatter(
            df_section["漏失速度"], df_section["井深"], color="blue"
        )
        scatter_axes[0].grid(linestyle="--", alpha=0.5)
        scatter_axes[0].set_xlabel("漏失速度")
        scatter_axes[0].set_ylabel("井深 (m)")
        scatter_axes[0].set_title("漏失速度 (散点图)")

        scatter_axes[1].scatter(df_section["漏失量"], df_section["井深"], color="green")
        scatter_axes[1].grid(linestyle="--", alpha=0.5)
        scatter_axes[1].set_xlabel("漏失量")
        scatter_axes[1].set_title("漏失量 (散点图)")

        scatter_axes[2].scatter(df_section["塑性黏度"], df_section["井深"], color="red")
        scatter_axes[2].grid(linestyle="--", alpha=0.5)
        scatter_axes[2].set_xlabel("塑性黏度")
        scatter_axes[2].set_title("塑性黏度 (散点图)")

        scatter_axes[3].scatter(df_section["钻速"], df_section["井深"], color="purple")
        scatter_axes[3].grid(linestyle="--", alpha=0.5)
        scatter_axes[3].set_xlabel("钻速")
        scatter_axes[3].set_title("钻速 (散点图)")

        scatter_axes[4].scatter(
            df_section["钻井液排量"], df_section["井深"], color="orange"
        )
        scatter_axes[4].grid(linestyle="--", alpha=0.5)
        scatter_axes[4].set_xlabel("钻井液排量")
        scatter_axes[4].set_title("钻井液排量 (散点图)")

        scatter_axes[5].scatter(df_section["泵压"], df_section["井深"], color="cyan")
        scatter_axes[5].grid(linestyle="--", alpha=0.5)
        scatter_axes[5].set_xlabel("泵压")
        scatter_axes[5].set_title("泵压 (散点图)")

        scatter_axes[6].scatter(
            df_section["裂缝宽度"], df_section["井深"], color="magenta"
        )
        scatter_axes[6].grid(linestyle="--", alpha=0.5)
        scatter_axes[6].set_xlabel("裂缝宽度")
        scatter_axes[6].set_title("裂缝宽度 (散点图)")

        # 绘制直方图
        hist_axes = axes[1]
        hist_axes[0].hist(df_section["漏失速度"], bins=15, color="blue", alpha=0.7)
        hist_axes[0].grid(linestyle="--", alpha=0.5)
        hist_axes[0].set_xlabel("漏失速度")
        hist_axes[0].set_ylabel("频次")
        hist_axes[0].set_title("漏失速度 (直方图)")

        hist_axes[1].hist(df_section["漏失量"], bins=15, color="green", alpha=0.7)
        hist_axes[1].grid(linestyle="--", alpha=0.5)
        hist_axes[1].set_xlabel("漏失量")
        hist_axes[1].set_title("漏失量 (直方图)")

        hist_axes[2].hist(df_section["塑性黏度"], bins=15, color="red", alpha=0.7)
        hist_axes[2].grid(linestyle="--", alpha=0.5)
        hist_axes[2].set_xlabel("塑性黏度")
        hist_axes[2].set_title("塑性黏度 (直方图)")

        hist_axes[3].hist(df_section["钻速"], bins=15, color="purple", alpha=0.7)
        hist_axes[3].grid(linestyle="--", alpha=0.5)
        hist_axes[3].set_xlabel("钻速")
        hist_axes[3].set_title("钻速 (直方图)")

        hist_axes[4].hist(df_section["钻井液排量"], bins=15, color="orange", alpha=0.7)
        hist_axes[4].grid(linestyle="--", alpha=0.5)
        hist_axes[4].set_xlabel("钻井液排量")
        hist_axes[4].set_title("钻井液排量 (直方图)")

        hist_axes[5].hist(df_section["泵压"], bins=15, color="cyan", alpha=0.7)
        hist_axes[5].grid(linestyle="--", alpha=0.5)
        hist_axes[5].set_xlabel("泵压")
        hist_axes[5].set_title("泵压 (直方图)")

        hist_axes[6].hist(df_section["裂缝宽度"], bins=15, color="magenta", alpha=0.7)
        hist_axes[6].grid(linestyle="--", alpha=0.5)
        hist_axes[6].set_xlabel("裂缝宽度")
        hist_axes[6].set_title("裂缝宽度 (直方图)")

        plt.suptitle(f"Depth Range: {start_depth}-{end_depth} m", fontsize=16)
        plt.tight_layout()

        # 保存 PNG 和 PDF 版本
        output_dir = resource_path("img/data_distribution/")
        os.makedirs(output_dir, exist_ok=True)

        # 保存 PNG 用于显示
        image_path = os.path.join(output_dir, "data_distribution.png")
        plt.savefig(image_path, dpi=300, bbox_inches="tight")

        # 保存矢量 PDF 用于下载
        pdf_path = os.path.join(output_dir, "data_distribution.pdf")
        plt.savefig(pdf_path, format="pdf", bbox_inches="tight")

        plt.close()

        return image_path, pdf_path

    def show_image(self, image_path):
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            QMessageBox.critical(
                self, "错误", "无法加载生成的图片，请检查文件路径或格式。"
            )
            return

        self.image_label.setPixmap(
            pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio)
        )
