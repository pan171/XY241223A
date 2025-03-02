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
)
from PyQt5.QtGui import QPixmap
import matplotlib.pyplot as plt
import os
import pandas as pd
import zipfile

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

        self.image_label = QLabel("录井数据分布图结果")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.image_label)
        self.image_container = QVBoxLayout()

        self.setLayout(self.layout)
        self.current_file_path = None

        self.download_btn = QPushButton("下载图片")
        self.download_btn.clicked.connect(self.download_image)
        self.layout.addWidget(self.download_btn)

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
            QMessageBox.warning(self, "警告", "没有可下载的图片，请先运行裂缝识别。")
            return

        zip_path, _ = QFileDialog.getSaveFileName(
            self, "保存图片压缩包", "distribution_images.zip", "ZIP 文件 (*.zip)"
        )
        if zip_path:
            with zipfile.ZipFile(zip_path, "w") as zipf:
                image_path = resource_path(
                    "img/data_distribution/data_distribution.png"
                )
                zipf.write(image_path, os.path.basename(image_path))

            QMessageBox.information(self, "成功", f"图片已成功打包至：\n{zip_path}")

    def run_data_distribution(self):
        QApplication.processEvents()

        start_depth = float(self.start_depth_input.text())
        end_depth = float(self.end_depth_input.text())

        file_path = self.current_file_path if self.current_file_path else None
        image_path = self.plot_resistivity_log(file_path, start_depth, end_depth)

        if image_path:
            self.show_image(image_path)
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
            return None

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
            return None

        df_section = df[(df["井深"] >= start_depth) & (df["井深"] < end_depth)]

        if df_section.empty:
            QMessageBox.warning(self, "警告", "所选深度范围内无数据。")
            return None

        _, axes = plt.subplots(nrows=1, ncols=7, figsize=(20, 10), sharey=True)
        for ax in axes:
            ax.set_ylim(bottom=max(df_section["井深"]), top=min(df_section["井深"]))

        # fig1: 漏失速度 - blue
        axes[0].scatter(df_section["漏失速度"], df_section["井深"], color="blue")
        axes[0].grid(linestyle="--", alpha=0.5)
        axes[0].set_xlabel("漏失速度")
        axes[0].set_ylabel("井深 (m)")
        axes[0].set_title("漏失速度")

        # fig2: 漏失量 - green
        axes[1].scatter(df_section["漏失量"], df_section["井深"], color="green")
        axes[1].grid(linestyle="--", alpha=0.5)
        axes[1].set_xlabel("漏失量")
        axes[1].set_title("漏失量")

        # fig3: 塑性黏度 - red
        axes[2].scatter(df_section["塑性黏度"], df_section["井深"], color="red")
        axes[2].grid(linestyle="--", alpha=0.5)
        axes[2].set_xlabel("塑性黏度")
        axes[2].set_title("塑性黏度")

        # fig4: 钻速 - purple
        axes[3].scatter(df_section["钻速"], df_section["井深"], color="purple")
        axes[3].grid(linestyle="--", alpha=0.5)
        axes[3].set_xlabel("钻速")
        axes[3].set_title("钻速")

        # fig5: 钻井液排量 - orange
        axes[4].scatter(df_section["钻井液排量"], df_section["井深"], color="orange")
        axes[4].grid(linestyle="--", alpha=0.5)
        axes[4].set_xlabel("钻井液排量")
        axes[4].set_title("钻井液排量 ")

        # fig6: 泵压 - cyan
        axes[5].scatter(df_section["泵压"], df_section["井深"], color="cyan")
        axes[5].grid(linestyle="--", alpha=0.5)
        axes[5].set_xlabel("泵压")
        axes[5].set_title("泵压")

        # fig7: 裂缝宽度 - magenta
        axes[6].scatter(df_section["裂缝宽度"], df_section["井深"], color="magenta")
        axes[6].grid(linestyle="--", alpha=0.5)
        axes[6].set_xlabel("裂缝宽度")
        axes[6].set_title("裂缝宽度")

        plt.suptitle(f"Depth Range: {start_depth}-{end_depth} m", fontsize=14)
        plt.tight_layout()

        output_dir = resource_path("img/data_distribution/")
        os.makedirs(output_dir, exist_ok=True)
        image_path = os.path.join(output_dir, "data_distribution.png")
        plt.savefig(image_path, dpi=300, bbox_inches="tight")
        plt.close()

        return image_path

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
