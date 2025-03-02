from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
    QFileDialog,
    QLabel,
    QHBoxLayout,
    QApplication,
    QMessageBox,
)
from PyQt5.QtGui import QPixmap
import matplotlib.pyplot as plt
import numpy as np
import os

import time

import zipfile
from pages.config import GlobalData, resource_path
import pandas as pd


class CrackIdentificationPage(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.layout = QVBoxLayout()

        self.upload_btn = QPushButton("上传 Excel 文件")
        self.upload_btn.clicked.connect(self.upload_file)
        self.layout.addWidget(self.upload_btn)

        self.identify_btn = QPushButton("裂缝识别")
        self.identify_btn.clicked.connect(self.run_fracture_identification)
        self.layout.addWidget(self.identify_btn)

        self.status_label = QLabel("状态: 等待操作")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setFixedHeight(25)
        self.status_label.setStyleSheet("font-size: 12px; padding: 2px;")
        self.layout.addWidget(self.status_label)

        self.image_label = QLabel("裂缝识别结果")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.image_label)

        nav_layout = QHBoxLayout()
        self.prev_btn = QPushButton("上一张")
        self.prev_btn.clicked.connect(self.show_prev_image)
        self.next_btn = QPushButton("下一张")
        self.next_btn.clicked.connect(self.show_next_image)
        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.next_btn)
        self.layout.addLayout(nav_layout)

        self.download_btn = QPushButton("下载所有图片")
        self.download_btn.clicked.connect(self.download_all_images)
        self.layout.addWidget(self.download_btn)

        self.setLayout(self.layout)

        self.image_paths = []
        self.current_image_index = 0
        self.current_file_path = None

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

    def run_fracture_identification(self):
        QApplication.processEvents()

        start_time = time.time()

        file_path = self.current_file_path if self.current_file_path else None
        self.image_paths = self.plot_resistivity_log_in_sections(
            file_path, section_depth=200
        )
        self.current_image_index = 0

        end_time = time.time()
        duration = end_time - start_time

        if self.image_paths:
            self.show_image(self.image_paths[self.current_image_index])
            self.status_label.setText(f"状态: 处理完成 ✅ （耗时: {duration:.2f} 秒）")
        else:
            self.status_label.setText("状态: 处理失败 ❌")

    def download_all_images(self):
        if not self.image_paths:
            QMessageBox.warning(self, "警告", "没有可下载的图片，请先运行裂缝识别。")
            return

        zip_path, _ = QFileDialog.getSaveFileName(
            self, "保存图片压缩包", "fracture_images.zip", "ZIP 文件 (*.zip)"
        )
        if zip_path:
            with zipfile.ZipFile(zip_path, "w") as zipf:
                for img_path in self.image_paths:
                    zipf.write(img_path, os.path.basename(img_path))

            QMessageBox.information(self, "成功", f"图片已成功打包至：\n{zip_path}")

    def plot_resistivity_log_in_sections(self, file_path=None, section_depth=200):
        self.status_label.setText("状态: 正在处理...")
        if file_path:
            df = pd.read_excel(file_path)
        else:
            df = GlobalData.filtered_df

        if df is None:
            self.status_label.setText("❌ 数据为空，请上传或加载数据。")
            print("❌ 数据为空，请上传或加载数据。")
            return []

        required_columns = ["Depth", "RLLD", "RLLS"]
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Input file must contain columns: {required_columns}")

        output_dir = resource_path("img/crack_identification/")
        os.makedirs(output_dir, exist_ok=True)
        image_paths = []

        df["RLLD/RLLS"] = df["RLLD"] / df["RLLS"]
        conditions = [
            df["RLLD/RLLS"] > 1.3,
            (df["RLLD/RLLS"] >= 0.8) & (df["RLLD/RLLS"] <= 1.3),
            df["RLLD/RLLS"] < 0.8,
        ]
        choices = [3, 2, 1]
        df["Explanation"] = np.select(conditions, choices, default=0)

        min_depth = df["Depth"].min()
        max_depth = df["Depth"].max()

        for start_depth in np.arange(min_depth, max_depth, section_depth):
            end_depth = start_depth + section_depth
            df_section = df[(df["Depth"] >= start_depth) & (df["Depth"] < end_depth)]

            _, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 10), sharey=True)
            for ax in axes:
                ax.set_ylim(
                    bottom=max(df_section["Depth"]), top=min(df_section["Depth"])
                )

            axes[0].plot(df_section["RLLD"], df_section["Depth"], color="blue")
            axes[0].grid(linestyle="--", alpha=0.5)
            axes[0].set_xlabel("RLLD (Ω·m)", fontsize=12)
            axes[0].set_ylabel("Depth (m)", fontsize=12)
            axes[0].set_title("RLLD")

            axes[1].plot(df_section["RLLS"], df_section["Depth"], color="green")
            axes[1].grid(linestyle="--", alpha=0.5)
            axes[1].set_xlabel("RLLS (Ω·m)")
            axes[1].set_title("RLLS")

            axes[2].plot(df_section["RLLD/RLLS"], df_section["Depth"], color="red")
            axes[2].grid(linestyle="--", alpha=0.5)

            axes[2].set_xlabel("RLLD/RLLS")
            axes[2].set_title("RLLD/RLLS")

            axes[3].step(
                df_section["Explanation"],
                df_section["Depth"],
                where="post",
                color="purple",
                linewidth=2,
            )
            axes[3].grid(linestyle="--", alpha=0.5)
            axes[3].set_xlabel("Explanation")
            axes[3].set_xticks([1, 2, 3])
            axes[3].set_xticklabels(["Low", "Middle", "High"])
            axes[3].set_title("Fracture Explanation")

            plt.suptitle(f"Depth Range: {start_depth}-{end_depth} m", fontsize=14)
            plt.tight_layout()

            image_path = os.path.join(
                output_dir, f"resistivity_log_{int(start_depth)}_{int(end_depth)}.svg"
            )
            plt.savefig(image_path, dpi=300, bbox_inches="tight")
            plt.close()

            image_paths.append(image_path)

        print(f"✅ 生成了 {len(image_paths)} 张图片，保存在 {output_dir}")
        return image_paths

    def show_image(self, image_path):
        pixmap = QPixmap(image_path)
        self.image_label.setPixmap(
            pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio)
        )

    def show_prev_image(self):
        if self.image_paths and self.current_image_index > 0:
            self.current_image_index -= 1
            self.show_image(self.image_paths[self.current_image_index])

    def show_next_image(self):
        if self.image_paths and self.current_image_index < len(self.image_paths) - 1:
            self.current_image_index += 1
            self.show_image(self.image_paths[self.current_image_index])
