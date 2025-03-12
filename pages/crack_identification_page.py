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
import numpy as np
import os
import pandas as pd
import zipfile
from PIL import Image  # 添加PIL库用于图像处理
import io

from pages.config import GlobalData, resource_path


class CrackIdentificationPage(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.layout = QVBoxLayout()

        # Create horizontal layout for upload buttons
        upload_layout = QHBoxLayout()

        self.upload_btn = QPushButton("上传 Excel 文件")
        self.upload_btn.clicked.connect(self.upload_file)
        upload_layout.addWidget(self.upload_btn)

        # Add the additional upload button
        self.upload_btn2 = QPushButton("Upload")
        self.upload_btn2.clicked.connect(self.upload_file)
        upload_layout.addWidget(self.upload_btn2)

        self.layout.addLayout(upload_layout)

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

        self.identify_btn = QPushButton("裂缝识别")
        self.identify_btn.clicked.connect(self.run_crack_identification)
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

        self.image_label = QLabel("裂缝识别结果")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(
            580, 280
        )  # Set minimum size for the image label
        self.image_layout.addWidget(self.image_label)

        self.scroll_area.setWidget(self.image_container)
        self.layout.addWidget(self.scroll_area)

        self.setLayout(self.layout)
        self.current_file_path = None
        self.setMinimumSize(600, 600)  # Set minimum window size

        self.download_btn = QPushButton("下载图片")
        self.download_btn.clicked.connect(self.download_image)
        self.layout.addWidget(self.download_btn)

    def upload_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择 Excel 文件", "", "Excel 文件 (*.xlsx *.xls)"
        )
        if file_path:
            self.current_file_path = file_path
            # 获取触发按钮的文本
            sender = self.sender()
            button_text = sender.text()

            df = pd.read_excel(file_path)

            if button_text == "Upload":
                # 检查必需的列是否存在
                required_columns = [
                    "CALI",
                    "AC",
                    "DEN",
                    "RLLD",
                    "RLLS",
                    "FPI",
                    "解释结论",
                ]
                missing_columns = [
                    col for col in required_columns if col not in df.columns
                ]
                if missing_columns:
                    QMessageBox.critical(
                        self, "错误", f"文件缺少必需的列: {', '.join(missing_columns)}"
                    )
                    return
                # 直接使用文件中的数据
                GlobalData.df = df.copy()
                GlobalData.filtered_df = df.copy()
            else:  # "上传 Excel 文件"
                GlobalData.df = df.copy()
                GlobalData.filtered_df = df.copy()

            self.status_label.setText("上传文件成功！")

    def download_image(self):
        if not hasattr(self, "image_label") or self.image_label.pixmap() is None:
            QMessageBox.warning(self, "警告", "没有可下载的图片，请先运行裂缝识别。")
            return

        pdf_path, _ = QFileDialog.getSaveFileName(
            self, "保存PDF文件", "crack_identification.pdf", "PDF 文件 (*.pdf)"
        )

        if pdf_path:
            try:
                # 获取原始图像路径
                image_path = resource_path(
                    "img/crack_identification/crack_identification.png"
                )

                # 使用PIL打开图像并保存为PDF
                image = Image.open(image_path)
                image.save(pdf_path, "PDF", resolution=100.0)

                QMessageBox.information(
                    self, "成功", f"图片已成功保存为PDF：\n{pdf_path}"
                )
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存PDF时出错：{str(e)}")

    def run_crack_identification(self):
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

        # 获取触发按钮的文本
        sender = self.sender()
        if isinstance(sender, QPushButton):
            button_text = sender.text()
        else:
            button_text = "上传 Excel 文件"  # 默认处理方式

        if button_text == "Upload":
            required_columns = [
                "CALI",
                "AC",
                "DEN",
                "RLLD",
                "RLLS",
                "FPI",
                "解释结论",
            ]
            if not all(col in df.columns for col in required_columns):
                QMessageBox.critical(
                    self, "错误", f"输入文件必须包含列: {required_columns}"
                )
                return None

            # 直接使用文件中的数据进行绘图
            df_section = df[(df["Depth"] >= start_depth) & (df["Depth"] < end_depth)]
        else:
            # 原有的数据处理逻辑
            required_columns = ["Depth", "RLLD", "RLLS", "CALI", "AC", "DEN"]
            if not all(col in df.columns for col in required_columns):
                QMessageBox.critical(
                    self, "错误", f"输入文件必须包含列: {required_columns}"
                )
                return None

            # 归一化 CALI
            df["CALI_normalized"] = (df["CALI"] - df["CALI"].min()) / (
                df["CALI"].max() - df["CALI"].min()
            )

            # 计算并归一化 AC/DEN
            df["AC_DEN_ratio"] = df["AC"] / df["DEN"]
            df["AC_DEN_normalized"] = (
                df["AC_DEN_ratio"] - df["AC_DEN_ratio"].min()
            ) / (df["AC_DEN_ratio"].max() - df["AC_DEN_ratio"].min())

            # 计算并归一化 deltaR
            df["deltaR"] = abs(df["RLLD"] - df["RLLS"])
            df["deltaR_normalized"] = (df["deltaR"] - df["deltaR"].min()) / (
                df["deltaR"].max() - df["deltaR"].min()
            )

            df["FPI"] = (
                0.5 * df["deltaR_normalized"]
                + 0.3 * df["AC_DEN_normalized"]
                + 0.2 * df["CALI_normalized"]
            )

            # 根据 FPI 值设置 Explanation
            df["Y"] = (df["RLLD"] - df["RLLS"]) / ((df["RLLD"] * df["RLLS"]) ** 0.5)
            conditions = [
                (df["Y"] > 0.1) & (df["FPI"] > 0.2),  # 高裂缝发育程度
                ((df["Y"] < 0.1) & (df["Y"] > 0))
                & (df["FPI"] > 0.2),  # 中等裂缝发育程度
                (df["Y"] < 0) & (df["FPI"] > 0.2),  # 低裂缝发育程度
                (df["FPI"] <= 0.2),  # 无裂缝发育
            ]
            choices = [3, 2, 1, 0]
            df["解释结论"] = np.select(conditions, choices, default=0)

            df_section = df[(df["Depth"] >= start_depth) & (df["Depth"] < end_depth)]

        if df_section.empty:
            QMessageBox.warning(self, "警告", "所选深度范围内无数据。")
            return None

        # 创建7个子图
        fig, axes = plt.subplots(nrows=1, ncols=7, figsize=(24, 10), sharey=True)

        # 设置所有图的深度范围（从上到下依次变大）
        for ax in axes:
            ax.set_ylim(bottom=max(df_section["Depth"]), top=min(df_section["Depth"]))
            ax.grid(linestyle="--", alpha=0.5)

        # 绘制CALI
        axes[0].plot(df_section["CALI"], df_section["Depth"], color="black")
        axes[0].set_xlabel("CALI")
        axes[0].set_ylabel("Depth (m)")
        axes[0].set_title("CALI")

        # 绘制AC
        axes[1].plot(df_section["AC"], df_section["Depth"], color="blue")
        axes[1].set_xlabel("AC")
        axes[1].set_title("AC")

        # 绘制DEN
        axes[2].plot(df_section["DEN"], df_section["Depth"], color="red")
        axes[2].set_xlabel("DEN")
        axes[2].set_title("DEN")

        # 绘制RLLD
        axes[3].plot(df_section["RLLD"], df_section["Depth"], color="darkgreen")
        axes[3].set_xlabel("RLLD (Ω·m)")
        axes[3].set_title("RLLD")

        # 绘制RLLS
        axes[4].plot(df_section["RLLS"], df_section["Depth"], color="purple")
        axes[4].set_xlabel("RLLS (Ω·m)")
        axes[4].set_title("RLLS")

        # 绘制FPI
        axes[5].plot(df_section["FPI"], df_section["Depth"], color="orange")
        axes[5].set_xlabel("FPI")
        axes[5].set_title("FPI")

        # 绘制解释结论（以颜色填充块的方式）
        # 首先创建一个带有深度和宽度的矩形数组
        depths = df_section["Depth"].values
        explanations = df_section["解释结论"].values

        # 定义颜色映射
        colors = {0: "white", 1: "green", 2: "blue", 3: "yellow"}

        # 遍历每个深度点并绘制相应颜色的矩形区域
        for i in range(len(depths) - 1):
            height = depths[i + 1] - depths[i]
            y_pos = depths[i]
            explanation = explanations[i]
            axes[6].add_patch(
                plt.Rectangle((0, y_pos), 3, height, color=colors[explanation])
            )

        # 设置Explanation子图的属性
        axes[6].set_xlim(0, 3)
        axes[6].set_xticks([0, 1, 2, 3])
        axes[6].set_xticklabels(["无", "低", "中", "高"])
        axes[6].set_xlabel("解释结论")
        axes[6].set_title("解释结论")

        plt.suptitle(f"Depth Range: {start_depth}-{end_depth} m", fontsize=14)
        plt.tight_layout()

        output_dir = resource_path("img/crack_identification/")
        os.makedirs(output_dir, exist_ok=True)
        image_path = os.path.join(output_dir, "crack_identification.png")
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
