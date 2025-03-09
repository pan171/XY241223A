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
import zipfile
import numpy as np

from pages.config import GlobalData, resource_path


class ParameterAnalysis(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.layout = QVBoxLayout()

        self.upload_btn = QPushButton("上传 Excel 文件")
        self.upload_btn.clicked.connect(self.upload_file)
        self.layout.addWidget(self.upload_btn)

        ##################### Line 1 ###########################
        comb_1 = QHBoxLayout()

        # parameters: start, end
        self.start_depth_label = QLabel("start_depth: ")
        self.start_depth_input = QLineEdit("500")
        self.start_depth_input.setPlaceholderText("起始深度")
        # self.start_depth_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        comb_1.addWidget(self.start_depth_label)
        comb_1.addWidget(self.start_depth_input)

        self.end_depth_label = QLabel("end_depth: ")
        self.end_depth_input = QLineEdit("1000")
        self.end_depth_input.setPlaceholderText("结束深度")
        # self.end_depth_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        comb_1.addWidget(self.end_depth_label)
        comb_1.addWidget(self.end_depth_input)

        # parameters: 𝛷, Swi, b, Rmf
        self.omega_label = QLabel("omega: ")
        self.omega_input = QLineEdit("1")
        self.omega_input.setPlaceholderText("omega")
        comb_1.addWidget(self.omega_label)
        comb_1.addWidget(self.omega_input)

        self.s_wi_label = QLabel("s_wi: ")
        self.s_wi_input = QLineEdit("1")
        self.s_wi_input.setPlaceholderText("s_wi")
        comb_1.addWidget(self.s_wi_label)
        comb_1.addWidget(self.s_wi_input)

        self.b_label = QLabel("b: ")
        self.b_input = QLineEdit("1")
        self.b_input.setPlaceholderText("b")
        comb_1.addWidget(self.b_label)
        comb_1.addWidget(self.b_input)

        self.r_mf_label = QLabel("r_mf: ")
        self.r_mf_input = QLineEdit("0.1")
        self.r_mf_input.setPlaceholderText("r_mf")
        comb_1.addWidget(self.r_mf_label)
        comb_1.addWidget(self.r_mf_input)

        self.layout.addLayout(comb_1)
        ################################################

        ##################### Line 2 ###########################
        # parameters: a, Rw, c1, c2, c3
        comb_2 = QHBoxLayout()

        self.a_label = QLabel("a: ")
        self.a_input = QLineEdit("0.5")
        self.a_input.setPlaceholderText("a")
        comb_2.addWidget(self.a_label)
        comb_2.addWidget(self.a_input)

        self.r_w_label = QLabel("r_w: ")
        self.r_w_input = QLineEdit("1")
        self.r_w_input.setPlaceholderText("r_w")
        comb_2.addWidget(self.r_w_label)
        comb_2.addWidget(self.r_w_input)

        self.c1_label = QLabel("c1: ")
        self.c1_input = QLineEdit("1")
        self.c1_input.setPlaceholderText("c1")
        comb_2.addWidget(self.c1_label)
        comb_2.addWidget(self.c1_input)

        self.c2_label = QLabel("c2: ")
        self.c2_input = QLineEdit("1")
        self.c2_input.setPlaceholderText("c2")
        comb_2.addWidget(self.c2_label)
        comb_2.addWidget(self.c2_input)

        self.c3_label = QLabel("c3: ")
        self.c3_input = QLineEdit("1")
        self.c3_input.setPlaceholderText("c3")
        comb_2.addWidget(self.c3_label)
        comb_2.addWidget(self.c3_input)

        self.layout.addLayout(comb_2)

        ################################################

        self.identify_btn = QPushButton("裂缝通道参数直方图绘制")
        self.identify_btn.clicked.connect(self.run_parameter_analysis)
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

        self.image_label = QLabel("裂缝通道参数直方图绘制结果")
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
                self, "警告", "没有可下载的图片，请先运行裂缝通道参数计算。"
            )
            return

        zip_path, _ = QFileDialog.getSaveFileName(
            self, "保存图片压缩包", "parameter_analysis.zip", "ZIP 文件 (*.zip)"
        )
        if zip_path:
            with zipfile.ZipFile(zip_path, "w") as zipf:
                image_path = resource_path(
                    "img/parameter_analysis/parameter_analysis.png"
                )
                zipf.write(image_path, os.path.basename(image_path))

            QMessageBox.information(self, "成功", f"图片已成功打包至：\n{zip_path}")

    def run_parameter_analysis(self):
        QApplication.processEvents()

        start_depth = float(self.start_depth_input.text())
        end_depth = float(self.end_depth_input.text())

        omega_ = float(self.omega_input.text())
        s_wi_ = float(self.s_wi_input.text())
        b_ = float(self.b_input.text())

        r_mf_ = float(self.r_mf_input.text())
        a_ = float(self.a_input.text())
        r_w_ = float(self.r_w_input.text())

        c1_ = float(self.c1_input.text())
        c2_ = float(self.c2_input.text())
        c3_ = float(self.c3_input.text())

        file_path = self.current_file_path if self.current_file_path else None
        image_path = self.plot_parameter_distribution(
            file_path,
            start_depth,
            end_depth,
            omega_,
            s_wi_,
            b_,
            r_mf_,
            a_,
            r_w_,
            c1_,
            c2_,
            c3_,
        )

        if image_path:
            self.show_image(image_path)
            self.status_label.setText("状态: 处理完成 ✅")
        else:
            self.status_label.setText("状态: 处理失败 ❌")

    def plot_parameter_distribution(
        self,
        file_path=None,
        start_depth=500,
        end_depth=1000,
        omega=1,
        s_wi=1,
        b=1,
        r_mf=0.1,
        a=0.5,
        r_w=1,
        c1=1,
        c2=1,
        c3=1,
    ):
        self.status_label.setText("状态: 正在处理...")
        if file_path:
            df = pd.read_excel(file_path)
        else:
            df = GlobalData.filtered_df

        if df is None:
            self.status_label.setText("❌ 数据为空，请上传或加载数据。")
            return None

        required_columns = ["Depth", "RLLD", "RLLS"]
        if not all(col in df.columns for col in required_columns):
            QMessageBox.critical(
                self, "错误", f"输入文件必须包含列: {required_columns}"
            )
            return None

        df_section = df[(df["Depth"] >= start_depth) & (df["Depth"] < end_depth)]
        if df_section.empty:
            QMessageBox.warning(self, "警告", "所选深度范围内无数据。")
            return None

        # Calculate parameters
        df_section["FVPA"] = (
            r_mf * ((1 / df_section["RLLS"]) - (1 / df_section["RLLD"]))
        ) ** a
        df_section["FVDC"] = ((1 / df_section["RLLS"]) - (1 / df_section["RLLD"])) / (
            (1 / r_mf) - (1 / r_w)
        )
        df_section["FVA"] = (0.064 / omega) * ((1 - s_wi) * df_section["FVDC"]) ** b
        df_section["Kf"] = (
            1.5 * (10**7) * omega * ((1 - s_wi) * df_section["FVDC"]) ** (2.63)
        )

        # Create frequency distribution plots
        _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

        # Plot FVPA distribution
        fvpa_bins = np.linspace(0, df_section["FVPA"].max(), 6)
        n1, bins1, _ = ax1.hist(
            df_section["FVPA"], bins=fvpa_bins, color="blue", alpha=0.7
        )
        ax1.set_title("裂缝宽度分布特征")
        ax1.set_xlabel("裂缝宽度 (mm)")
        ax1.set_ylabel("频率")
        ax1.grid(True, linestyle="--", alpha=0.5)

        # Add cumulative frequency line for FVPA
        ax1_cum = ax1.twinx()
        cumsum1 = np.cumsum(n1)
        cumsum1_normalized = cumsum1 / cumsum1[-1]
        ax1_cum.plot(bins1[:-1], cumsum1_normalized, color="red", linewidth=2)
        ax1_cum.set_ylabel("累计频率")
        ax1_cum.set_ylim(0, 1)

        # Plot FVA distribution
        fva_bins = np.linspace(0, df_section["FVA"].max(), 6)
        n2, bins2, _ = ax2.hist(
            df_section["FVA"], bins=fva_bins, color="orange", alpha=0.7
        )
        ax2.set_title("裂缝孔隙度分布特征")
        ax2.set_xlabel("裂缝孔隙度 (%)")
        ax2.set_ylabel("频率")
        ax2.grid(True, linestyle="--", alpha=0.5)

        # Add cumulative frequency line for FVA
        ax2_cum = ax2.twinx()
        cumsum2 = np.cumsum(n2)
        cumsum2_normalized = cumsum2 / cumsum2[-1]
        ax2_cum.plot(bins2[:-1], cumsum2_normalized, color="red", linewidth=2)
        ax2_cum.set_ylabel("累计频率")
        ax2_cum.set_ylim(0, 1)

        # Plot FVDC distribution
        fvdc_bins = np.linspace(0, df_section["FVDC"].max(), 6)
        n3, bins3, _ = ax3.hist(
            df_section["FVDC"], bins=fvdc_bins, color="green", alpha=0.7
        )
        ax3.set_title("裂缝密度分布特征")
        ax3.set_xlabel("裂缝密度")
        ax3.set_ylabel("频率")
        ax3.grid(True, linestyle="--", alpha=0.5)

        # Add cumulative frequency line for FVDC
        ax3_cum = ax3.twinx()
        cumsum3 = np.cumsum(n3)
        cumsum3_normalized = cumsum3 / cumsum3[-1]
        ax3_cum.plot(bins3[:-1], cumsum3_normalized, color="red", linewidth=2)
        ax3_cum.set_ylabel("累计频率")
        ax3_cum.set_ylim(0, 1)

        # Plot Kf distribution
        kf_bins = np.linspace(0, df_section["Kf"].max(), 6)
        n4, bins4, _ = ax4.hist(df_section["Kf"], bins=kf_bins, color="red", alpha=0.7)
        ax4.set_title("裂缝渗透率分布特征")
        ax4.set_xlabel("裂缝渗透率")
        ax4.set_ylabel("频率")
        ax4.grid(True, linestyle="--", alpha=0.5)

        # Add cumulative frequency line for Kf
        ax4_cum = ax4.twinx()
        cumsum4 = np.cumsum(n4)
        cumsum4_normalized = cumsum4 / cumsum4[-1]
        ax4_cum.plot(bins4[:-1], cumsum4_normalized, color="blue", linewidth=2)
        ax4_cum.set_ylabel("累计频率")
        ax4_cum.set_ylim(0, 1)

        plt.tight_layout()

        output_dir = resource_path("img/parameter_analysis/")
        os.makedirs(output_dir, exist_ok=True)
        image_path = os.path.join(output_dir, "parameter_analysis.png")
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
