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


class ComprehensiveOutcomePage(QWidget):
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

        self.fmi_upload_btn = QPushButton("上传 FMI")
        self.fmi_upload_btn.clicked.connect(self.upload_fmi_image)
        comb_1.addWidget(self.fmi_upload_btn)

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

        self.identify_btn = QPushButton("绘制测录井综合图")
        self.identify_btn.clicked.connect(self.run_draw_outcome)
        self.layout.addWidget(self.identify_btn)

        self.status_label = QLabel("状态: 等待操作")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setFixedHeight(25)
        self.status_label.setStyleSheet("font-size: 12px; padding: 2px;")
        self.layout.addWidget(self.status_label)

        self.image_label = QLabel("测录井综合图结果")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.image_label)
        self.image_container = QVBoxLayout()

        self.setLayout(self.layout)

        self.current_file_path = None

        self.download_btn = QPushButton("下载图片")
        self.download_btn.clicked.connect(self.download_image)
        self.layout.addWidget(self.download_btn)

        self.setGeometry(100, 100, 600, 400)

    def upload_fmi_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择 FMI 图片", "", "图片文件 (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_path:
            output_dir = resource_path("img/")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "fmi_example.png")
            try:
                with open(file_path, "rb") as src, open(output_path, "wb") as dst:
                    dst.write(src.read())
                QMessageBox.information(self, "成功", "FMI 图片上传成功！")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"图片上传失败: {str(e)}")

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
            self, "保存图片压缩包", "comprehensive_outcome.zip", "ZIP 文件 (*.zip)"
        )
        if zip_path:
            with zipfile.ZipFile(zip_path, "w") as zipf:
                image_path = resource_path(
                    "img/comprehensive_outcome/comprehensive_outcome.png"
                )
                zipf.write(image_path, os.path.basename(image_path))

            QMessageBox.information(self, "成功", f"图片已成功打包至：\n{zip_path}")

    def run_draw_outcome(self):
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
        image_path = self.plot_parameter_log(
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

    def plot_parameter_log(
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

        required_columns = ["Depth", "AC", "DEN", "GR", "CNL", "RLLD", "RLLS"]
        if not all(col in df.columns for col in required_columns):
            QMessageBox.critical(
                self, "错误", f"输入文件必须包含列: {required_columns}"
            )
            return None

        # directly obtain
        # x: Depth
        # fig1: AC —— blue
        # fig2: DEN —— green
        # fig3: GR —— red
        # fig4: CNL —— yellow
        # fig5: RLLD & RLLS —— purple & orange
        # calc
        # fig6: FVA & FVPA —— black & brown
        # fig7: FVDC & FG —— cyan & magenta
        df["FVPA"] = (r_mf * ((1 / df["RLLS"]) - (1 / df["RLLD"]))) ** a
        df["FVDC"] = ((1 / df["RLLS"]) - (1 / df["RLLD"])) / ((1 / r_mf) - (1 / r_w))
        df["FVA"] = (0.064 / omega) * ((1 - s_wi) * df["FVDC"]) ** b
        df["FG"] = c1 * df["FVPA"] + c2 * df["FVA"] + c3 * df["FVDC"]
        df["Kf"] = 1.5 * (10**7) * omega * ((1 - s_wi) * df["FVDC"]) ** (2.63)

        df_section = df[(df["Depth"] >= start_depth) & (df["Depth"] < end_depth)]
        if df_section.empty:
            QMessageBox.warning(self, "警告", "所选深度范围内无数据。")
            return None

        _, axes = plt.subplots(nrows=1, ncols=12, figsize=(31, 10), sharey=True)
        for ax in axes:
            ax.set_ylim(bottom=max(df_section["Depth"]), top=min(df_section["Depth"]))

        # 1. axes[0]
        axes[0].set_ylabel("Depth (m)")
        axes[0].plot(
            df_section["FVDC"], df_section["Depth"], color="cyan", label="FVDC"
        )
        axes[0].fill_betweenx(
            df_section["Depth"], 0, df_section["FVDC"], color="cyan", alpha=0.5
        )
        axes[0].grid(linestyle="--", alpha=0.5)
        axes[0].set_xlabel("padding")
        axes[0].set_title("padding")
        axes[0].legend()

        axes[1].plot(df_section["AC"], df_section["Depth"], color="blue")
        axes[1].grid(linestyle="--", alpha=0.5)
        axes[1].set_xlabel("AC")
        axes[1].set_title("AC")

        axes[2].plot(df_section["DEN"], df_section["Depth"], color="green")
        axes[2].grid(linestyle="--", alpha=0.5)
        axes[2].set_xlabel("DEN")
        axes[2].set_title("DEN")

        axes[3].plot(df_section["GR"], df_section["Depth"], color="red")
        axes[3].grid(linestyle="--", alpha=0.5)
        axes[3].set_xlabel("GR")
        axes[3].set_title("GR")

        axes[4].plot(df_section["CNL"], df_section["Depth"], color="yellow")
        axes[4].grid(linestyle="--", alpha=0.5)
        axes[4].set_xlabel("CNL")
        axes[4].set_title("CNL")

        axes[5].plot(
            df_section["RLLD"], df_section["Depth"], color="purple", label="RLLD"
        )
        axes[5].plot(
            df_section["RLLS"], df_section["Depth"], color="orange", label="RLLS"
        )
        axes[5].grid(linestyle="--", alpha=0.5)
        axes[5].set_xlabel("RLLD - RLLS")
        axes[5].set_title("RLLD & RLLS")
        axes[5].legend()

        axes[6].plot(
            df_section["FVPA"], df_section["Depth"], color="brown", label="FVPA"
        )
        axes[6].fill_betweenx(
            df_section["Depth"], df_section["FVPA"], color="brown", alpha=0.5
        )
        axes[6].plot(df_section["FVA"], df_section["Depth"], color="black", label="FVA")
        axes[6].fill_betweenx(
            df_section["Depth"], df_section["FVA"], color="black", alpha=0.5
        )
        axes[6].grid(linestyle="--", alpha=0.5)
        axes[6].set_xlabel("FVA - FVPA")
        axes[6].set_title("FVA & FVPA")
        axes[6].legend()

        axes[7].plot(
            df_section["FVDC"], df_section["Depth"], color="cyan", label="FVDC"
        )
        axes[7].fill_betweenx(
            df_section["Depth"], 0, df_section["FVDC"], color="cyan", alpha=0.5
        )
        axes[7].grid(linestyle="--", alpha=0.5)
        axes[7].set_xlabel("FVDC")
        axes[7].set_title("FVDC")
        axes[7].legend()

        axes[8].plot(df_section["Kf"], df_section["Depth"], color="magenta", label="Kf")
        axes[8].fill_betweenx(
            df_section["Depth"], df_section["Kf"], color="magenta", alpha=0.5
        )
        axes[8].grid(linestyle="--", alpha=0.5)
        axes[8].set_xlabel("Kf")
        axes[8].set_title("Kf")
        axes[8].legend()

        axes[9].plot(
            df_section["FVDC"], df_section["Depth"], color="cyan", label="FVDC"
        )
        axes[9].fill_betweenx(
            df_section["Depth"], 0, df_section["FVDC"], color="cyan", alpha=0.5
        )
        axes[9].grid(linestyle="--", alpha=0.5)
        axes[9].set_xlabel("padding")
        axes[9].set_title("padding")
        axes[9].legend()

        axes[10].plot(
            df_section["FVDC"], df_section["Depth"], color="cyan", label="FVDC"
        )
        axes[10].fill_betweenx(
            df_section["Depth"], 0, df_section["FVDC"], color="cyan", alpha=0.5
        )
        axes[10].grid(linestyle="--", alpha=0.5)
        axes[10].set_xlabel("padding")
        axes[10].set_title("padding")
        axes[10].legend()

        # 3. 插入 FMI 图片
        fmi_image_path = resource_path("data/example.jpg")  # 确保图片路径正确
        if os.path.exists(fmi_image_path):
            fmi_img = plt.imread(fmi_image_path)
            axes[11].imshow(
                fmi_img,
                aspect="auto",
                extent=[0, 1, max(df_section["Depth"]), min(df_section["Depth"])],
            )
        axes[11].set_title("FMI")
        axes[11].axis("off")  # 不显示坐标轴

        plt.suptitle(f"Depth Range: {start_depth}-{end_depth} m", fontsize=14)
        plt.tight_layout()

        output_dir = resource_path("img/comprehensive_outcome/")
        os.makedirs(output_dir, exist_ok=True)
        image_path = os.path.join(output_dir, "comprehensive_outcome.png")
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
