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
        self.use_existing_columns = False
        self.initUI()

    def initUI(self):
        self.layout = QVBoxLayout()

        upload_btn_layout = QHBoxLayout()
        self.upload_btn = QPushButton("上传 Excel 文件")
        self.upload_btn.clicked.connect(self.upload_file)
        upload_btn_layout.addWidget(self.upload_btn)

        self.upload_btn_en = QPushButton("Upload")
        self.upload_btn_en.clicked.connect(
            lambda: self.upload_file(use_existing_columns=True)
        )
        upload_btn_layout.addWidget(self.upload_btn_en)

        self.layout.addLayout(upload_btn_layout)

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

        self.download_btn = QPushButton("下载 PDF")
        self.download_btn.clicked.connect(self.download_pdf)
        self.layout.addWidget(self.download_btn)

        self.setLayout(self.layout)
        self.setMinimumSize(600, 600)  # Set minimum window size

    def upload_file(self, use_existing_columns=False):
        self.use_existing_columns = use_existing_columns
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择 Excel 文件", "", "Excel 文件 (*.xlsx *.xls)"
        )
        if file_path:
            self.current_file_path = file_path
            df = pd.read_excel(file_path)
            GlobalData.df = df.copy()
            GlobalData.filtered_df = df.copy()
            status_text = (
                "上传文件成功！" if not use_existing_columns else "Upload file success!"
            )
            self.status_label.setText(status_text)
            if use_existing_columns:
                QMessageBox.information(
                    self,
                    "提示",
                    "请确保上传的文件包含以下列：\nFVA, FVPA, FVDC, Kf, Depth",
                )

    def download_pdf(self):
        if not hasattr(self, "image_label") or self.image_label.pixmap() is None:
            QMessageBox.warning(
                self, "警告", "没有可下载的图表，请先运行裂缝通道参数计算。"
            )
            return

        pdf_path, _ = QFileDialog.getSaveFileName(
            self, "保存 PDF 文件", "parameter_analysis.pdf", "PDF 文件 (*.pdf)"
        )
        if pdf_path:
            try:
                # 使用已生成的图表数据重新创建一个PDF版本
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

                # 获取当前数据范围
                df_section = GlobalData.filtered_df
                start_depth = float(self.start_depth_input.text())
                end_depth = float(self.end_depth_input.text())

                df_section = df_section[
                    (df_section["Depth"] >= start_depth)
                    & (df_section["Depth"] < end_depth)
                ]

                # 安全的累计频率计算函数
                def safe_cumulative_freq(data, bins, ax):
                    try:
                        # 移除无效值
                        valid_data = data.replace([np.inf, -np.inf], np.nan).dropna()
                        if len(valid_data) == 0:
                            return [], [], []

                        n, bins_out, patches = ax.hist(
                            valid_data, bins=bins, color="blue", alpha=0.7
                        )
                        ax.clear()  # 清除测试直方图
                        cumsum = np.cumsum(n)
                        if len(cumsum) > 0 and cumsum[-1] > 0:
                            normalized = cumsum / cumsum[-1]
                        else:
                            normalized = cumsum
                        return n, bins_out, normalized
                    except Exception:
                        # 如果出现任何错误，返回空数组
                        return [], [], []

                # Plot FVPA distribution
                try:
                    fvpa_max = (
                        df_section["FVPA"].replace([np.inf, -np.inf], np.nan).max()
                    )
                    fvpa_bins = np.linspace(
                        0, fvpa_max if not np.isnan(fvpa_max) else 1, 6
                    )
                    n1, bins1, norm1 = safe_cumulative_freq(
                        df_section["FVPA"], fvpa_bins, ax1
                    )

                    ax1.hist(
                        df_section["FVPA"],
                        bins=fvpa_bins,
                        color="blue",
                        alpha=0.7,
                        label="频率",
                    )
                    ax1_cum = ax1.twinx()
                    if len(norm1) > 0:
                        line1 = ax1_cum.plot(
                            bins1[:-1],
                            norm1,
                            color="red",
                            linewidth=2,
                            label="累计频率",
                        )[0]
                        ax1_cum.set_ylim(0, 1)

                        # 添加图例
                        lines1, labels1 = ax1.get_legend_handles_labels()
                        lines2, labels2 = ax1_cum.get_legend_handles_labels()
                        ax1.legend(
                            lines1 + [line1], labels1 + labels2, loc="upper right"
                        )
                except Exception:
                    pass

                ax1.set_title("裂缝宽度分布特征")
                ax1.set_xlabel("裂缝宽度 (mm)")
                ax1.set_ylabel("频率")
                ax1.grid(True, linestyle="--", alpha=0.5)

                # Plot FVA distribution
                try:
                    fva_max = df_section["FVA"].replace([np.inf, -np.inf], np.nan).max()
                    fva_bins = np.linspace(
                        0, fva_max if not np.isnan(fva_max) else 1, 6
                    )
                    n2, bins2, norm2 = safe_cumulative_freq(
                        df_section["FVA"], fva_bins, ax2
                    )

                    ax2.hist(
                        df_section["FVA"],
                        bins=fva_bins,
                        color="orange",
                        alpha=0.7,
                        label="频率",
                    )
                    ax2_cum = ax2.twinx()
                    if len(norm2) > 0:
                        line2 = ax2_cum.plot(
                            bins2[:-1],
                            norm2,
                            color="red",
                            linewidth=2,
                            label="累计频率",
                        )[0]
                        ax2_cum.set_ylim(0, 1)

                        # 添加图例
                        lines1, labels1 = ax2.get_legend_handles_labels()
                        lines2, labels2 = ax2_cum.get_legend_handles_labels()
                        ax2.legend(
                            lines1 + [line2], labels1 + labels2, loc="upper right"
                        )
                except Exception:
                    pass

                ax2.set_title("裂缝孔隙度分布特征")
                ax2.set_xlabel("裂缝孔隙度 (%)")
                ax2.set_ylabel("频率")
                ax2.grid(True, linestyle="--", alpha=0.5)

                # Plot FVDC distribution
                try:
                    fvdc_max = (
                        df_section["FVDC"].replace([np.inf, -np.inf], np.nan).max()
                    )
                    fvdc_bins = np.linspace(
                        0, fvdc_max if not np.isnan(fvdc_max) else 1, 6
                    )
                    n3, bins3, norm3 = safe_cumulative_freq(
                        df_section["FVDC"], fvdc_bins, ax3
                    )

                    ax3.hist(
                        df_section["FVDC"],
                        bins=fvdc_bins,
                        color="green",
                        alpha=0.7,
                        label="频率",
                    )
                    ax3_cum = ax3.twinx()
                    if len(norm3) > 0:
                        line3 = ax3_cum.plot(
                            bins3[:-1],
                            norm3,
                            color="red",
                            linewidth=2,
                            label="累计频率",
                        )[0]
                        ax3_cum.set_ylim(0, 1)

                        # 添加图例
                        lines1, labels1 = ax3.get_legend_handles_labels()
                        lines2, labels2 = ax3_cum.get_legend_handles_labels()
                        ax3.legend(
                            lines1 + [line3], labels1 + labels2, loc="upper right"
                        )
                except Exception:
                    pass

                ax3.set_title("裂缝密度分布特征")
                ax3.set_xlabel("裂缝密度")
                ax3.set_ylabel("频率")
                ax3.grid(True, linestyle="--", alpha=0.5)

                # Plot Kf distribution
                try:
                    kf_max = df_section["Kf"].replace([np.inf, -np.inf], np.nan).max()
                    kf_bins = np.linspace(0, kf_max if not np.isnan(kf_max) else 1, 6)
                    n4, bins4, norm4 = safe_cumulative_freq(
                        df_section["Kf"], kf_bins, ax4
                    )

                    ax4.hist(
                        df_section["Kf"],
                        bins=kf_bins,
                        color="red",
                        alpha=0.7,
                        label="频率",
                    )
                    ax4_cum = ax4.twinx()
                    if len(norm4) > 0:
                        line4 = ax4_cum.plot(
                            bins4[:-1],
                            norm4,
                            color="blue",
                            linewidth=2,
                            label="累计频率",
                        )[0]
                        ax4_cum.set_ylim(0, 1)

                        # 添加图例
                        lines1, labels1 = ax4.get_legend_handles_labels()
                        lines2, labels2 = ax4_cum.get_legend_handles_labels()
                        ax4.legend(
                            lines1 + [line4], labels1 + labels2, loc="upper right"
                        )
                except Exception:
                    pass

                ax4.set_title("裂缝渗透率分布特征")
                ax4.set_xlabel("裂缝渗透率")
                ax4.set_ylabel("频率")
                ax4.grid(True, linestyle="--", alpha=0.5)

                plt.tight_layout()

                # 直接保存为PDF而不是PNG
                plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
                plt.close()

                QMessageBox.information(
                    self, "成功", f"PDF文件已成功保存至：\n{pdf_path}"
                )
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存PDF文件时出错：\n{str(e)}")

    def run_parameter_analysis(self):
        QApplication.processEvents()

        # 获取参数
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

        # 使用当前文件路径和use_existing_columns标志
        file_path = self.current_file_path
        use_existing_columns = self.use_existing_columns

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
            use_existing_columns=use_existing_columns,
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
        use_existing_columns=False,
    ):
        self.status_label.setText("状态: 正在处理...")
        if file_path:
            df = pd.read_excel(file_path)
        else:
            df = GlobalData.filtered_df

        if df is None:
            self.status_label.setText("❌ 数据为空，请上传或加载数据。")
            return None

        # Modified column check logic
        if use_existing_columns:
            # 检查必要的列
            required_columns = ["Depth"]
            # 尝试查找匹配的列名（不区分大小写）
            column_mapping = {}
            for col in ["FVA", "FVPA", "FVDC", "Kf"]:
                found = False
                for df_col in df.columns:
                    if df_col.upper() == col.upper():
                        column_mapping[col] = df_col
                        found = True
                        break
                if not found:
                    QMessageBox.critical(self, "错误", f"未找到必要的列: {col}")
                    return None

            # 使用映射的列名
            df = df.rename(columns=column_mapping)
        else:
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

        # Modified parameter calculation logic
        if not use_existing_columns:
            df_section["FVPA"] = (
                r_mf * ((1 / df_section["RLLS"]) - (1 / df_section["RLLD"]))
            ) ** a
            df_section["FVDC"] = (
                (1 / df_section["RLLS"]) - (1 / df_section["RLLD"])
            ) / ((1 / r_mf) - (1 / r_w))
            df_section["FVA"] = (0.064 / omega) * ((1 - s_wi) * df_section["FVDC"]) ** b
            df_section["Kf"] = (
                1.5 * (10**7) * omega * ((1 - s_wi) * df_section["FVDC"]) ** (2.63)
            )
        else:
            # Use existing columns directly
            df_section["FVPA"] = df_section["FVPA"]
            df_section["FVDC"] = df_section["FVDC"]
            df_section["FVA"] = df_section["FVA"]
            df_section["Kf"] = df_section["Kf"]

        # Create frequency distribution plots
        _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

        # Plot FVPA distribution
        fvpa_bins = np.linspace(0, df_section["FVPA"].max(), 6)
        n1, bins1, hist1 = ax1.hist(
            df_section["FVPA"],
            bins=fvpa_bins,
            color="blue",
            alpha=0.7,
            label="频率",  # 添加标签
        )
        ax1.set_title("裂缝宽度分布特征")
        ax1.set_xlabel("裂缝宽度 (mm)")
        ax1.set_ylabel("频率")
        ax1.grid(True, linestyle="--", alpha=0.5)

        # Add cumulative frequency line for FVPA
        ax1_cum = ax1.twinx()
        cumsum1 = np.cumsum(n1)
        cumsum1_normalized = cumsum1 / cumsum1[-1]
        line1 = ax1_cum.plot(
            bins1[:-1], cumsum1_normalized, color="red", linewidth=2, label="累计频率"
        )[0]  # 添加标签并获取线对象
        ax1_cum.set_ylim(0, 1)

        # 合并两个图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_cum.get_legend_handles_labels()
        ax1.legend(lines1 + [line1], labels1 + labels2, loc="upper right")

        # Plot FVA distribution
        fva_bins = np.linspace(0, df_section["FVA"].max(), 6)
        n2, bins2, hist2 = ax2.hist(
            df_section["FVA"], bins=fva_bins, color="orange", alpha=0.7, label="频率"
        )
        ax2.set_title("裂缝孔隙度分布特征")
        ax2.set_xlabel("裂缝孔隙度 (%)")
        ax2.set_ylabel("频率")
        ax2.grid(True, linestyle="--", alpha=0.5)

        # Add cumulative frequency line for FVA
        ax2_cum = ax2.twinx()
        cumsum2 = np.cumsum(n2)
        cumsum2_normalized = cumsum2 / cumsum2[-1]
        line2 = ax2_cum.plot(
            bins2[:-1], cumsum2_normalized, color="red", linewidth=2, label="累计频率"
        )[0]
        ax2_cum.set_ylabel("累计频率")
        ax2_cum.set_ylim(0, 1)

        # 合并两个图例
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_cum.get_legend_handles_labels()
        ax2.legend(lines1 + [line2], labels1 + labels2, loc="upper right")

        # Plot FVDC distribution
        fvdc_bins = np.linspace(0, df_section["FVDC"].max(), 6)
        n3, bins3, hist3 = ax3.hist(
            df_section["FVDC"], bins=fvdc_bins, color="green", alpha=0.7, label="频率"
        )
        ax3.set_title("裂缝密度分布特征")
        ax3.set_xlabel("裂缝密度")
        ax3.set_ylabel("频率")
        ax3.grid(True, linestyle="--", alpha=0.5)

        # Add cumulative frequency line for FVDC
        ax3_cum = ax3.twinx()
        cumsum3 = np.cumsum(n3)
        cumsum3_normalized = cumsum3 / cumsum3[-1]
        line3 = ax3_cum.plot(
            bins3[:-1], cumsum3_normalized, color="red", linewidth=2, label="累计频率"
        )[0]
        ax3_cum.set_ylabel("累计频率")
        ax3_cum.set_ylim(0, 1)

        # 合并两个图例
        lines1, labels1 = ax3.get_legend_handles_labels()
        lines2, labels2 = ax3_cum.get_legend_handles_labels()
        ax3.legend(lines1 + [line3], labels1 + labels2, loc="upper right")

        # Plot Kf distribution
        kf_bins = np.linspace(0, df_section["Kf"].max(), 6)
        n4, bins4, hist4 = ax4.hist(
            df_section["Kf"], bins=kf_bins, color="red", alpha=0.7, label="频率"
        )
        ax4.set_title("裂缝渗透率分布特征")
        ax4.set_xlabel("裂缝渗透率")
        ax4.set_ylabel("频率")
        ax4.grid(True, linestyle="--", alpha=0.5)

        # Add cumulative frequency line for Kf
        ax4_cum = ax4.twinx()
        cumsum4 = np.cumsum(n4)
        cumsum4_normalized = cumsum4 / cumsum4[-1]
        line4 = ax4_cum.plot(
            bins4[:-1], cumsum4_normalized, color="blue", linewidth=2, label="累计频率"
        )[0]
        ax4_cum.set_ylabel("累计频率")
        ax4_cum.set_ylim(0, 1)

        # 合并两个图例
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_cum.get_legend_handles_labels()
        ax4.legend(lines1 + [line4], labels1 + labels2, loc="upper right")

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
