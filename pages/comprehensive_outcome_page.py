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
import shutil

from pages.config import GlobalData, resource_path


class ComprehensiveOutcomePage(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.layout = QVBoxLayout()

        # Create horizontal layout for upload buttons
        upload_buttons_layout = QHBoxLayout()

        self.upload_btn = QPushButton("上传 Excel 文件")
        self.upload_btn.clicked.connect(self.upload_file)
        upload_buttons_layout.addWidget(self.upload_btn)

        self.fmi_upload_btn = QPushButton("上传 FMI")
        self.fmi_upload_btn.clicked.connect(self.upload_fmi_image)
        upload_buttons_layout.addWidget(self.fmi_upload_btn)

        self.layout.addLayout(upload_buttons_layout)

        ##################### Line 1 ###########################
        comb_1 = QHBoxLayout()

        # parameters: start, end
        self.start_depth_label = QLabel("start_depth: ")
        self.start_depth_input = QLineEdit("3500")
        self.start_depth_input.setPlaceholderText("起始深度")
        # self.start_depth_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        comb_1.addWidget(self.start_depth_label)
        comb_1.addWidget(self.start_depth_input)

        self.end_depth_label = QLabel("end_depth: ")
        self.end_depth_input = QLineEdit("4000")
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

        # self.fmi_upload_btn = QPushButton("上传 FMI")
        # self.fmi_upload_btn.clicked.connect(self.upload_fmi_image)
        # comb_2.addWidget(self.fmi_upload_btn)

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

        # Create a scroll area for the image
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setMinimumHeight(300)  # Set minimum height

        # Create a container widget for the image
        self.image_container = QWidget()
        self.image_layout = QVBoxLayout(self.image_container)

        self.image_label = QLabel("测录井综合图结果")
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

        pdf_path, _ = QFileDialog.getSaveFileName(
            self, "保存PDF文件", "", "PDF 文件 (*.pdf)"
        )
        if pdf_path:
            try:
                # Get the pre-generated PDF path
                source_pdf_path = resource_path(
                    "img/comprehensive_outcome/comprehensive_outcome.pdf"
                )

                # Simply copy the already-generated vector PDF
                shutil.copy2(source_pdf_path, pdf_path)

                QMessageBox.information(
                    self, "成功", f"图片已成功保存为可编辑的PDF：\n{pdf_path}"
                )
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存PDF文件时出错：{str(e)}")

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

        required_columns = [
            "Depth",
            "AC",
            "DEN",
            "GR",
            "CNL",
            "RLLD",
            "RLLS",
            "Kf",
            "FVA",
            "FVPA",
            "FVDC",
            "Qloss",
            "解释结论",
        ]
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
        # df["FVPA"] = (r_mf * ((1 / df["RLLS"]) - (1 / df["RLLD"]))) ** a
        # df["FVDC"] = ((1 / df["RLLS"]) - (1 / df["RLLD"])) / ((1 / r_mf) - (1 / r_w))
        # df["FVA"] = (0.064 / omega) * ((1 - s_wi) * df["FVDC"]) ** b
        # df["FG"] = c1 * df["FVPA"] + c2 * df["FVA"] + c3 * df["FVDC"]
        # df["Kf"] = 1.5 * (10**7) * omega * ((1 - s_wi) * df["FVDC"]) ** (2.63)

        df_section = df[(df["Depth"] >= start_depth) & (df["Depth"] < end_depth)]
        if df_section.empty:
            QMessageBox.warning(self, "警告", "所选深度范围内无数据。")
            return None

        # Create figure with GridSpec to control subplot widths
        _ = plt.figure(figsize=(32, 10))
        gs = plt.GridSpec(
            1, 12, width_ratios=[0.25] + [1] * 11
        )  # Updated to 12 subplots
        axes = [plt.subplot(gs[0, i]) for i in range(12)]

        for ax in axes:
            ax.set_ylim(bottom=max(df_section["Depth"]), top=min(df_section["Depth"]))
            if ax != axes[0]:
                ax.set_ylabel("")
                ax.set_yticks([])

        # 岩性 - Draw rock layers instead of markers
        axes[0].set_xlabel("岩性")
        axes[0].set_title("岩性")
        axes[0].set_ylabel("Depth (m)")
        axes[0].grid(False)

        # Define patterns and colors for each lithology type
        lithology_patterns = {
            0: {
                "hatch": "...",
                "color": "saddlebrown",
                "label": "Coarse",
            },  # Coarse - dark brown
            1: {
                "hatch": "++",
                "color": "peru",
                "label": "Medium",
            },  # Medium - medium brown
            2: {
                "hatch": "--",
                "color": "burlywood",
                "label": "Fine",
            },  # Fine - light brown
        }

        # Add default 岩性道 column if not present
        if "岩性道" not in df_section.columns:
            df_section["岩性道"] = 1  # Default to medium grain size

        # Group by consecutive lithology values
        df_section = df_section.copy()
        df_section["岩性道"] = df_section["岩性道"].astype(float)

        # Create a list to store layer information
        lithology_layers = []
        current_litho = None
        layer_start = None

        # Identify continuous layers
        for _, row in df_section.sort_values("Depth").iterrows():
            depth = row["Depth"]
            litho = row["岩性道"] if pd.notna(row["岩性道"]) else None

            if litho != current_litho:
                if current_litho is not None and layer_start is not None:
                    lithology_layers.append((layer_start, depth, current_litho))
                current_litho = litho
                layer_start = depth

        # Add the last layer
        if current_litho is not None and layer_start is not None:
            lithology_layers.append(
                (layer_start, max(df_section["Depth"]), current_litho)
            )

        # Draw the layers
        legend_elements = []
        for start_depth, end_depth, litho in lithology_layers:
            if litho in lithology_patterns:
                pattern = lithology_patterns[litho]
                # 增加 hatch 的密度，通过重复 pattern 字符
                dense_hatch = pattern["hatch"] * 3  # 重复3次增加密度

                # 绘制填充
                axes[0].fill_betweenx(
                    [start_depth, end_depth],
                    0,
                    1,
                    color=pattern["color"],
                    hatch=dense_hatch,
                    alpha=0.5,  # 降低填充色的透明度使 hatch 更明显
                    linewidth=0.5,  # 添加轮廓线
                    edgecolor="black",  # 添加黑色边框
                )

                # Add black line between layers
                axes[0].axhline(y=end_depth, color="black", linewidth=0.5)

                # Add to legend (only once per lithology type)
                if pattern["label"] not in [e.get_label() for e in legend_elements]:
                    from matplotlib.patches import Patch

                    legend_elements.append(
                        Patch(
                            facecolor=pattern["color"],
                            hatch=dense_hatch,
                            label=pattern["label"],
                            alpha=0.5,
                            linewidth=0.5,
                            edgecolor="black",
                        )
                    )

        # Add legend
        axes[0].legend(handles=legend_elements, loc="lower center", fontsize="small")
        axes[0].set_xlim(0, 1)  # Set x-axis limits
        axes[0].set_xticks([])  # Hide x-axis ticks

        # Plot other curves
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

        # Plot FVA and FVPA with additional scatter plots for LT, BP, and FDCNN
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

        # Add scatter plots for LT, BP, and FDCNN
        if "LT" in df_section.columns:
            non_zero_lt = df_section[df_section["LT"] != 0]
            axes[6].scatter(
                non_zero_lt["LT"],
                non_zero_lt["Depth"],
                color="red",
                marker="s",  # square marker
                s=50,  # marker size
                label="LT",
            )

        if "BP" in df_section.columns:
            non_zero_bp = df_section[df_section["BP"] != 0]
            axes[6].scatter(
                non_zero_bp["BP"],
                non_zero_bp["Depth"],
                color="black",
                marker="D",  # diamond marker
                s=50,  # marker size
                label="BP",
            )

        if "FDCNN" in df_section.columns:
            non_zero_fdcnn = df_section[df_section["FDCNN"] != 0]
            axes[6].scatter(
                non_zero_fdcnn["FDCNN"],
                non_zero_fdcnn["Depth"],
                color="blue",
                marker="^",  # triangle marker
                s=50,  # marker size
                label="FDCNN",
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

        # Qloss
        axes[9].barh(
            df_section["Depth"],
            df_section["Qloss"]
            if "Qloss" in df_section.columns
            else [0] * len(df_section),
            height=1,  # 设置柱状图高度
            color="red",
            alpha=0.6,
            label="Qloss",
        )
        axes[9].grid(linestyle="--", alpha=0.5)
        axes[9].set_xlabel("Qloss")
        axes[9].set_title("Qloss")
        axes[9].legend(loc="upper right")

        # 如果Qloss列不存在，显示提示信息
        if "Qloss" not in df_section.columns:
            axes[9].text(
                0.5,
                0.5,
                "No Qloss data",
                horizontalalignment="center",
                verticalalignment="center",
                transform=axes[9].transAxes,
            )

        # Replace the padding subplot with interpretation results
        axes[10].set_xlabel("解释结论")
        axes[10].set_title("解释结论")
        axes[10].grid(linestyle="--", alpha=0.5)

        # Define color mapping
        colors = {0: "white", 1: "green", 2: "blue", 3: "yellow"}

        # Get depth and interpretation result values
        depths = df_section["Depth"].values

        # Add default "解释结论" column if it doesn't exist
        if "解释结论" not in df_section.columns:
            df_section["解释结论"] = 0

        # Convert to numeric and handle any invalid values
        df_section["解释结论"] = pd.to_numeric(
            df_section["解释结论"], errors="coerce"
        ).fillna(0)
        # Ensure values are integers
        df_section["解释结论"] = df_section["解释结论"].astype(int)

        results = df_section["解释结论"].values

        # Plot lines for each interpretation result
        for i in range(len(depths)):
            result = int(results[i])  # Convert to int
            color = colors.get(
                result, "white"
            )  # Use get() to handle any unexpected values
            # Draw horizontal line at each depth with increased line width
            axes[10].hlines(y=depths[i], xmin=0, xmax=1, color=color, linewidth=3)

        # Set appropriate x-axis limits
        axes[10].set_xlim(0, 1)

        # FMI image moved to last subplot (axes[11])
        fmi_image_path = resource_path("data/example.jpg")
        if os.path.exists(fmi_image_path):
            fmi_img = plt.imread(fmi_image_path)
            axes[11].imshow(
                fmi_img,
                aspect="auto",
                extent=[0, 1, max(df_section["Depth"]), min(df_section["Depth"])],
            )
        axes[11].set_title("FMI")
        axes[11].axis("off")

        plt.suptitle(f"Depth Range: {start_depth}-{end_depth} m", fontsize=14)
        plt.tight_layout()

        # 在最后保存图片时同时保存PNG和PDF
        output_dir = resource_path("img/comprehensive_outcome/")
        os.makedirs(output_dir, exist_ok=True)

        # Save PNG for display
        image_path = os.path.join(output_dir, "comprehensive_outcome.png")
        plt.savefig(image_path, dpi=300, bbox_inches="tight")

        # Save vector PDF for download
        pdf_path = os.path.join(output_dir, "comprehensive_outcome.pdf")
        plt.savefig(pdf_path, format="pdf", bbox_inches="tight")

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
