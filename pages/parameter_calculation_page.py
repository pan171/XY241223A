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

from pages.config import GlobalData, resource_path


class ParameterCalculationPage(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.upload_mode = "standard"  # Flag to track which upload button was used

    def initUI(self):
        self.layout = QVBoxLayout()

        # Create horizontal layout for upload buttons
        upload_buttons_layout = QHBoxLayout()

        self.upload_btn = QPushButton("上传 Excel 文件")
        self.upload_btn.clicked.connect(lambda: self.upload_file("standard"))
        upload_buttons_layout.addWidget(self.upload_btn)

        # Add new Upload button
        self.new_upload_btn = QPushButton("Upload")
        self.new_upload_btn.clicked.connect(lambda: self.upload_file("advanced"))
        upload_buttons_layout.addWidget(self.new_upload_btn)

        self.fmi_upload_btn = QPushButton("上传 FMI")
        self.fmi_upload_btn.clicked.connect(self.upload_fmi_image)
        upload_buttons_layout.addWidget(self.fmi_upload_btn)

        self.layout.addLayout(upload_buttons_layout)

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

        self.identify_btn = QPushButton("裂缝通道参数计算")
        self.identify_btn.clicked.connect(self.run_parameter_calculation)
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

        self.image_label = QLabel("裂缝通道参数计算结果")
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

    def upload_file(self, mode="standard"):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择 Excel 文件", "", "Excel 文件 (*.xlsx *.xls)"
        )
        if file_path:
            self.upload_mode = mode
            self.current_file_path = file_path
            df = pd.read_excel(file_path)
            GlobalData.df = df.copy()
            GlobalData.filtered_df = df.copy()
            self.status_label.setText("上传文件成功！")

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
                # Get the source image path
                source_image_path = resource_path(
                    "img/parameter_calculation/parameter_calculation.png"
                )

                # Create a new figure with the same size as the original plot
                plt.figure(figsize=(32, 10))

                # Read and display the image
                img = plt.imread(source_image_path)
                plt.imshow(img)
                plt.axis("off")  # Hide axes

                # Save as PDF
                plt.savefig(pdf_path, format="pdf", dpi=300, bbox_inches="tight")
                plt.close()

                QMessageBox.information(
                    self, "成功", f"图片已成功保存为PDF：\n{pdf_path}"
                )
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存PDF文件时出错：{str(e)}")

    def run_parameter_calculation(self):
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

        # Check if using advanced mode (Upload button)
        if self.upload_mode == "advanced":
            # Check for required columns in advanced mode
            required_columns = [
                "岩性道",
                "AC",
                "DEN",
                "GR",
                "CNL",
                "RLLD",
                "RLLS",
                "FVA",
                "FVPA",
                "FVDC",
                "Kf",
                "解释结论",
            ]
            if not all(col in df.columns for col in required_columns):
                QMessageBox.critical(
                    self, "错误", f"该模式输入文件必须包含列: {required_columns}"
                )
                return None

            # Plot with advanced visualization
            return self.plot_advanced_parameter_log(df, start_depth, end_depth)
        else:
            # Standard plotting mode
            required_columns = ["Depth", "AC", "DEN", "GR", "CNL", "RLLD", "RLLS"]
            if not all(col in df.columns for col in required_columns):
                QMessageBox.critical(
                    self, "错误", f"输入文件必须包含列: {required_columns}"
                )
                return None

            # Add default 岩性道 column if not present
            if "岩性道" not in df.columns:
                df["岩性道"] = 1  # Default to medium grain size
                df["解释结论"] = 0  # Default interpretation result

            # Calculate additional parameters
            df["FVPA"] = (r_mf * ((1 / df["RLLS"]) - (1 / df["RLLD"]))) ** a
            df["FVDC"] = ((1 / df["RLLS"]) - (1 / df["RLLD"])) / (
                (1 / r_mf) - (1 / r_w)
            )
            df["FVA"] = (0.064 / omega) * ((1 - s_wi) * df["FVDC"]) ** b
            df["FG"] = c1 * df["FVPA"] + c2 * df["FVA"] + c3 * df["FVDC"]
            df["Kf"] = 1.5 * (10**7) * omega * ((1 - s_wi) * df["FVDC"]) ** (2.63)

            # Filter by depth range
            df_section = df[(df["Depth"] >= start_depth) & (df["Depth"] < end_depth)]
            if df_section.empty:
                QMessageBox.warning(self, "警告", "所选深度范围内无数据。")
                return None

            # Create figure with GridSpec to control subplot widths
            _ = plt.figure(figsize=(32, 10))
            gs = plt.GridSpec(1, 11, width_ratios=[0.25] + [1] * 10)
            axes = [plt.subplot(gs[0, i]) for i in range(11)]

            for ax in axes:
                ax.set_ylim(
                    bottom=max(df_section["Depth"]), top=min(df_section["Depth"])
                )
                # Hide ylabel and yticks for all axes except the first one
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
                    axes[0].fill_betweenx(
                        [start_depth, end_depth],
                        0,
                        1,
                        color=pattern["color"],
                        hatch=pattern["hatch"],
                        alpha=0.7,
                        linewidth=0,
                    )
                    # Add black line between layers
                    axes[0].axhline(y=end_depth, color="black", linewidth=0.5)

                    # Add to legend (only once per lithology type)
                    if pattern["label"] not in [e.get_label() for e in legend_elements]:
                        from matplotlib.patches import Patch

                        legend_elements.append(
                            Patch(
                                facecolor=pattern["color"],
                                hatch=pattern["hatch"],
                                label=pattern["label"],
                                alpha=0.7,
                            )
                        )

            # Add legend
            axes[0].legend(
                handles=legend_elements, loc="lower center", fontsize="small"
            )
            axes[0].set_xlim(0, 1)  # Set x-axis limits
            axes[0].set_xticks([])  # Hide x-axis ticks

            # Original subplots shifted one position to the right
            # fig1: AC —— blue
            axes[1].plot(df_section["AC"], df_section["Depth"], color="blue")
            axes[1].grid(linestyle="--", alpha=0.5)
            axes[1].set_xlabel("AC")
            axes[1].set_title("AC")

            # fig2: DEN —— green
            axes[2].plot(df_section["DEN"], df_section["Depth"], color="green")
            axes[2].grid(linestyle="--", alpha=0.5)
            axes[2].set_xlabel("DEN")
            axes[2].set_title("DEN")

            # fig3: GR —— red
            axes[3].plot(df_section["GR"], df_section["Depth"], color="red")
            axes[3].grid(linestyle="--", alpha=0.5)
            axes[3].set_xlabel("GR")
            axes[3].set_title("GR")

            # fig4: CNL —— yellow
            axes[4].plot(df_section["CNL"], df_section["Depth"], color="yellow")
            axes[4].grid(linestyle="--", alpha=0.5)
            axes[4].set_xlabel("CNL")
            axes[4].set_title("CNL")

            # fig5: RLLD & RLLS —— purple & orange
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

            # Rest of the subplots shifted one position
            # Update indices for remaining plots (6 through 10)
            axes[6].plot(
                df_section["FVPA"], df_section["Depth"], color="brown", label="FVPA"
            )
            axes[6].fill_betweenx(
                df_section["Depth"], df_section["FVPA"], color="brown", alpha=0.5
            )
            # Add FVA plot (only line, no fill)
            axes[6].plot(
                df_section["FVA"], df_section["Depth"], color="black", label="FVA"
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

            axes[8].plot(
                df_section["Kf"], df_section["Depth"], color="magenta", label="Kf"
            )
            axes[8].fill_betweenx(
                df_section["Depth"], df_section["Kf"], color="magenta", alpha=0.5
            )
            axes[8].grid(linestyle="--", alpha=0.5)
            axes[8].set_xlabel("Kf")
            axes[8].set_title("Kf")
            axes[8].legend()

            axes[9].set_xlabel("解释结论")
            axes[9].set_title("解释结论")
            axes[9].grid(linestyle="--", alpha=0.5)

            # Define color mapping
            colors = {0: "white", 1: "green", 2: "blue", 3: "yellow"}

            # Get depth and interpretation result values
            depths = df_section["Depth"].values
            results = df_section["解释结论"].values

            # Plot lines for each interpretation result
            for i in range(len(depths)):
                result = results[i]
                if pd.notna(result):
                    result = int(result)  # Ensure it's an integer
                    color = colors[result]  # 直接使用映射的颜色
                    # Draw horizontal line at each depth
                    axes[9].hlines(
                        y=depths[i], xmin=0, xmax=1, color=color, linewidth=2
                    )

            # Set appropriate x-axis limits
            axes[9].set_xlim(0, 1)

            # FMI image
            fmi_image_path = resource_path("data/example.jpg")
            if os.path.exists(fmi_image_path):
                fmi_img = plt.imread(fmi_image_path)
                axes[10].imshow(
                    fmi_img,
                    aspect="auto",
                    extent=[0, 1, max(df_section["Depth"]), min(df_section["Depth"])],
                )
            axes[10].set_title("FMI")
            axes[10].axis("off")

            plt.suptitle(f"Depth Range: {start_depth}-{end_depth} m", fontsize=14)
            plt.tight_layout()

            output_dir = resource_path("img/parameter_calculation/")
            os.makedirs(output_dir, exist_ok=True)
            image_path = os.path.join(output_dir, "parameter_calculation.png")
            plt.savefig(image_path, dpi=300, bbox_inches="tight")
            plt.close()

            return image_path

    def plot_advanced_parameter_log(self, df, start_depth, end_depth):
        """Plot parameter log with advanced visualization for "岩性道" and "解释结论" columns"""
        # Filter by depth range
        depth_col = (
            "Depth" if "Depth" in df.columns else df.columns[0]
        )  # Assume first column is depth if not named "Depth"
        df_section = df[(df[depth_col] >= start_depth) & (df[depth_col] < end_depth)]

        if df_section.empty:
            QMessageBox.warning(self, "警告", "所选深度范围内无数据。")
            return None

        # Create figure with GridSpec to control subplot widths
        _ = plt.figure(figsize=(32, 10))
        gs = plt.GridSpec(1, 11, width_ratios=[0.25] + [1] * 10)
        axes = [plt.subplot(gs[0, i]) for i in range(11)]

        for ax in axes:
            ax.set_ylim(
                bottom=max(df_section[depth_col]), top=min(df_section[depth_col])
            )
            # Hide ylabel and yticks for all axes except the first one
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

        # Group by consecutive lithology values
        df_section = df_section.copy()
        df_section["岩性道"] = df_section["岩性道"].astype(float)

        # Create a list to store layer information
        lithology_layers = []
        current_litho = None
        layer_start = None

        # Identify continuous layers
        for _, row in df_section.sort_values(depth_col).iterrows():
            depth = row[depth_col]
            litho = row["岩性道"] if pd.notna(row["岩性道"]) else None

            if litho != current_litho:
                if current_litho is not None and layer_start is not None:
                    lithology_layers.append((layer_start, depth, current_litho))
                current_litho = litho
                layer_start = depth

        # Add the last layer
        if current_litho is not None and layer_start is not None:
            lithology_layers.append(
                (layer_start, max(df_section[depth_col]), current_litho)
            )

        # Draw the layers
        legend_elements = []
        for start_depth, end_depth, litho in lithology_layers:
            if litho in lithology_patterns:
                pattern = lithology_patterns[litho]
                axes[0].fill_betweenx(
                    [start_depth, end_depth],
                    0,
                    1,
                    color=pattern["color"],
                    hatch=pattern["hatch"],
                    alpha=0.7,
                    linewidth=0,
                )
                # Add black line between layers
                axes[0].axhline(y=end_depth, color="black", linewidth=0.5)

                # Add to legend (only once per lithology type)
                if pattern["label"] not in [e.get_label() for e in legend_elements]:
                    from matplotlib.patches import Patch

                    legend_elements.append(
                        Patch(
                            facecolor=pattern["color"],
                            hatch=pattern["hatch"],
                            label=pattern["label"],
                            alpha=0.7,
                        )
                    )

        # Add legend
        axes[0].legend(handles=legend_elements, loc="lower center", fontsize="small")
        axes[0].set_xlim(0, 1)  # Set x-axis limits
        axes[0].set_xticks([])  # Hide x-axis ticks

        # Standard curve plots (AC, DEN, GR, CNL)
        axes[1].plot(df_section["AC"], df_section[depth_col], color="blue")
        axes[1].grid(linestyle="--", alpha=0.5)
        axes[1].set_xlabel("AC")
        axes[1].set_title("AC")

        axes[2].plot(df_section["DEN"], df_section[depth_col], color="green")
        axes[2].grid(linestyle="--", alpha=0.5)
        axes[2].set_xlabel("DEN")
        axes[2].set_title("DEN")

        axes[3].plot(df_section["GR"], df_section[depth_col], color="red")
        axes[3].grid(linestyle="--", alpha=0.5)
        axes[3].set_xlabel("GR")
        axes[3].set_title("GR")

        axes[4].plot(df_section["CNL"], df_section[depth_col], color="yellow")
        axes[4].grid(linestyle="--", alpha=0.5)
        axes[4].set_xlabel("CNL")
        axes[4].set_title("CNL")

        # RLLD & RLLS
        axes[5].plot(
            df_section["RLLD"], df_section[depth_col], color="purple", label="RLLD"
        )
        axes[5].plot(
            df_section["RLLS"], df_section[depth_col], color="orange", label="RLLS"
        )
        axes[5].grid(linestyle="--", alpha=0.5)
        axes[5].set_xlabel("RLLD - RLLS")
        axes[5].set_title("RLLD & RLLS")
        axes[5].legend()

        # FVA & FVPA
        axes[6].plot(
            df_section["FVPA"], df_section[depth_col], color="brown", label="FVPA"
        )
        axes[6].fill_betweenx(
            df_section[depth_col], df_section["FVPA"], color="brown", alpha=0.5
        )
        # Add FVA plot (only line, no fill)
        axes[6].plot(
            df_section["FVA"], df_section[depth_col], color="black", label="FVA"
        )
        axes[6].grid(linestyle="--", alpha=0.5)
        axes[6].set_xlabel("FVA - FVPA")
        axes[6].set_title("FVA & FVPA")
        axes[6].legend()

        # FVDC
        axes[7].plot(
            df_section["FVDC"], df_section[depth_col], color="cyan", label="FVDC"
        )
        axes[7].fill_betweenx(
            df_section[depth_col], 0, df_section["FVDC"], color="cyan", alpha=0.5
        )
        axes[7].grid(linestyle="--", alpha=0.5)
        axes[7].set_xlabel("FVDC")
        axes[7].set_title("FVDC")
        axes[7].legend()

        # Kf
        axes[8].plot(
            df_section["Kf"], df_section[depth_col], color="magenta", label="Kf"
        )
        axes[8].fill_betweenx(
            df_section[depth_col], df_section["Kf"], color="magenta", alpha=0.5
        )
        axes[8].grid(linestyle="--", alpha=0.5)
        axes[8].set_xlabel("Kf")
        axes[8].set_title("Kf")
        axes[8].legend()

        # 使用线段
        axes[9].set_xlabel("解释结论")
        axes[9].set_title("解释结论")
        axes[9].grid(linestyle="--", alpha=0.5)

        # Define color mapping
        colors = {0: "white", 1: "green", 2: "blue", 3: "yellow"}

        # Get depth and interpretation result values
        depths = df_section[depth_col].values
        results = df_section["解释结论"].values

        # Plot lines for each interpretation result
        for i in range(len(depths)):
            result = results[i]
            if pd.notna(result):
                result = int(result)  # Ensure it's an integer
                color = colors[result]  # 直接使用映射的颜色
                # Draw horizontal line at each depth
                axes[9].hlines(y=depths[i], xmin=0, xmax=1, color=color, linewidth=2)

        # Set appropriate x-axis limits
        axes[9].set_xlim(0, 1)

        # FMI image
        fmi_image_path = resource_path("data/example.jpg")
        if os.path.exists(fmi_image_path):
            fmi_img = plt.imread(fmi_image_path)
            axes[10].imshow(
                fmi_img,
                aspect="auto",
                extent=[0, 1, max(df_section[depth_col]), min(df_section[depth_col])],
            )
        axes[10].set_title("FMI")
        axes[10].axis("off")

        plt.suptitle(f"Depth Range: {start_depth}-{end_depth} m", fontsize=14)
        plt.tight_layout()

        output_dir = resource_path("img/parameter_calculation/")
        os.makedirs(output_dir, exist_ok=True)
        image_path = os.path.join(output_dir, "parameter_calculation.png")
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
