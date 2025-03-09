from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QLineEdit,
    QApplication,
    QHBoxLayout,
    QMessageBox,
    QFileDialog,
    QGroupBox,
    QFrame,
)
import numpy as np
import pandas as pd
import os


def model_1(w, delta_p, tau_y, r_w, v_m):
    """定义Lietard-Griffiths模型方程"""
    return (
        (delta_p / tau_y) * w**3
        + 6 * r_w * (delta_p / tau_y) * w**2
        - (9 / np.pi) * v_m
    )


def df_model_1(w, delta_p, tau_y, r_w):
    """计算Lietard-Griffiths模型方程的导数"""
    return 3 * (delta_p / tau_y) * w**2 + 12 * r_w * (delta_p / tau_y) * w


def newton_method_Lietard(w0, delta_p, tau_y, r_w, v_m, tol=1e-6, max_iter=100):
    """使用牛顿法求解 w"""
    w = w0
    for _ in range(max_iter):
        f_w = model_1(w, delta_p, tau_y, r_w, v_m)
        df_w = df_model_1(w, delta_p, tau_y, r_w)

        if abs(df_w) < 1e-8:
            print("导数接近零，牛顿法可能失败。")
            return None

        w_new = w - f_w / df_w

        if abs(w_new - w) < tol:
            return w_new

        w = w_new

    print("牛顿法未能在最大迭代次数内收敛")
    return None


def model_2(w, delta_p, Q_loss, mu, r_w, v_m):
    """定义 Verga 模型方程"""
    term1 = (6 * Q_loss * mu) / (np.pi * w**3)
    term2 = np.log((v_m / (np.pi * w) + r_w**2) / r_w)
    return term1 * term2 - delta_p


def df_model_2(w, Q_loss, mu, r_w, v_m):
    """计算 Verga 模型方程的导数"""
    term1 = (-18 * Q_loss * mu) / (np.pi * w**4)
    term2 = np.log((v_m / (np.pi * w) + r_w**2) / r_w)

    term3_numerator = -v_m / (np.pi * w**2)
    term3_denominator = (v_m / (np.pi * w) + r_w**2) * r_w
    term3 = (6 * Q_loss * mu) / (np.pi * w**3) * (term3_numerator / term3_denominator)

    return term1 * term2 + term3


def newton_method_Verga(w0, delta_p, Q_loss, mu, r_w, v_m, tol=1e-6, max_iter=100):
    """使用牛顿法求解 w"""
    w = w0
    for _ in range(max_iter):
        f_w = model_2(w, delta_p, Q_loss, mu, r_w, v_m)
        df_w = df_model_2(w, Q_loss, mu, r_w, v_m)

        if abs(df_w) < 1e-8:
            print("导数接近零，牛顿法可能失败。")
            return None

        w_new = w - f_w / df_w

        if abs(w_new - w) < tol:
            return w_new

        w = w_new

    print("牛顿法未能在最大迭代次数内收敛")
    return None


class HydrodynamicPage(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.lg_result = None
        self.verga_result = None

    def initUI(self):
        # Set global font size and styles - reduced font size
        font_style = "font-size: 14px;"
        title_style = "font-size: 16px; font-weight: bold; color: #2c3e50;"
        label_width = 130  # Increased width for labels to fit content
        input_width = 180  # Increased width for inputs

        # Set up the main frame with a light background color
        self.setStyleSheet("background-color: #f5f5f7;")

        # Main horizontal layout
        self.layout = QHBoxLayout()
        self.layout.setSpacing(20)  # Spacing between the two models
        self.layout.setContentsMargins(5, 5, 5, 5)  # Reduced main layout margins

        # Left module: Lietard-Griffiths Model
        lg_group = QGroupBox("Lietard-Griffiths Model")
        lg_group.setStyleSheet("""
            QGroupBox {
                font-size: 16px;
                font-weight: bold;
                border: 2px solid #3498db;
                border-radius: 8px;
                margin-top: 15px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #2980b9;
            }
        """)
        lg_group.setMinimumWidth(430)
        lg_group.setMaximumWidth(500)

        self.lg_layout = QVBoxLayout(lg_group)
        self.lg_layout.setContentsMargins(15, 10, 15, 10)  # Reduced vertical margins
        self.lg_layout.setSpacing(8)  # Reduced spacing between elements

        # Default values for Lietard-Griffiths
        lg_defaults = {
            "delta_p": "1.0",
            "tau_y": "0.5",
            "r_w": "0.1",
            "v_m": "0.02",
            "w0": "1.0",
        }

        # Create input fields with labels and default values
        self.lg_inputs = {}
        lg_parameters = ["delta_p", "tau_y", "r_w", "v_m", "w0"]

        # Parameter descriptions
        lg_descriptions = {
            "delta_p": "Delta P (Pa)",
            "tau_y": "Tau Y (Pa)",
            "r_w": "r_w (m)",
            "v_m": "V_m (m³)",
            "w0": "Initial Guess",
        }

        for param in lg_parameters:
            frame = QFrame()
            frame.setStyleSheet("background-color: white; border-radius: 5px;")
            row = QHBoxLayout(frame)
            row.setContentsMargins(8, 5, 8, 5)

            label = QLabel(f"{lg_descriptions[param]}: ")
            label.setFixedWidth(label_width)
            label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            label.setStyleSheet(font_style)

            input_field = QLineEdit()
            input_field.setFixedWidth(input_width)
            input_field.setStyleSheet(
                font_style + "padding: 5px; border: 1px solid #bdc3c7;"
            )
            input_field.setText(lg_defaults[param])  # Set default value

            row.addWidget(label)
            row.addWidget(input_field)
            row.addStretch()

            self.lg_inputs[param] = input_field
            self.lg_layout.addWidget(frame)

        # Output for Lietard-Griffiths
        output_frame = QFrame()
        output_frame.setStyleSheet("background-color: #e8f4fc; border-radius: 5px;")
        output_row = QHBoxLayout(output_frame)
        output_row.setContentsMargins(8, 8, 8, 8)

        output_label = QLabel("Result (w): ")
        output_label.setFixedWidth(label_width)
        output_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        output_label.setStyleSheet(font_style)

        self.lg_output_label = QLabel("-")
        self.lg_output_label.setStyleSheet(
            font_style + "font-weight: bold; color: #16a085;"
        )
        self.lg_output_label.setMinimumWidth(180)  # Ensure enough space for result

        output_row.addWidget(output_label)
        output_row.addWidget(self.lg_output_label)
        output_row.addStretch()

        self.lg_layout.addWidget(output_frame)

        # Right module: Verga Model
        verga_group = QGroupBox("Verga Model")
        verga_group.setStyleSheet("""
            QGroupBox {
                font-size: 16px;
                font-weight: bold;
                border: 2px solid #9b59b6;
                border-radius: 8px;
                margin-top: 15px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #8e44ad;
            }
        """)
        verga_group.setMinimumWidth(430)
        verga_group.setMaximumWidth(500)

        self.verga_layout = QVBoxLayout(verga_group)
        self.verga_layout.setContentsMargins(15, 10, 15, 10)  # Reduced vertical margins
        self.verga_layout.setSpacing(8)  # Reduced spacing between elements

        # Default values for Verga
        verga_defaults = {
            "delta_p": "10.0",
            "Q_loss": "0.01",
            "mu": "0.001",
            "r_w": "0.05",
            "v_m": "0.02",
            "w0": "0.01",
        }

        # Parameter descriptions
        verga_descriptions = {
            "delta_p": "Delta P (Pa)",
            "Q_loss": "Q_loss (m³/s)",
            "mu": "mu (Pa·s)",
            "r_w": "r_w (m)",
            "v_m": "V_m (m³)",
            "w0": "Initial Guess",
        }

        self.verga_inputs = {}
        verga_parameters = ["delta_p", "Q_loss", "mu", "r_w", "v_m", "w0"]

        for param in verga_parameters:
            frame = QFrame()
            frame.setStyleSheet("background-color: white; border-radius: 5px;")
            row = QHBoxLayout(frame)
            row.setContentsMargins(8, 5, 8, 5)

            label = QLabel(f"{verga_descriptions[param]}: ")
            label.setFixedWidth(label_width)
            label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            label.setStyleSheet(font_style)

            input_field = QLineEdit()
            input_field.setFixedWidth(input_width)
            input_field.setStyleSheet(
                font_style + "padding: 5px; border: 1px solid #bdc3c7;"
            )
            input_field.setText(verga_defaults[param])  # Set default value

            row.addWidget(label)
            row.addWidget(input_field)
            row.addStretch()

            self.verga_inputs[param] = input_field
            self.verga_layout.addWidget(frame)

        # Output for Verga
        output_frame = QFrame()
        output_frame.setStyleSheet("background-color: #f5eef8; border-radius: 5px;")
        output_row = QHBoxLayout(output_frame)
        output_row.setContentsMargins(8, 8, 8, 8)

        output_label = QLabel("Result (w): ")
        output_label.setFixedWidth(label_width)
        output_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        output_label.setStyleSheet(font_style)

        self.verga_output_label = QLabel("-")
        self.verga_output_label.setStyleSheet(
            font_style + "font-weight: bold; color: #8e44ad;"
        )
        self.verga_output_label.setMinimumWidth(180)  # Ensure enough space for result

        output_row.addWidget(output_label)
        output_row.addWidget(self.verga_output_label)
        output_row.addStretch()

        self.verga_layout.addWidget(output_frame)

        # Add layouts to main layout
        self.layout.addWidget(lg_group)
        self.layout.addWidget(verga_group)

        # Create horizontal control panel instead of vertical
        control_frame = QFrame()
        control_frame.setStyleSheet(
            "background-color: white; border-radius: 6px; border: 1px solid #dcdde1;"
        )
        control_inner_layout = QHBoxLayout(control_frame)
        control_inner_layout.setContentsMargins(10, 6, 10, 6)  # Reduced padding

        # Calculation button - smaller size
        self.identify_btn = QPushButton("计算")
        self.identify_btn.setStyleSheet("""
            QPushButton {
                font-size: 12px;
                font-weight: bold;
                background-color: #1e824c;
                color: white;
                padding: 4px 8px;
                border-radius: 3px;
                min-height: 24px;
                max-width: 80px;
            }
            QPushButton:hover {
                background-color: #196f3d;
            }
        """)
        self.identify_btn.clicked.connect(self.run_calculation)

        # Status Label - in the middle, flexible width
        self.status_label = QLabel("状态: 等待操作")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("""
            font-size: 12px;
            padding: 4px;
            background-color: #f8f9fa;
            border-radius: 3px;
            margin: 0 10px;
            min-height: 24px;
        """)

        # Download Data Button - smaller size
        self.download_btn = QPushButton("下载数据")
        self.download_btn.setStyleSheet("""
            QPushButton {
                font-size: 12px;
                font-weight: bold;
                background-color: #3498db;
                color: white;
                padding: 4px 8px;
                border-radius: 3px;
                min-height: 24px;
                max-width: 80px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        self.download_btn.clicked.connect(self.download_data)

        # Add to control layout in a single row
        control_inner_layout.addWidget(self.identify_btn)
        control_inner_layout.addWidget(self.status_label, 1)
        control_inner_layout.addWidget(self.download_btn)

        # Add to main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.addLayout(self.layout)
        main_layout.addSpacing(2)  # Further reduced spacing
        main_layout.addWidget(control_frame)

        self.setLayout(main_layout)
        self.setMinimumSize(900, 650)

    def run_calculation(self):
        QApplication.processEvents()
        self.status_label.setText("状态: 计算中...")
        self.status_label.setStyleSheet("""
            font-size: 12px;
            padding: 4px;
            min-height: 24px;
            background-color: #fff3cd;
            color: #856404;
            border-radius: 3px;
            margin: 0 10px;
        """)

        success = True
        error_message = ""

        try:
            # Lietard-Griffiths Model Calculation
            delta_p = float(self.lg_inputs["delta_p"].text())
            tau_y = float(self.lg_inputs["tau_y"].text())
            r_w = float(self.lg_inputs["r_w"].text())
            v_m = float(self.lg_inputs["v_m"].text())
            w0 = float(self.lg_inputs["w0"].text())

            self.lg_result = newton_method_Lietard(
                w0, delta_p, tau_y, r_w, v_m, tol=1e-6, max_iter=100
            )

            if self.lg_result is not None:
                self.lg_output_label.setText(f"{self.lg_result:.6f}")
            else:
                self.lg_output_label.setText("计算失败")
                success = False
                error_message += "Lietard-Griffiths模型计算失败。\n"

            # Verga Model Calculation
            delta_p = float(self.verga_inputs["delta_p"].text())
            Q_loss = float(self.verga_inputs["Q_loss"].text())
            mu = float(self.verga_inputs["mu"].text())
            r_w = float(self.verga_inputs["r_w"].text())
            v_m = float(self.verga_inputs["v_m"].text())
            w0 = float(self.verga_inputs["w0"].text())

            self.verga_result = newton_method_Verga(
                w0, delta_p, Q_loss, mu, r_w, v_m, tol=1e-6, max_iter=100
            )

            if self.verga_result is not None:
                self.verga_output_label.setText(f"{self.verga_result:.6f}")
            else:
                self.verga_output_label.setText("计算失败")
                success = False
                error_message += "Verga模型计算失败。\n"

            if success:
                self.status_label.setText("状态: 计算完成 ✅")
                self.status_label.setStyleSheet("""
                    font-size: 12px;
                    padding: 4px;
                    min-height: 24px;
                    background-color: #d4edda;
                    color: #155724;
                    border-radius: 3px;
                    margin: 0 10px;
                """)
                QMessageBox.information(self, "计算完成", "计算已成功完成！")
            else:
                self.status_label.setText("状态: 部分计算失败 ⚠️")
                self.status_label.setStyleSheet("""
                    font-size: 12px;
                    padding: 4px;
                    min-height: 24px;
                    background-color: #fff3cd;
                    color: #856404;
                    border-radius: 3px;
                    margin: 0 10px;
                """)
                QMessageBox.warning(self, "计算部分失败", error_message)

        except ValueError as e:
            self.status_label.setText("状态: 输入错误 ❌")
            self.status_label.setStyleSheet("""
                font-size: 12px;
                padding: 4px;
                min-height: 24px;
                background-color: #f8d7da;
                color: #721c24;
                border-radius: 3px;
                margin: 0 10px;
            """)
            self.lg_output_label.setText("-")
            self.verga_output_label.setText("-")
            QMessageBox.critical(
                self, "输入错误", f"请检查输入值是否为有效数字: {str(e)}"
            )

    def download_data(self):
        try:
            # Check if calculations were performed
            if self.lg_result is None and self.verga_result is None:
                QMessageBox.warning(self, "无数据", "请先进行计算后再下载数据。")
                return

            # Create data for excel file
            data = {
                "Parameter": [
                    "Delta P (Pa)",
                    "Tau Y (Pa)",
                    "r_w (m)",
                    "V_m (m³)",
                    "Initial w0",
                    "Result w",
                    "",
                    "Delta P (Pa)",
                    "Q_loss (m³/s)",
                    "mu (Pa·s)",
                    "r_w (m)",
                    "V_m (m³)",
                    "Initial w0",
                    "Result w",
                ],
                "Lietard-Griffiths Model": [
                    self.lg_inputs["delta_p"].text(),
                    self.lg_inputs["tau_y"].text(),
                    self.lg_inputs["r_w"].text(),
                    self.lg_inputs["v_m"].text(),
                    self.lg_inputs["w0"].text(),
                    str(self.lg_result) if self.lg_result is not None else "计算失败",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                ],
                "Verga Model": [
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    self.verga_inputs["delta_p"].text(),
                    self.verga_inputs["Q_loss"].text(),
                    self.verga_inputs["mu"].text(),
                    self.verga_inputs["r_w"].text(),
                    self.verga_inputs["v_m"].text(),
                    self.verga_inputs["w0"].text(),
                    str(self.verga_result)
                    if self.verga_result is not None
                    else "计算失败",
                ],
            }

            # Convert to DataFrame
            df = pd.DataFrame(data)

            # Ask user where to save the file
            options = QFileDialog.Options()
            file_name, _ = QFileDialog.getSaveFileName(
                self,
                "保存Excel文件",
                "",
                "Excel Files (*.xlsx);;All Files (*)",
                options=options,
            )

            if file_name:
                # Add .xlsx extension if not present
                if not file_name.endswith(".xlsx"):
                    file_name += ".xlsx"

                # Create Excel writer with pandas
                with pd.ExcelWriter(file_name, engine="openpyxl") as writer:
                    df.to_excel(writer, sheet_name="Calculation Results", index=False)

                self.status_label.setText(
                    f"状态: 数据已成功下载至 {os.path.basename(file_name)} ✅"
                )
                self.status_label.setStyleSheet("""
                    font-size: 12px;
                    padding: 4px;
                    min-height: 24px;
                    background-color: #d4edda;
                    color: #155724;
                    border-radius: 3px;
                    margin: 0 10px;
                """)

                QMessageBox.information(
                    self, "下载完成", f"数据已成功保存至:\n{file_name}"
                )

        except Exception as e:
            self.status_label.setText("状态: 下载失败 ❌")
            self.status_label.setStyleSheet("""
                font-size: 12px;
                padding: 4px;
                min-height: 24px;
                background-color: #f8d7da;
                color: #721c24;
                border-radius: 3px;
                margin: 0 10px;
            """)
            QMessageBox.critical(self, "下载错误", f"保存数据时出错: {str(e)}")
