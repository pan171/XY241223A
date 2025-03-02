from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QLineEdit,
    QApplication,
    QHBoxLayout,
)
import numpy as np


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

    def initUI(self):
        self.layout = QHBoxLayout()

        # Left module: Lietard-Griffiths Model
        self.lg_layout = QVBoxLayout()
        self.lg_label = QLabel("Lietard-Griffiths Model")
        self.lg_label.setAlignment(Qt.AlignCenter)
        self.lg_layout.addWidget(self.lg_label)

        self.lg_inputs = {}
        lg_parameters = ["delta_p", "tau_y", "r_w", "v_m", "w0"]
        for param in lg_parameters:
            row = QHBoxLayout()
            label = QLabel(f"{param}: ")
            input_field = QLineEdit()
            row.addWidget(label)
            row.addWidget(input_field)
            self.lg_inputs[param] = input_field
            self.lg_layout.addLayout(row)

        self.lg_output_label = QLabel("w_root: ")
        self.lg_layout.addWidget(self.lg_output_label)

        # Right module: Verga Model
        self.verga_layout = QVBoxLayout()
        self.verga_label = QLabel("Verga Model")
        self.verga_label.setAlignment(Qt.AlignCenter)
        self.verga_layout.addWidget(self.verga_label)

        self.verga_inputs = {}
        verga_parameters = ["delta_p", "Q_loss", "mu", "r_w", "v_m", "w0"]
        for param in verga_parameters:
            row = QHBoxLayout()
            label = QLabel(f"{param}: ")
            input_field = QLineEdit()
            row.addWidget(label)
            row.addWidget(input_field)
            self.verga_inputs[param] = input_field
            self.verga_layout.addLayout(row)

        self.verga_output_label = QLabel("w_root: ")
        self.verga_layout.addWidget(self.verga_output_label)

        # Calculation button
        self.identify_btn = QPushButton("计算")
        self.identify_btn.clicked.connect(self.run_calculation)

        # Status Label
        self.status_label = QLabel("状态: 等待操作")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setFixedHeight(25)
        self.status_label.setStyleSheet("font-size: 12px; padding: 2px;")

        # Download Data Button
        self.download_btn = QPushButton("下载数据")
        self.download_btn.clicked.connect(self.download_data)

        # Add layouts to main layout
        self.layout.addLayout(self.lg_layout)
        self.layout.addLayout(self.verga_layout)

        main_layout = QVBoxLayout()
        main_layout.addLayout(self.layout)
        main_layout.addWidget(self.identify_btn)
        main_layout.addWidget(self.status_label)
        main_layout.addWidget(self.download_btn)

        self.setLayout(main_layout)

    def run_calculation(self):
        QApplication.processEvents()
        self.status_label.setText("状态: 计算中...")

        try:
            # Lietard-Griffiths Model Calculation
            delta_p = float(self.lg_inputs["delta_p"].text())
            tau_y = float(self.lg_inputs["tau_y"].text())
            r_w = float(self.lg_inputs["r_w"].text())
            v_m = float(self.lg_inputs["v_m"].text())
            w0 = float(self.lg_inputs["w0"].text())
            w_root_lg = newton_method_Lietard(
                w0, delta_p, tau_y, r_w, v_m, tol=1e-6, max_iter=100
            )
            self.lg_output_label.setText(f"w_root: {w_root_lg:.4f}")

            # Verga Model Calculation (Dummy Computation for now)
            delta_p = float(self.verga_inputs["delta_p"].text())
            Q_loss = float(self.verga_inputs["Q_loss"].text())
            mu = float(self.verga_inputs["mu"].text())
            r_w = float(self.verga_inputs["r_w"].text())
            v_m = float(self.verga_inputs["v_m"].text())
            w0 = float(self.verga_inputs["w0"].text())
            w_root_verga = newton_method_Verga(
                w0, delta_p, Q_loss, mu, r_w, v_m, tol=1e-6, max_iter=100
            )
            self.verga_output_label.setText(f"w_root: {w_root_verga:.4f}")

            self.status_label.setText("状态: 计算完成 ✅")
        except ValueError:
            self.status_label.setText("状态: 输入错误 ❌")

    def download_data(self):
        # Placeholder for downloading data functionality
        self.status_label.setText("状态: 数据已准备好下载，此按钮还需修改 ✅")
