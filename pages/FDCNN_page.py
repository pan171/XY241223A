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
    QRadioButton,
    QButtonGroup,
)
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from models.fdcnn_model import CustomModel as FDCNNModel


class LossCanvas(FigureCanvas):
    def __init__(self, parent=None, width=4, height=3, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(LossCanvas, self).__init__(self.fig)
        self.setParent(parent)
        self.fig.tight_layout()
        self.plot_empty()

    def plot_empty(self):
        self.axes.clear()
        # self.axes.set_title("Training Loss Curve")
        self.axes.set_xlabel("Iterations")
        self.axes.set_ylabel("Loss")
        self.axes.grid(True, linestyle="--", alpha=0.7)
        self.draw()

    def plot_loss_curve(self, iterations, train_losses, val_losses=None):
        self.axes.clear()
        self.axes.plot(iterations, train_losses, "b-", linewidth=2, label="训练损失")
        if val_losses is not None and len(val_losses) > 0:
            # 创建相同长度的x轴值
            val_x = np.linspace(iterations[0], iterations[-1], len(val_losses))
            self.axes.plot(val_x, val_losses, "r-", linewidth=2, label="测试损失")
            self.axes.legend(loc='upper right')
            
        self.axes.set_xlabel("迭代次数")
        self.axes.set_ylabel("损失")
        self.axes.grid(True, linestyle="--", alpha=0.7)
        self.fig.tight_layout()
        self.draw()

    def save_figure(self, filename):
        self.fig.savefig(filename, dpi=300, bbox_inches="tight")


# Simulate FDCNN network training loss
def train_fdcnn_network(X_tensor, y_tensor, hidden_size, penalty_factor, has_constraints, train_size=None, test_size=None, epochs=100, batch_size=8, lr=0.01):
    """训练FDCNN神经网络模型并返回训练过程数据
    
    Args:
        X_tensor: 输入特征张量
        y_tensor: 目标值张量
        hidden_size: 隐藏层大小
        penalty_factor: 惩罚因子
        has_constraints: 是否应用约束
        train_size: 训练集大小
        test_size: 测试集大小
        epochs: 训练周期数
        batch_size: 批次大小
        lr: 学习率
    """
    # 设置设备（CPU或GPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建数据集
    dataset = TensorDataset(X_tensor, y_tensor.view(-1, 1))
    
    # 确定训练集和测试集大小
    total_size = len(dataset)
    if train_size is None or test_size is None:
        # 默认使用80%数据作为训练集，20%作为测试集
        test_size = int(total_size * 0.2)
        train_size = total_size - test_size
    else:
        # 限制训练集和测试集大小
        train_size = min(train_size, total_size)  # 训练集大小不超过总数据量
        test_size = min(test_size, total_size)    # 测试集大小不超过总数据量
    
    # 生成随机索引，用于选择数据
    indices = torch.randperm(total_size).tolist()
    
    # 创建训练集和测试集（使用不重叠的索引）
    train_indices = indices[:train_size]
    test_indices = indices[train_size:train_size + test_size]  # 确保没有重叠
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 创建模型
    input_size = X_tensor.shape[1]  # 特征数量
    output_size = 1
    model = FDCNNModel(input_size, hidden_size, output_size, rho=penalty_factor, device=device)
    model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0 if not has_constraints else penalty_factor * 0.001)
    
    # 训练循环
    iterations = []
    train_losses = []
    test_losses = []
    
    total_iterations = 0
    best_test_loss = float('inf')
    patience = 10  # 早停耐心值
    patience_counter = 0
    early_stop_message = ""
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # 前向传播
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 应用约束（如果启用）
            if has_constraints:
                # 为约束使用原始的特征数量限制和控制参数
                # 对于约束来说，保留fn和fm的原始含义
                # 约束应用于前7个特征（如果有），或者最多特征数量
                fm = min(7, input_size)  # 对应于原来的n
                fn = min(3, fm)          # 对应于原来的m
                model.customed_enforce_constraints(fm=fm, fn=fn)
            
            epoch_loss += loss.item()
            total_iterations += 1
            
            # 每个小批量后记录训练损失
            iterations.append(total_iterations)
            train_losses.append(loss.item())
        
        # 测试阶段
        model.eval()
        test_loss = 0
        if len(test_loader) > 0:  # 确保测试集不为空
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X)
                    test_loss += criterion(outputs, batch_y).item()
            
            avg_test_loss = test_loss / len(test_loader)
            test_losses.append(avg_test_loss)
            
            # 早停检查
            if avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    early_stop_message = f"早停触发，在迭代 {total_iterations} 处停止训练"
                    break
        else:
            # 如果测试集为空，则不记录测试损失，也不执行早停
            test_losses.append(0.0)  # 添加一个占位值，保持与迭代次数同步
                
    # 返回训练历史和模型
    return model, iterations, train_losses, test_losses, early_stop_message


class FDCNNPage(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.training_results = None
        self.data_uploaded = False
        self.model = None
        self.prediction_results = []  # 存储预测结果
        self.scaler_X = None
        self.scaler_y = None
        self.X_tensor = None
        self.y_tensor = None

    def initUI(self):
        # Set global font size and styles
        font_style = "font-size: 14px;"
        # title_style = "font-size: 16px; font-weight: bold; color: #2c3e50;"
        label_width = 150  # Width for labels
        input_width = 180  # Width for inputs

        # Set up the main frame with a light background color
        self.setStyleSheet("background-color: #f5f5f7;")

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(15, 15, 15, 15)

        # Parameters Group with two columns
        params_group = QGroupBox("FDCNN Neural Network Parameters")
        params_group.setStyleSheet("""
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

        # Create a horizontal layout for the two columns
        params_horizontal_layout = QHBoxLayout(params_group)
        params_horizontal_layout.setContentsMargins(15, 10, 15, 10)
        params_horizontal_layout.setSpacing(15)

        # Left column - Training parameters
        left_column = QVBoxLayout()
        left_column.setSpacing(8)

        # Default values
        default_values = {
            "hidden_size": "15",
            "penalty_factor": "20",
            "n": "3",
            "m": "7",
        }

        # Parameter descriptions
        param_descriptions = {
            "hidden_size": "隐藏层纬度",
            "penalty_factor": "惩罚因子",
            "n": "测试集数目",
            "m": "训练集数目",
        }

        # Create input fields
        self.inputs = {}
        parameters = ["hidden_size", "penalty_factor", "n", "m"]

        for param in parameters:
            frame = QFrame()
            frame.setStyleSheet("background-color: white; border-radius: 5px;")
            row = QHBoxLayout(frame)
            row.setContentsMargins(8, 5, 8, 5)

            label = QLabel(f"{param_descriptions[param]}: ")
            label.setFixedWidth(label_width)
            label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            label.setStyleSheet(font_style)

            input_field = QLineEdit()
            input_field.setFixedWidth(input_width)
            input_field.setStyleSheet(
                font_style + "padding: 5px; border: 1px solid #bdc3c7;"
            )
            input_field.setText(default_values[param])

            row.addWidget(label)
            row.addWidget(input_field)
            row.addStretch()

            self.inputs[param] = input_field
            left_column.addWidget(frame)

        # Constraints option (checkbox)
        constraint_frame = QFrame()
        constraint_frame.setStyleSheet("background-color: white; border-radius: 5px;")
        constraint_layout = QHBoxLayout(constraint_frame)
        constraint_layout.setContentsMargins(8, 5, 8, 5)

        constraint_label = QLabel("Has Constraints: ")
        constraint_label.setFixedWidth(label_width)
        constraint_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        constraint_label.setStyleSheet(font_style)

        self.constraint_group = QButtonGroup()

        self.yes_radio = QRadioButton("Yes")
        self.yes_radio.setChecked(True)
        self.yes_radio.setStyleSheet(font_style)

        self.no_radio = QRadioButton("No")
        self.no_radio.setStyleSheet(font_style)

        self.constraint_group.addButton(self.yes_radio)
        self.constraint_group.addButton(self.no_radio)

        constraint_layout.addWidget(constraint_label)
        constraint_layout.addWidget(self.yes_radio)
        constraint_layout.addWidget(self.no_radio)
        constraint_layout.addStretch()

        left_column.addWidget(constraint_frame)

        # Data Upload and Buttons
        data_panel = QFrame()
        data_panel.setStyleSheet("background-color: white; border-radius: 5px;")
        data_layout = QHBoxLayout(data_panel)
        data_layout.setContentsMargins(8, 8, 8, 8)

        # Upload Data Button
        self.upload_btn = QPushButton("上传数据")
        self.upload_btn.setStyleSheet("""
            QPushButton {
                font-size: 12px;
                font-weight: bold;
                background-color: #f39c12;
                color: white;
                padding: 4px 12px;
                border-radius: 3px;
                min-height: 28px;
            }
            QPushButton:hover {
                background-color: #e67e22;
            }
        """)
        self.upload_btn.clicked.connect(self.upload_data)

        # Train Network Button (moved from control panel)
        self.train_btn = QPushButton("训练网络")
        self.train_btn.setStyleSheet("""
            QPushButton {
                font-size: 12px;
                font-weight: bold;
                background-color: #1e824c;
                color: white;
                padding: 4px 12px;
                border-radius: 3px;
                min-height: 28px;
            }
            QPushButton:hover {
                background-color: #196f3d;
            }
        """)
        self.train_btn.clicked.connect(self.train_network)

        # Save Plot Button (moved from control panel)
        self.save_plot_btn = QPushButton("保存图片")
        self.save_plot_btn.setStyleSheet("""
            QPushButton {
                font-size: 12px;
                font-weight: bold;
                background-color: #9b59b6;
                color: white;
                padding: 4px 12px;
                border-radius: 3px;
                min-height: 28px;
            }
            QPushButton:hover {
                background-color: #8e44ad;
            }
        """)
        self.save_plot_btn.clicked.connect(self.save_plot)

        # Data Status Label
        self.data_status = QLabel("Data Status: No data uploaded")
        self.data_status.setStyleSheet("""
            font-size: 12px;
            padding: 4px 8px;
            background-color: #f8f9fa;
            border-radius: 3px;
        """)

        data_layout.addWidget(self.upload_btn)
        data_layout.addWidget(self.train_btn)
        data_layout.addWidget(self.save_plot_btn)
        data_layout.addWidget(self.data_status, 1)

        left_column.addWidget(data_panel)
        left_column.addStretch()

        # Right column - Inference section
        right_column = QVBoxLayout()
        right_column.setSpacing(8)

        # Inference Group
        inference_group = QGroupBox("Inference")
        inference_group.setStyleSheet("""
            QGroupBox {
                font-size: 14px;
                font-weight: bold;
                border: 2px solid #2ecc71;
                border-radius: 8px;
                margin-top: 5px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #27ae60;
            }
        """)

        inference_layout = QVBoxLayout(inference_group)
        inference_layout.setContentsMargins(10, 10, 10, 10)
        inference_layout.setSpacing(8)

        # Input fields for inference
        inference_inputs = {
            "well_depth": "井深",
            "loss_rate": "漏失速度",
            "loss_volume": "漏失量",
            "plastic_viscosity": "塑性粘度",
            "drilling_speed": "钻速",
            "mud_flow": "钻井液排量",
            "pump_pressure": "泵压",
        }

        self.inference_input_fields = {}

        for key, label_text in inference_inputs.items():
            frame = QFrame()
            frame.setStyleSheet("background-color: white; border-radius: 5px;")
            row = QHBoxLayout(frame)
            row.setContentsMargins(8, 5, 8, 5)

            label = QLabel(f"{label_text}: ")
            label.setFixedWidth(
                label_width - 30
            )  # Slightly narrower for the right column
            label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            label.setStyleSheet(font_style)

            input_field = QLineEdit()
            input_field.setFixedWidth(
                input_width - 20
            )  # Slightly narrower for the right column
            input_field.setStyleSheet(
                font_style + "padding: 5px; border: 1px solid #bdc3c7;"
            )

            row.addWidget(label)
            row.addWidget(input_field)
            row.addStretch()

            self.inference_input_fields[key] = input_field
            inference_layout.addWidget(frame)

        # Predict button
        predict_frame = QFrame()
        predict_frame.setStyleSheet("background-color: white; border-radius: 5px;")
        predict_layout = QHBoxLayout(predict_frame)
        predict_layout.setContentsMargins(8, 5, 8, 5)

        self.predict_btn = QPushButton("预测")
        self.predict_btn.setStyleSheet("""
            QPushButton {
                font-size: 14px;
                font-weight: bold;
                background-color: #3498db;
                color: white;
                padding: 6px 15px;
                border-radius: 3px;
                min-height: 30px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        self.predict_btn.clicked.connect(self.predict_fracture_width)

        predict_layout.addStretch()
        predict_layout.addWidget(self.predict_btn)
        predict_layout.addStretch()

        inference_layout.addWidget(predict_frame)

        # Output field - Fracture width
        output_frame = QFrame()
        output_frame.setStyleSheet("background-color: white; border-radius: 5px;")
        output_layout = QHBoxLayout(output_frame)
        output_layout.setContentsMargins(8, 5, 8, 5)

        output_label = QLabel("裂缝宽度: ")
        output_label.setFixedWidth(label_width - 30)
        output_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        output_label.setStyleSheet(font_style)

        self.fracture_width_output = QLineEdit()
        self.fracture_width_output.setFixedWidth(input_width - 20)
        self.fracture_width_output.setStyleSheet(
            font_style + "padding: 5px; border: 1px solid #bdc3c7;"
        )

        output_layout.addWidget(output_label)
        output_layout.addWidget(self.fracture_width_output)
        output_layout.addStretch()

        inference_layout.addWidget(output_frame)
        inference_layout.addStretch()

        right_column.addWidget(inference_group)
        right_column.addStretch()

        # Add both columns to the horizontal layout
        params_horizontal_layout.addLayout(left_column)
        params_horizontal_layout.addLayout(right_column)

        # Add parameters group to main layout
        main_layout.addWidget(params_group)

        # Canvas for loss curve plot (smaller size)
        plot_group = QGroupBox("Training Loss Curve")
        plot_group.setStyleSheet("""
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
        plot_layout = QVBoxLayout(plot_group)
        plot_layout.setContentsMargins(15, 15, 15, 15)

        # Smaller canvas with more padding to prevent overlaps
        self.canvas = LossCanvas(self, width=4, height=3)
        plot_layout.addWidget(self.canvas)

        main_layout.addWidget(plot_group)

        # Control panel with multiple buttons
        control_frame = QFrame()
        control_frame.setStyleSheet(
            "background-color: white; border-radius: 6px; border: 1px solid #dcdde1;"
        )
        control_layout = QHBoxLayout(control_frame)
        control_layout.setContentsMargins(10, 6, 10, 6)

        # Load Model Button
        self.load_model_btn = QPushButton("加载模型")
        self.load_model_btn.setStyleSheet("""
            QPushButton {
                font-size: 12px;
                font-weight: bold;
                background-color: #2980b9;
                color: white;
                padding: 4px 8px;
                border-radius: 3px;
                min-height: 28px;
            }
            QPushButton:hover {
                background-color: #2471a3;
            }
        """)
        self.load_model_btn.clicked.connect(self.load_model)

        # Save Model Button
        self.save_model_btn = QPushButton("保存模型")
        self.save_model_btn.setStyleSheet("""
            QPushButton {
                font-size: 12px;
                font-weight: bold;
                background-color: #27ae60;
                color: white;
                padding: 4px 8px;
                border-radius: 3px;
                min-height: 28px;
            }
            QPushButton:hover {
                background-color: #229954;
            }
        """)
        self.save_model_btn.clicked.connect(self.save_model)
        
        # Status label
        self.status_label = QLabel("状态: 准备就绪")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("""
            font-size: 12px;
            padding: 4px;
            background-color: #f8f9fa;
            border-radius: 3px;
            margin: 0 10px;
            min-height: 24px;
        """)

        # Export data button
        self.export_btn = QPushButton("导出数据")
        self.export_btn.setStyleSheet("""
            QPushButton {
                font-size: 12px;
                font-weight: bold;
                background-color: #e74c3c;
                color: white;
                padding: 4px 8px;
                border-radius: 3px;
                min-height: 28px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        self.export_btn.clicked.connect(self.export_data)

        # Add to control layout
        control_layout.addWidget(self.load_model_btn)
        control_layout.addWidget(self.save_model_btn)
        control_layout.addWidget(self.status_label, 1)
        control_layout.addWidget(self.export_btn)

        # Add control panel to main layout
        main_layout.addWidget(control_frame)

        self.setLayout(main_layout)
        self.setMinimumSize(800, 700)

        # Initial button states
        self.train_btn.setEnabled(False)
        self.save_plot_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        self.save_model_btn.setEnabled(False)
        self.predict_btn.setEnabled(False)

    def upload_data(self):
        """处理数据上传，支持Excel和CSV格式"""
        try:
            options = QFileDialog.Options()
            file_name, _ = QFileDialog.getOpenFileName(
                self,
                "上传数据文件",
                "",
                "数据文件 (*.csv *.xlsx *.xls);;全部文件 (*)",
                options=options,
            )

            if file_name:
                # 根据文件扩展名加载数据
                if file_name.endswith('.csv'):
                    df = pd.read_csv(file_name)
                else:  # Excel格式
                    df = pd.read_excel(file_name)
                
                # 检查数据是否包含所需列
                required_columns = ["井深", "漏失速度", "漏失量", "塑性黏度", "钻速", "钻井液排量", "泵压", "裂缝宽度"]
                input_columns = required_columns[:-1]  # 不包括目标列
                
                # 检查英文列名（可能的替代列名）
                alt_required_columns = ["well_depth", "loss_rate", "loss_volume", 
                              "plastic_viscosity", "drilling_speed", 
                              "mud_flow", "pump_pressure", "fracture_width"]
                
                # 尝试匹配列名（中文或英文）
                if not all(col in df.columns for col in required_columns):
                    if not all(col in df.columns for col in alt_required_columns):
                        # 如果列名与预期不一致，但数据有7个输入列和1个输出列，假设它们按顺序排列
                        if len(df.columns) >= 8:  # 至少需要8列（7个特征+1个目标）
                            df.columns = list(df.columns[:7]) + ["裂缝宽度"] + list(df.columns[8:])
                            QMessageBox.warning(
                                self, 
                                "列名警告", 
                                "数据列名与预期不匹配，已自动映射前7列为特征，第8列为裂缝宽度。请确认数据正确性。"
                            )
                        else:
                            raise ValueError(f"数据文件必须包含以下列：{', '.join(required_columns)}")
                
                # 保存原始数据
                self.original_df = df.copy()
                
                # 提取特征和目标
                if all(col in df.columns for col in required_columns):
                    X = df[input_columns].values
                    y = df["裂缝宽度"].values
                elif all(col in df.columns for col in alt_required_columns):
                    X = df[alt_required_columns[:-1]].values
                    y = df[alt_required_columns[-1]].values
                else:
                    # 使用前7列作为特征，第8列作为目标
                    X = df.iloc[:, :7].values
                    y = df.iloc[:, 7].values
                
                # 创建特征标准化器
                self.scaler_X = StandardScaler()
                self.scaler_y = StandardScaler()
                
                # 标准化特征和目标
                X_scaled = self.scaler_X.fit_transform(X)
                y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
                
                # 保存数据为Tensor
                self.X_tensor = torch.FloatTensor(X_scaled)
                self.y_tensor = torch.FloatTensor(y_scaled)
                
                # 保存原始数据用于展示
                self.X_original = X
                self.y_original = y
                
                # 更新状态
                self.status_label.setText(f"状态：已加载数据：{os.path.basename(file_name)} ✅")
                self.status_label.setStyleSheet("""
                    font-size: 12px;
                    padding: 4px;
                    min-height: 24px;
                    background-color: #d4edda;
                    color: #155724;
                    border-radius: 3px;
                    margin: 0 10px;
                """)

                self.data_status.setText(f"数据状态：已加载 {os.path.basename(file_name)}，共 {len(df)} 条记录")
                self.data_status.setStyleSheet("""
                    font-size: 12px;
                    padding: 4px 8px;
                    background-color: #d4edda;
                    color: #155724;
                    border-radius: 3px;
                """)

                self.data_uploaded = True
                self.train_btn.setEnabled(True)
                self.model = None  # 重置已有模型
                self.save_model_btn.setEnabled(False)  # 禁用保存模型按钮，直到模型被训练
                self.predict_btn.setEnabled(False)  # 禁用预测按钮，直到模型被训练
                self.export_btn.setEnabled(False)  # 禁用导出按钮，直到模型被训练

                QMessageBox.information(
                    self,
                    "上传成功",
                    f"数据文件加载成功：\n{file_name}\n共 {len(df)} 条记录",
                )
                
                # 显示样本数据
                sample_msg = "数据样例：\n"
                if len(df) > 0:
                    sample_row = df.iloc[0]
                    sample_inputs = []
                    if all(col in df.columns for col in input_columns):
                        cols_to_show = input_columns
                    elif all(col in df.columns for col in alt_required_columns[:-1]):
                        cols_to_show = alt_required_columns[:-1]
                    else:
                        cols_to_show = df.columns[:7]
                    
                    for i, col in enumerate(cols_to_show):
                        if col in df.columns:
                            sample_inputs.append(f"{col}: {sample_row[col]}")
                        else:
                            sample_inputs.append(f"特征{i+1}: {sample_row.iloc[i]}")
                    
                    # 获取目标列
                    if "裂缝宽度" in df.columns:
                        target_val = sample_row["裂缝宽度"]
                    elif "fracture_width" in df.columns:
                        target_val = sample_row["fracture_width"]
                    else:
                        target_val = sample_row.iloc[7]
                    
                    sample_msg += "\n".join(sample_inputs)
                    sample_msg += f"\n\n裂缝宽度: {target_val}"
                    
                    QMessageBox.information(self, "数据样例", sample_msg)
                
        except Exception as e:
            self.status_label.setText("状态：上传失败 ❌")
            self.status_label.setStyleSheet("""
                font-size: 12px;
                padding: 4px;
                min-height: 24px;
                background-color: #f8d7da;
                color: #721c24;
                border-radius: 3px;
                margin: 0 10px;
            """)
            QMessageBox.critical(
                self, "上传错误", f"数据上传错误：{str(e)}"
            )

    def train_network(self):
        """执行FDCNN神经网络训练"""
        if not self.data_uploaded:
            QMessageBox.warning(
                self, "无数据", "请先上传数据文件。"
            )
            return

        try:
            # 更新状态
            self.status_label.setText("状态：训练神经网络中...")
            self.status_label.setStyleSheet("""
                font-size: 12px;
                padding: 4px;
                min-height: 24px;
                background-color: #fff3cd;
                color: #856404;
                border-radius: 3px;
                margin: 0 10px;
            """)
            QApplication.processEvents()

            # 获取参数值
            hidden_size = int(self.inputs["hidden_size"].text())
            penalty_factor = int(self.inputs["penalty_factor"].text())
            n = int(self.inputs["n"].text())  # 测试数据数量
            m = int(self.inputs["m"].text())  # 训练数据数量
            has_constraints = self.yes_radio.isChecked()
            
            # 训练集大小和测试集大小
            train_size = m
            test_size = n
            
            # 配置训练参数
            epochs = 30  # 固定训练周期数
            batch_size = max(2, min(8, train_size // 10))  # 根据训练数据大小调整批次大小
            
            # 确保批次大小至少为1，且不大于训练数据量
            batch_size = max(1, min(batch_size, train_size))
            
            # 执行训练
            self.model, iterations, train_losses, test_losses, early_stop_message = train_fdcnn_network(
                self.X_tensor, self.y_tensor, 
                hidden_size, penalty_factor, has_constraints,
                train_size=train_size, test_size=test_size,
                epochs=epochs, batch_size=batch_size
            )
            
            # 存储结果
            self.training_results = {
                "iterations": iterations,
                "train_losses": train_losses,
                "test_losses": test_losses,
                "parameters": {
                    "hidden_size": hidden_size,
                    "penalty_factor": penalty_factor,
                    "n": n,
                    "m": m,
                    "has_constraints": has_constraints,
                    "epochs": epochs,
                    "batch_size": batch_size
                },
            }

            # 绘制损失曲线
            self.canvas.plot_loss_curve(iterations, train_losses, test_losses)

            # 更新状态
            status_message = "状态：训练完成 ✅"
            if early_stop_message:
                status_message += f" ({early_stop_message})"
                
            self.status_label.setText(status_message)
            self.status_label.setStyleSheet("""
                font-size: 12px;
                padding: 4px;
                min-height: 24px;
                background-color: #d4edda;
                color: #155724;
                border-radius: 3px;
                margin: 0 10px;
            """)

            # 启用相关按钮
            self.save_plot_btn.setEnabled(True)
            self.export_btn.setEnabled(True)
            self.save_model_btn.setEnabled(True)
            self.predict_btn.setEnabled(True)  # 启用预测按钮
            
            # 计算训练和测试集上的最终损失
            train_final_loss = train_losses[-1] if train_losses else 0
            test_final_loss = test_losses[-1] if test_losses else 0

            QMessageBox.information(
                self, "训练完成", 
                f"FDCNN神经网络训练已完成！\n"
                f"训练损失: {train_final_loss:.4f}\n"
                f"测试损失: {test_final_loss:.4f}\n"
                + (f"\n{early_stop_message}" if early_stop_message else "")
            )

        except ValueError as e:
            self.status_label.setText("状态：训练错误 ❌")
            self.status_label.setStyleSheet("""
                font-size: 12px;
                padding: 4px;
                min-height: 24px;
                background-color: #f8d7da;
                color: #721c24;
                border-radius: 3px;
                margin: 0 10px;
            """)
            QMessageBox.critical(
                self, "输入错误", f"请检查所有输入值是否有效：{str(e)}"
            )

        except Exception as e:
            self.status_label.setText("状态：训练失败 ❌")
            self.status_label.setStyleSheet("""
                font-size: 12px;
                padding: 4px;
                min-height: 24px;
                background-color: #f8d7da;
                color: #721c24;
                border-radius: 3px;
                margin: 0 10px;
            """)
            QMessageBox.critical(
                self, "错误", f"训练过程中发生错误：{str(e)}"
            )

    def save_plot(self):
        """Save the current plot as an image file"""
        try:
            options = QFileDialog.Options()
            file_name, _ = QFileDialog.getSaveFileName(
                self,
                "Save Plot",
                "",
                "PNG Files (*.png);;JPG Files (*.jpg);;PDF Files (*.pdf);;All Files (*)",
                options=options,
            )

            if file_name:
                # Add extension if not provided
                if not any(file_name.endswith(ext) for ext in [".png", ".jpg", ".pdf"]):
                    file_name += ".png"

                # Save the figure
                self.canvas.save_figure(file_name)

                self.status_label.setText(
                    f"Status: Plot saved to {os.path.basename(file_name)} ✅"
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
                    self, "Save Successful", f"Plot saved successfully to:\n{file_name}"
                )
        except Exception as e:
            self.status_label.setText("Status: Save failed ❌")
            self.status_label.setStyleSheet("""
                font-size: 12px;
                padding: 4px;
                min-height: 24px;
                background-color: #f8d7da;
                color: #721c24;
                border-radius: 3px;
                margin: 0 10px;
            """)
            QMessageBox.critical(self, "Save Error", f"Error saving plot: {str(e)}")

    def save_model(self):
        """保存训练好的模型"""
        if self.model is None:
            QMessageBox.warning(self, "无模型", "请先训练神经网络模型")
            return
            
        try:
            options = QFileDialog.Options()
            file_name, _ = QFileDialog.getSaveFileName(
                self,
                "保存模型",
                "",
                "PyTorch模型 (*.pt);;所有文件 (*)",
                options=options,
            )
            
            if file_name:
                # 确保扩展名为.pt
                if not file_name.endswith('.pt'):
                    file_name += '.pt'
                    
                # 保存模型状态
                model_state = {
                    'model_state_dict': self.model.state_dict(),
                    'input_size': self.model.layer1.in_features,
                    'hidden_size': self.model.layer1.out_features,
                    'output_size': self.model.layer2.out_features,
                    'rho': self.model.rho,  # 保存rho参数
                    'scaler_X_mean': self.scaler_X.mean_,
                    'scaler_X_scale': self.scaler_X.scale_,
                    'scaler_y_mean': self.scaler_y.mean_,
                    'scaler_y_scale': self.scaler_y.scale_,
                    'parameters': self.training_results['parameters'] if hasattr(self, 'training_results') else {}
                }
                
                torch.save(model_state, file_name)
                
                self.status_label.setText(f"状态：模型已保存至 {os.path.basename(file_name)} ✅")
                self.status_label.setStyleSheet("""
                    font-size: 12px;
                    padding: 4px;
                    min-height: 24px;
                    background-color: #d4edda;
                    color: #155724;
                    border-radius: 3px;
                    margin: 0 10px;
                """)
                
                QMessageBox.information(self, "保存成功", f"模型已成功保存至：\n{file_name}")
        except Exception as e:
            self.status_label.setText("状态：模型保存失败 ❌")
            self.status_label.setStyleSheet("""
                font-size: 12px;
                padding: 4px;
                min-height: 24px;
                background-color: #f8d7da;
                color: #721c24;
                border-radius: 3px;
                margin: 0 10px;
            """)
            QMessageBox.critical(self, "保存错误", f"保存模型时出错：{str(e)}")
    
    def load_model(self):
        """加载保存的模型"""
        try:
            options = QFileDialog.Options()
            file_name, _ = QFileDialog.getOpenFileName(
                self,
                "加载模型",
                "",
                "PyTorch模型 (*.pt);;所有文件 (*)",
                options=options,
            )
            
            if file_name:
                # 加载模型状态
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model_state = torch.load(file_name, map_location=device)
                
                # 创建模型
                input_size = model_state['input_size']
                hidden_size = model_state['hidden_size']
                output_size = model_state['output_size']
                rho = model_state.get('rho', 20)  # 默认为20，如果没有保存这个参数
                
                self.model = FDCNNModel(input_size, hidden_size, output_size, rho=rho, device=device)
                self.model.load_state_dict(model_state['model_state_dict'])
                self.model.eval()  # 设置为评估模式
                
                # 恢复标准化器
                self.scaler_X = StandardScaler()
                self.scaler_X.mean_ = model_state['scaler_X_mean']
                self.scaler_X.scale_ = model_state['scaler_X_scale']
                
                self.scaler_y = StandardScaler()
                self.scaler_y.mean_ = model_state['scaler_y_mean']
                self.scaler_y.scale_ = model_state['scaler_y_scale']
                
                # 恢复参数（如果有）
                if 'parameters' in model_state:
                    params = model_state['parameters']
                    # 更新UI中的参数
                    for param_name, param_value in params.items():
                        if param_name in self.inputs:
                            self.inputs[param_name].setText(str(param_value))
                    
                    # 更新约束选项
                    if 'has_constraints' in params:
                        if params['has_constraints']:
                            self.yes_radio.setChecked(True)
                        else:
                            self.no_radio.setChecked(True)
                
                self.status_label.setText(f"状态：模型已从 {os.path.basename(file_name)} 加载 ✅")
                self.status_label.setStyleSheet("""
                    font-size: 12px;
                    padding: 4px;
                    min-height: 24px;
                    background-color: #d4edda;
                    color: #155724;
                    border-radius: 3px;
                    margin: 0 10px;
                """)
                
                # 启用相关按钮
                self.predict_btn.setEnabled(True)
                self.save_model_btn.setEnabled(True)
                
                QMessageBox.information(
                    self, 
                    "加载成功", 
                    f"模型已成功从以下位置加载：\n{file_name}\n\n"
                    f"输入特征数：{input_size}\n"
                    f"隐藏层大小：{hidden_size}\n"
                    f"现在您可以使用该模型进行预测。"
                )
        except Exception as e:
            self.status_label.setText("状态：模型加载失败 ❌")
            self.status_label.setStyleSheet("""
                font-size: 12px;
                padding: 4px;
                min-height: 24px;
                background-color: #f8d7da;
                color: #721c24;
                border-radius: 3px;
                margin: 0 10px;
            """)
            QMessageBox.critical(self, "加载错误", f"加载模型时出错：{str(e)}")

    def export_data(self):
        if self.training_results is None:
            QMessageBox.warning(self, "无数据", "请先训练神经网络。")
            return

        try:
            # 询问用户保存文件的位置
            options = QFileDialog.Options()
            file_name, _ = QFileDialog.getSaveFileName(
                self,
                "保存Excel文件",
                "",
                "Excel文件 (*.xlsx);;CSV文件 (*.csv);;所有文件 (*)",
                options=options,
            )

            if not file_name:
                return

            # 添加适当的扩展名（如果不存在）
            if not (file_name.endswith(".xlsx") or file_name.endswith(".csv")):
                file_name += ".xlsx"

            # 准备用于导出的数据 - 训练损失
            iterations = self.training_results["iterations"]
            train_losses = self.training_results["train_losses"]
            
            # 检查数组长度
            if len(iterations) != len(train_losses):
                print(f"警告: 迭代次数({len(iterations)})和训练损失({len(train_losses)})长度不一致。尝试调整...")
                # 取较短的长度
                min_length = min(len(iterations), len(train_losses))
                iterations = iterations[:min_length]
                train_losses = train_losses[:min_length]
                
            # 创建基本数据框
            data = {}
            try:
                data["迭代次数"] = np.array(iterations, dtype=float)
                data["训练损失"] = np.array(train_losses, dtype=float)
            except Exception as arr_err:
                print(f"转换数据时出错: {str(arr_err)}")
                # 尝试逐个元素添加到列表中
                iterations_clean = []
                train_losses_clean = []
                for i in range(min(len(iterations), len(train_losses))):
                    try:
                        iterations_clean.append(float(iterations[i]))
                        train_losses_clean.append(float(train_losses[i]))
                    except (ValueError, TypeError):
                        continue
                data["迭代次数"] = iterations_clean
                data["训练损失"] = train_losses_clean
            
            # 如果有测试损失，创建匹配长度的数组
            test_losses_resampled = None
            if "test_losses" in self.training_results and self.training_results["test_losses"]:
                test_losses = self.training_results["test_losses"]
                # 将测试损失重采样到与训练损失相同的长度
                if len(test_losses) > 0:
                    try:
                        # 使用线性插值重采样测试损失
                        if len(test_losses) != len(data["迭代次数"]):
                            x_test = np.linspace(0, 1, len(test_losses))
                            x_new = np.linspace(0, 1, len(data["迭代次数"]))
                            test_losses_resampled = np.interp(x_new, x_test, test_losses)
                        else:
                            test_losses_resampled = np.array(test_losses, dtype=float)
                        
                        # 添加到数据字典
                        data["测试损失"] = test_losses_resampled
                    except Exception as test_err:
                        print(f"处理测试损失时出错: {str(test_err)}")
                        # 不添加测试损失数据
            
            # 创建训练损失数据框
            try:
                df = pd.DataFrame(data)
            except Exception as df_err:
                print(f"创建DataFrame时出错: {str(df_err)}")
                # 尝试创建一个空的DataFrame然后逐列添加
                df = pd.DataFrame()
                for key, values in data.items():
                    try:
                        df[key] = values
                    except Exception:
                        print(f"添加列 {key} 时出错，跳过此列")

            # 添加参数信息到一个单独的表格
            try:
                params_data = {
                    "参数": [
                        "隐藏层大小",
                        "惩罚因子",
                        "参数n",
                        "参数m",
                        "是否有约束",
                        "训练周期",
                        "批次大小",
                    ],
                    "值": [
                        self.training_results["parameters"]["hidden_size"],
                        self.training_results["parameters"]["penalty_factor"],
                        self.training_results["parameters"]["n"],
                        self.training_results["parameters"]["m"],
                        "是" if self.training_results["parameters"]["has_constraints"] else "否",
                        self.training_results["parameters"].get("epochs", "未指定"),
                        self.training_results["parameters"].get("batch_size", "未指定"),
                    ],
                }
                params_df = pd.DataFrame(params_data)
            except Exception as param_err:
                print(f"创建参数DataFrame时出错: {str(param_err)}")
                # 创建一个简单的参数DataFrame
                params_df = pd.DataFrame({"参数": ["错误"], "值": ["创建参数表格时出错"]})

            # 导出到文件
            if file_name.endswith(".xlsx"):
                try:
                    with pd.ExcelWriter(file_name, engine="openpyxl") as writer:
                        df.to_excel(writer, sheet_name="训练损失", index=False)
                        params_df.to_excel(writer, sheet_name="参数", index=False)
                        
                        # 如果有预测结果，也导出
                        if hasattr(self, 'prediction_results') and self.prediction_results:
                            # 检查预测结果是否为空
                            if len(self.prediction_results) > 0:
                                try:
                                    # 尝试创建预测结果DataFrame
                                    pred_df = pd.DataFrame(self.prediction_results)
                                    pred_df.to_excel(writer, sheet_name="预测结果", index=False)
                                except Exception as pred_err:
                                    print(f"导出预测结果时出错: {str(pred_err)}")
                                    # 继续导出其他数据，不因预测结果错误而中断
                except Exception as excel_err:
                    print(f"写入Excel文件时出错: {str(excel_err)}")
                    # 尝试只保存训练损失数据
                    try:
                        df.to_csv(file_name.replace(".xlsx", ".csv"), index=False)
                        QMessageBox.warning(
                            self,
                            "导出警告",
                            f"Excel导出失败，已保存为CSV格式: {file_name.replace('.xlsx', '.csv')}"
                        )
                        file_name = file_name.replace(".xlsx", ".csv")
                    except Exception:
                        raise  # 如果CSV也失败，则抛出原始错误
            else:  # CSV格式
                df.to_csv(file_name, index=False)
                # 对于CSV，我们不能包括多个表格，所以我们会在消息中注明这一点
                QMessageBox.information(
                    self,
                    "CSV导出",
                    "注意：参数信息和预测结果未包含在CSV格式中。",
                )

            # 同时保存损失曲线图
            try:
                img_file = os.path.splitext(file_name)[0] + "_loss_curve.png"
                self.canvas.save_figure(img_file)
            except Exception as img_err:
                print(f"保存损失曲线图时出错: {str(img_err)}")
                img_file = None

            # 更新状态
            self.status_label.setText(
                f"状态：数据已导出至 {os.path.basename(file_name)} ✅"
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

            success_msg = f"数据已成功保存至：\n{file_name}"
            if img_file:
                success_msg += f"\n\n损失曲线图已保存至：\n{img_file}"
                
            QMessageBox.information(self, "导出完成", success_msg)

        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            self.status_label.setText("状态：导出失败 ❌")
            self.status_label.setStyleSheet("""
                font-size: 12px;
                padding: 4px;
                min-height: 24px;
                background-color: #f8d7da;
                color: #721c24;
                border-radius: 3px;
                margin: 0 10px;
            """)
            print(f"导出数据时出错：\n{error_msg}")
            QMessageBox.critical(self, "导出错误", f"保存数据时出错：{str(e)}\n\n详细信息已打印到控制台。")

    def predict_fracture_width(self):
        """使用训练好的FDCNN模型预测裂缝宽度"""
        try:
            if self.model is None:
                QMessageBox.warning(self, "无模型", "请先训练神经网络模型")
                return
                
            # 获取输入值
            well_depth = float(self.inference_input_fields["well_depth"].text() or 0)
            loss_rate = float(self.inference_input_fields["loss_rate"].text() or 0)
            loss_volume = float(self.inference_input_fields["loss_volume"].text() or 0)
            plastic_viscosity = float(
                self.inference_input_fields["plastic_viscosity"].text() or 0
            )
            drilling_speed = float(
                self.inference_input_fields["drilling_speed"].text() or 0
            )
            mud_flow = float(self.inference_input_fields["mud_flow"].text() or 0)
            pump_pressure = float(
                self.inference_input_fields["pump_pressure"].text() or 0
            )
            
            # 创建输入特征向量
            input_features = np.array([
                well_depth, loss_rate, loss_volume, plastic_viscosity, 
                drilling_speed, mud_flow, pump_pressure
            ]).reshape(1, -1)
            
            # 标准化输入特征
            input_scaled = self.scaler_X.transform(input_features)
            
            # 转换为PyTorch Tensor
            input_tensor = torch.FloatTensor(input_scaled)
            
            # 预测
            self.model.eval()  # 设置为评估模式
            with torch.no_grad():
                output_tensor = self.model(input_tensor)
                
            # 转换回原始尺度
            output_scaled = output_tensor.numpy().reshape(-1, 1)
            output_original = self.scaler_y.inverse_transform(output_scaled)
            
            # 获取预测结果
            result = output_original[0, 0]
            
            # 格式化为4位小数
            formatted_result = f"{result:.4f}"
            
            # 显示结果
            self.fracture_width_output.setText(formatted_result)
            self.fracture_width_output.setStyleSheet(
                "font-size: 14px; padding: 5px; border: 1px solid #27ae60; color: #155724;"
            )

            self.status_label.setText("状态：预测完成 ✅")
            self.status_label.setStyleSheet("""
                font-size: 12px;
                padding: 4px;
                min-height: 24px;
                background-color: #d4edda;
                color: #155724;
                border-radius: 3px;
                margin: 0 10px;
            """)
            
            # 收集预测结果数据 - 确保所有字段都有值
            try:
                prediction_data = {
                    "井深": float(well_depth),
                    "漏失速度": float(loss_rate),
                    "漏失量": float(loss_volume),
                    "塑性黏度": float(plastic_viscosity),
                    "钻速": float(drilling_speed),
                    "钻井液排量": float(mud_flow),
                    "泵压": float(pump_pressure),
                    "预测裂缝宽度": float(formatted_result)
                }
                
                # 确保所有键值对格式一致
                if not hasattr(self, 'prediction_results'):
                    self.prediction_results = []
                    
                # 添加到预测结果列表
                self.prediction_results.append(prediction_data)
                
                # 如果出现结构不一致的预测结果，可以尝试统一格式
                if len(self.prediction_results) > 1:
                    # 获取所有预测结果的键
                    all_keys = set()
                    for pred in self.prediction_results:
                        all_keys.update(pred.keys())
                    
                    # 确保所有预测结果都有相同的键
                    for i, pred in enumerate(self.prediction_results):
                        for key in all_keys:
                            if key not in pred:
                                # 对缺失的键添加默认值
                                self.prediction_results[i][key] = 0.0
            except Exception as pred_err:
                print(f"保存预测结果时出错: {str(pred_err)}")
                # 继续执行，不因预测结果保存错误而中断
            
            # 显示实际输入特征和预测结果
            input_summary = (
                f"井深: {well_depth}\n"
                f"漏失速度: {loss_rate}\n"
                f"漏失量: {loss_volume}\n"
                f"塑性黏度: {plastic_viscosity}\n"
                f"钻速: {drilling_speed}\n"
                f"钻井液排量: {mud_flow}\n"
                f"泵压: {pump_pressure}\n\n"
                f"预测裂缝宽度: {formatted_result}"
            )
            
            QMessageBox.information(self, "预测详情", input_summary)

        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            self.status_label.setText(f"状态：预测失败 - {str(e)} ❌")
            self.status_label.setStyleSheet("""
                font-size: 12px;
                padding: 4px;
                min-height: 24px;
                background-color: #f8d7da;
                color: #721c24;
                border-radius: 3px;
                margin: 0 10px;
            """)
            print(f"预测过程中出错：\n{error_msg}")

            QMessageBox.warning(
                self,
                "预测错误",
                f"预测过程中出错：{str(e)}\n请检查您的输入值。\n\n详细信息已打印到控制台。",
            )
