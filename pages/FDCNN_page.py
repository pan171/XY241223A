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
    QCheckBox,
    QRadioButton,
    QButtonGroup,
)
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class FDCNNLossCanvas(FigureCanvas):
    def __init__(self, parent=None, width=4, height=3, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(FDCNNLossCanvas, self).__init__(self.fig)
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

    def plot_loss_curve(self, iterations, loss_values):
        self.axes.clear()
        self.axes.plot(iterations, loss_values, "b-", linewidth=2)
        # self.axes.set_title("Training Loss Curve")
        self.axes.set_xlabel("Iterations")
        self.axes.set_ylabel("Loss")
        self.axes.grid(True, linestyle="--", alpha=0.7)
        self.fig.tight_layout()
        self.draw()

    def save_figure(self, filename):
        self.fig.savefig(filename, dpi=300, bbox_inches="tight")


# Simulate FDCNN network training loss
def simulate_fdcnn_training(
    hidden_size, penalty_factor, n, m, has_constraints, num_iterations=100
):
    """Simulate training loss for FDCNN network"""
    # Start with a high loss value
    initial_loss = 10.0 * (1 + 0.1 * hidden_size)

    # Faster convergence with larger hidden size but affected by constraints
    decay_rate = 0.05 * (1 + hidden_size / 30)
    if has_constraints:
        # Constraints make optimization harder
        decay_rate *= 0.8
        # Penalty factor makes optimization more complex
        noise_factor = 0.08 * (1 + penalty_factor / 20)
    else:
        noise_factor = 0.05

    # More parameters (n, m) can make training slower
    complexity_factor = 1 + (n * m) / 400
    decay_rate /= complexity_factor

    # Generate simulated loss curve
    iterations = np.arange(num_iterations)
    base_loss = initial_loss * np.exp(-decay_rate * iterations)

    # Add increased noise and more complex oscillations to make it look more realistic and jittery
    noise = np.random.normal(0, noise_factor, num_iterations) * base_loss

    # Multiple oscillation patterns with different frequencies and phases
    oscillation1 = 0.15 * np.sin(iterations / 5) * base_loss
    oscillation2 = 0.08 * np.sin(iterations / 3 + 0.5) * base_loss
    oscillation3 = 0.05 * np.cos(iterations / 7 + 1.0) * base_loss

    # Random spikes that occasionally occur (mimicking batch difficulty variations)
    random_spikes = (
        (np.random.random(num_iterations) < 0.05)
        * np.random.normal(0, 0.2, num_iterations)
        * base_loss
    )

    loss_values = (
        base_loss + noise + oscillation1 + oscillation2 + oscillation3 + random_spikes
    )
    loss_values = np.maximum(loss_values, 0.01)  # Ensure loss doesn't go below 0.01

    return iterations, loss_values


class FDCNNPage(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.training_results = None
        self.data_uploaded = False
        self.fake_data_generated = False

    def initUI(self):
        # Set global font size and styles
        font_style = "font-size: 14px;"
        title_style = "font-size: 16px; font-weight: bold; color: #2c3e50;"
        label_width = 150  # Width for labels
        input_width = 180  # Width for inputs

        # Set up the main frame with a light background color
        self.setStyleSheet("background-color: #f5f5f7;")

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(15, 15, 15, 15)

        # Parameters Group
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
            "n": "20",
            "m": "10",
        }

        # Parameter descriptions
        param_descriptions = {
            "hidden_size": "Hidden Layer Size",
            "penalty_factor": "Penalty Factor",
            "n": "Parameter n",
            "m": "Parameter m",
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
        params_horizontal_layout.addLayout(left_column, 1)
        params_horizontal_layout.addLayout(right_column, 1)

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
        self.canvas = FDCNNLossCanvas(self, width=4, height=3)
        plot_layout.addWidget(self.canvas)

        main_layout.addWidget(plot_group)

        # Control panel with multiple buttons
        control_frame = QFrame()
        control_frame.setStyleSheet(
            "background-color: white; border-radius: 6px; border: 1px solid #dcdde1;"
        )
        control_layout = QHBoxLayout(control_frame)
        control_layout.setContentsMargins(10, 6, 10, 6)

        # Generate Sample Button
        self.generate_btn = QPushButton("生成样本")
        self.generate_btn.setStyleSheet("""
            QPushButton {
                font-size: 12px;
                font-weight: bold;
                background-color: #3498db;
                color: white;
                padding: 4px 8px;
                border-radius: 3px;
                min-height: 28px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        self.generate_btn.clicked.connect(self.generate_sample)

        # Status label
        self.status_label = QLabel("Status: Ready")
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
        control_layout.addWidget(self.generate_btn)
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
        self.predict_btn.setEnabled(True)

    def upload_data(self):
        """Handle data upload"""
        try:
            options = QFileDialog.Options()
            file_name, _ = QFileDialog.getOpenFileName(
                self,
                "Upload Data File",
                "",
                "Data Files (*.csv *.xlsx *.txt);;All Files (*)",
                options=options,
            )

            if file_name:
                # Here you would actually load and process the file
                # For now, just update the status
                self.status_label.setText(
                    f"Status: Data uploaded: {os.path.basename(file_name)}"
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

                self.data_status.setText(
                    f"Data Status: Loaded {os.path.basename(file_name)}"
                )
                self.data_status.setStyleSheet("""
                    font-size: 12px;
                    padding: 4px 8px;
                    background-color: #d4edda;
                    color: #155724;
                    border-radius: 3px;
                """)

                self.data_uploaded = True
                self.train_btn.setEnabled(True)

                QMessageBox.information(
                    self,
                    "Upload Successful",
                    f"Data file loaded successfully: \n{file_name}",
                )
        except Exception as e:
            self.status_label.setText("Status: Upload failed ❌")
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
                self, "Upload Error", f"Error uploading data: {str(e)}"
            )

    def generate_sample(self):
        """Generate sample data for demonstration"""
        try:
            self.status_label.setText("Status: Generating sample data...")
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

            # Create a fake dataset
            # In a real application, this would involve actual data generation

            self.data_status.setText("Data Status: Sample data generated")
            self.data_status.setStyleSheet("""
                font-size: 12px;
                padding: 4px 8px;
                background-color: #d4edda;
                color: #155724;
                border-radius: 3px;
            """)

            # Plot initial fake loss curve
            iterations = np.arange(50)
            initial_loss = np.linspace(5, 1, 50) + np.random.normal(0, 0.2, 50)
            self.canvas.plot_loss_curve(iterations, initial_loss)

            self.status_label.setText("Status: Sample generated ✅")
            self.status_label.setStyleSheet("""
                font-size: 12px;
                padding: 4px;
                min-height: 24px;
                background-color: #d4edda;
                color: #155724;
                border-radius: 3px;
                margin: 0 10px;
            """)

            self.fake_data_generated = True
            self.train_btn.setEnabled(True)
            self.save_plot_btn.setEnabled(True)

            QMessageBox.information(
                self, "Sample Generated", "Sample data has been generated successfully."
            )
        except Exception as e:
            self.status_label.setText("Status: Sample generation failed ❌")
            self.status_label.setStyleSheet("""
                font-size: 12px;
                padding: 4px;
                min-height: 24px;
                background-color: #f8d7da;
                color: #721c24;
                border-radius: 3px;
                margin: 0 10px;
            """)
            QMessageBox.critical(self, "Error", f"Failed to generate sample: {str(e)}")

    def train_network(self):
        """Execute FDCNN network training"""
        if not (self.data_uploaded or self.fake_data_generated):
            QMessageBox.warning(
                self, "No Data", "Please upload data or generate a sample first."
            )
            return

        try:
            # Update status
            self.status_label.setText("Status: Training network...")
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

            # Get parameter values
            hidden_size = int(self.inputs["hidden_size"].text())
            penalty_factor = int(self.inputs["penalty_factor"].text())
            n = int(self.inputs["n"].text())
            m = int(self.inputs["m"].text())
            has_constraints = self.yes_radio.isChecked()

            # Run simulation - this simulates the actual network training
            iterations, loss_values = simulate_fdcnn_training(
                hidden_size, penalty_factor, n, m, has_constraints
            )

            # Store results
            self.training_results = {
                "iterations": iterations,
                "loss_values": loss_values,
                "parameters": {
                    "hidden_size": hidden_size,
                    "penalty_factor": penalty_factor,
                    "n": n,
                    "m": m,
                    "has_constraints": has_constraints,
                },
            }

            # Plot the results
            self.canvas.plot_loss_curve(iterations, loss_values)

            # Update status
            self.status_label.setText("Status: Training completed ✅")
            self.status_label.setStyleSheet("""
                font-size: 12px;
                padding: 4px;
                min-height: 24px;
                background-color: #d4edda;
                color: #155724;
                border-radius: 3px;
                margin: 0 10px;
            """)

            # Enable export and save buttons
            self.save_plot_btn.setEnabled(True)
            self.export_btn.setEnabled(True)

            QMessageBox.information(
                self,
                "Training Complete",
                "FDCNN network training completed successfully!",
            )

        except ValueError as e:
            self.status_label.setText("Status: Training error ❌")
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
                self, "Input Error", f"Please check that all inputs are valid: {str(e)}"
            )

        except Exception as e:
            self.status_label.setText("Status: Training failed ❌")
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
                self, "Error", f"An error occurred during training: {str(e)}"
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

    def export_data(self):
        if self.training_results is None:
            QMessageBox.warning(self, "No Data", "Please train the network first.")
            return

        try:
            # Ask user where to save the file
            options = QFileDialog.Options()
            file_name, _ = QFileDialog.getSaveFileName(
                self,
                "Save Excel File",
                "",
                "Excel Files (*.xlsx);;CSV Files (*.csv);;All Files (*)",
                options=options,
            )

            if not file_name:
                return

            # Add appropriate extension if not present
            if not (file_name.endswith(".xlsx") or file_name.endswith(".csv")):
                file_name += ".xlsx"

            # Prepare data for export
            data = {
                "Iteration": self.training_results["iterations"],
                "Loss": self.training_results["loss_values"],
            }

            df = pd.DataFrame(data)

            # Add parameter information to a separate sheet
            params_data = {
                "Parameter": [
                    "Hidden Layer Size",
                    "Penalty Factor",
                    "Parameter n",
                    "Parameter m",
                    "Has Constraints",
                ],
                "Value": [
                    self.training_results["parameters"]["hidden_size"],
                    self.training_results["parameters"]["penalty_factor"],
                    self.training_results["parameters"]["n"],
                    self.training_results["parameters"]["m"],
                    "Yes"
                    if self.training_results["parameters"]["has_constraints"]
                    else "No",
                ],
            }

            params_df = pd.DataFrame(params_data)

            # Export to file
            if file_name.endswith(".xlsx"):
                with pd.ExcelWriter(file_name, engine="openpyxl") as writer:
                    df.to_excel(writer, sheet_name="Training Loss", index=False)
                    params_df.to_excel(writer, sheet_name="Parameters", index=False)
            else:  # CSV format
                df.to_csv(file_name, index=False)
                # For CSV, we can't include multiple sheets, so we'll note this in the message
                QMessageBox.information(
                    self,
                    "CSV Export",
                    "Note: Parameter information is not included in CSV format.",
                )

            # Update status
            self.status_label.setText(
                f"Status: Data exported to {os.path.basename(file_name)} ✅"
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
                self, "Export Complete", f"Data successfully saved to:\n{file_name}"
            )

        except Exception as e:
            self.status_label.setText("Status: Export failed ❌")
            self.status_label.setStyleSheet("""
                font-size: 12px;
                padding: 4px;
                min-height: 24px;
                background-color: #f8d7da;
                color: #721c24;
                border-radius: 3px;
                margin: 0 10px;
            """)
            QMessageBox.critical(self, "Export Error", f"Error saving data: {str(e)}")

    def predict_fracture_width(self):
        """Predict fracture width based on input parameters"""
        try:
            # Get input values
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

            # Simple mock calculation for demonstration
            # In a real application, this would use the trained FDCNN model
            mock_result = (
                well_depth * 0.001
                + loss_rate * 0.05
                + loss_volume * 0.02
                + plastic_viscosity * 0.1
                + drilling_speed * 0.03
                + mud_flow * 0.04
                + pump_pressure * 0.001
            )

            # Format to 4 decimal places
            result = f"{mock_result:.4f}"

            # Display the result
            self.fracture_width_output.setText(result)
            self.fracture_width_output.setStyleSheet(
                "font-size: 14px; padding: 5px; border: 1px solid #27ae60; color: #155724;"
            )

            self.status_label.setText("Status: Prediction completed successfully")
            self.status_label.setStyleSheet("""
                font-size: 12px;
                padding: 4px;
                min-height: 24px;
                background-color: #d4edda;
                color: #155724;
                border-radius: 3px;
                margin: 0 10px;
            """)

        except Exception as e:
            self.status_label.setText(f"Status: Prediction failed - {str(e)}")
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
                self, "Prediction Error", f"Error during prediction: {str(e)}"
            )
