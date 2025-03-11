import sys
import os
import traceback

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QPushButton,
    QVBoxLayout,
    QLabel,
    QStackedWidget,
    QHBoxLayout,
    QGroupBox,
    QSizePolicy,
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

from pages.login_page import LoginWindow
from pages.config import resource_path, clean_img_folder
from pages.data_processing_page import DataProcessingPage
from pages.crack_identification_page import CrackIdentificationPage
from pages.parameter_calculation_page import ParameterCalculationPage
from pages.parameter_analysis_page import ParameterAnalysis
from pages.comprehensive_outcome_page import ComprehensiveOutcomePage
from pages.distribution_page import DistributionPage
from pages.hydrodynamic_page import HydrodynamicPage
from pages.bp_page import BPPage
from pages.FDCNN_page import FDCNNPage

os.environ["QT_MAC_WANTS_LAYER"] = "1"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("裂缝通道评价系统")
        self.setGeometry(100, 100, 1200, 800)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.layout.setSpacing(10)

        self.nav_buttons = QWidget()
        self.nav_layout = QHBoxLayout(self.nav_buttons)
        self.nav_layout.setSpacing(5)
        self.nav_layout.setContentsMargins(5, 5, 5, 5)

        self.pages = QStackedWidget()
        self.pages.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.pages_map = {}

        # Home Page
        home_page = QLabel()
        pixmap = QPixmap(resource_path("data/homepage.png"))
        home_page.setPixmap(pixmap)
        home_page.setAlignment(Qt.AlignCenter)
        home_page.setScaledContents(False)
        home_page.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.add_page("主页", home_page)

        # Data Processing Page
        self.data_processing_page = DataProcessingPage()
        self.add_page("数据预处理", self.data_processing_page)

        # Logging Evaluation Section
        logging_group = QGroupBox("测井评价")
        logging_layout = QVBoxLayout()
        logging_layout.setSpacing(5)
        logging_buttons_layout = QHBoxLayout()
        self.add_group_page(
            logging_buttons_layout, "裂缝识别", CrackIdentificationPage()
        )

        self.add_group_page(
            logging_buttons_layout, "裂缝通道参数计算", ParameterCalculationPage()
        )

        # Add new page for crack analysis
        self.add_group_page(logging_buttons_layout, "分析", ParameterAnalysis())
        logging_layout.addLayout(logging_buttons_layout)
        logging_group.setLayout(logging_layout)

        # Mud Logging Evaluation Section
        mud_logging_group = QGroupBox("录井评价")
        mud_logging_layout = QVBoxLayout()
        mud_logging_layout.setSpacing(5)
        mud_logging_buttons_layout1 = QHBoxLayout()
        mud_logging_buttons_layout2 = QHBoxLayout()
        self.add_group_page(
            mud_logging_buttons_layout1, "流体力学模型", HydrodynamicPage()
        )
        self.add_group_page(mud_logging_buttons_layout1, "BP神经网络模型", BPPage())
        self.add_group_page(mud_logging_buttons_layout2, "FDCNN模型", FDCNNPage())
        self.add_group_page(
            mud_logging_buttons_layout2, "录井数据分布", DistributionPage()
        )
        mud_logging_layout.addLayout(mud_logging_buttons_layout1)
        mud_logging_layout.addLayout(mud_logging_buttons_layout2)
        mud_logging_group.setLayout(mud_logging_layout)

        self.comprehensive_outcome_page = ComprehensiveOutcomePage()
        self.add_page("测录井综合图", self.comprehensive_outcome_page)

        self.layout.addWidget(self.nav_buttons)
        self.layout.addWidget(logging_group)
        self.layout.addWidget(mud_logging_group)
        self.layout.addWidget(self.pages)

    def add_page(self, name, widget):
        btn = QPushButton(name)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setMinimumHeight(30)
        btn.setStyleSheet("""
            QPushButton {
                background-color: #0056b3;
                color: white;
                padding: 3px;
                border-radius: 3px;
                border: 1px solid transparent;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #007BFF;
                border: 1px solid white;
            }
            QPushButton:pressed {
                background-color: #003d80;
            }
        """)
        btn.clicked.connect(lambda: self.pages.setCurrentWidget(self.pages_map[name]))
        self.nav_layout.addWidget(btn)

        self.pages_map[name] = widget
        self.pages.addWidget(widget)

    def add_group_page(self, layout, name, widget):
        btn = QPushButton(name)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setMinimumHeight(30)
        btn.setStyleSheet("""
            QPushButton {
                background-color: #006400;
                color: white;
                padding: 3px;
                border-radius: 3px;
                border: 1px solid transparent;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #008000;
                border: 1px solid white;
            }
            QPushButton:pressed {
                background-color: #004d00;
            }
        """)
        btn.clicked.connect(lambda: self.pages.setCurrentWidget(self.pages_map[name]))
        layout.addWidget(btn)

        self.pages_map[name] = widget
        self.pages.addWidget(widget)

    def resizeEvent(self, event):
        super().resizeEvent(event)


def excepthook(exc_type, exc_value, exc_traceback):
    """show error after packaged"""
    with open("error.log", "w") as f:
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=f)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    login_window = LoginWindow(main_window)
    login_window.show()
    app.aboutToQuit.connect(clean_img_folder)
    sys.excepthook = excepthook
    sys.exit(app.exec_())
