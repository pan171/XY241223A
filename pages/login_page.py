from PyQt5.QtWidgets import (
    QLineEdit,
    QWidget,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QMessageBox,
    QGraphicsBlurEffect,
)
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtCore import Qt
from pages.config import resource_path

user_data = {"test": "123"}


class LoginWindow(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.initUI()

    def initUI(self):
        self.setWindowTitle("裂缝通道评价系统")
        self.resize(900, 600)

        # 使用 QLabel 设置背景图片，并使其足够大（按比例填充整个窗口）
        self.bg_label = QLabel(self)
        self.bg_label.setGeometry(0, 0, self.width(), self.height())
        pixmap = QPixmap(resource_path("data/login.png")).scaled(
            self.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation
        )
        self.bg_label.setPixmap(pixmap)
        self.bg_label.setAlignment(Qt.AlignCenter)
        self.bg_label.lower()  # 将背景标签放到最底层

        # 添加虚化效果
        blur_effect = QGraphicsBlurEffect()
        blur_effect.setBlurRadius(8)  # 调整虚化半径以获得理想效果
        self.bg_label.setGraphicsEffect(blur_effect)

        # 原有布局代码保持不变（界面控件会显示在背景标签之上）
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setAlignment(Qt.AlignCenter)

        title = QLabel("裂缝通道评价系统", self)
        title.setFont(QFont("Arial", 20))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: white;")

        self.user_input = QLineEdit()
        self.user_input.setPlaceholderText("用户名")
        self.pwd_input = QLineEdit()
        self.pwd_input.setPlaceholderText("密码")
        self.pwd_input.setEchoMode(QLineEdit.Password)

        self.user_input.setMinimumHeight(30)
        self.pwd_input.setMinimumHeight(30)

        login_btn = QPushButton("进入")
        login_btn.setStyleSheet(
            "background-color: #007BFF; color: white; padding: 5px; border-radius: 5px;"
        )
        login_btn.clicked.connect(self.open_main)

        register_btn = QPushButton("注册")
        register_btn.setStyleSheet(
            "background-color: #28A745; color: white; padding: 5px; border-radius: 5px;"
        )
        register_btn.clicked.connect(self.open_register)

        # 添加游客访问按钮
        guest_btn = QPushButton("游客访问")
        guest_btn.setStyleSheet(
            "background-color: #6C757D; color: white; padding: 5px; border-radius: 5px;"
        )
        guest_btn.clicked.connect(self.guest_access)

        # Use a percentage of width instead of fixed width
        input_width = min(400, self.width() // 3)

        # Create a container widget for the form elements to center them properly
        form_container = QWidget()
        form_layout = QVBoxLayout(form_container)
        form_layout.setAlignment(Qt.AlignCenter)

        self.user_input.setMinimumWidth(input_width)
        self.pwd_input.setMinimumWidth(input_width)
        login_btn.setMinimumWidth(input_width)
        register_btn.setMinimumWidth(input_width)
        guest_btn.setMinimumWidth(input_width)  # 设置游客按钮宽度

        # Maximum width to maintain good readability on large screens
        self.user_input.setMaximumWidth(400)
        self.pwd_input.setMaximumWidth(400)
        login_btn.setMaximumWidth(400)
        register_btn.setMaximumWidth(400)
        guest_btn.setMaximumWidth(400)  # 设置游客按钮最大宽度

        form_layout.addWidget(title)
        form_layout.addWidget(self.user_input)
        form_layout.addWidget(self.pwd_input)
        form_layout.addWidget(login_btn)
        form_layout.addWidget(register_btn)
        form_layout.addWidget(guest_btn)  # 添加游客按钮到布局

        layout.addWidget(form_container)
        self.setLayout(layout)

    def resizeEvent(self, event):
        # Update background image when window is resized
        self.bg_label.setGeometry(0, 0, self.width(), self.height())
        pixmap = QPixmap(resource_path("data/login.png")).scaled(
            self.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation
        )
        self.bg_label.setPixmap(pixmap)
        super().resizeEvent(event)

    def open_main(self):
        username = self.user_input.text()
        password = self.pwd_input.text()

        if username in user_data and user_data[username] == password:
            self.main_window.showMaximized()  # Use maximized instead of fullscreen for better compatibility
            self.close()
        else:
            QMessageBox.warning(self, "登录失败", "用户名或密码错误")

    def guest_access(self):
        # 游客直接进入主页面，无需验证
        self.main_window.showMaximized()
        self.close()

    def open_register(self):
        self.register_window = RegisterWindow()
        self.register_window.show()


class RegisterWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("裂缝通道评价系统")
        self.resize(900, 600)

        # 添加背景图片
        self.bg_label = QLabel(self)
        self.bg_label.setGeometry(0, 0, self.width(), self.height())
        pixmap = QPixmap(resource_path("data/login.png")).scaled(
            self.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation
        )
        self.bg_label.setPixmap(pixmap)
        self.bg_label.setAlignment(Qt.AlignCenter)
        self.bg_label.lower()  # 将背景标签放到最底层

        blur_effect = QGraphicsBlurEffect()
        blur_effect.setBlurRadius(8)
        self.bg_label.setGraphicsEffect(blur_effect)

        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setAlignment(Qt.AlignCenter)

        # Create a container widget for the form elements
        form_container = QWidget()
        form_layout = QVBoxLayout(form_container)
        form_layout.setAlignment(Qt.AlignCenter)

        title = QLabel("用户注册", self)
        title.setFont(QFont("Arial", 20))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: white;")

        self.new_user_input = QLineEdit()
        self.new_user_input.setPlaceholderText("输入用户名")

        self.new_pwd_input = QLineEdit()
        self.new_pwd_input.setPlaceholderText("输入密码")
        self.new_pwd_input.setEchoMode(QLineEdit.Password)

        self.confirm_pwd_input = QLineEdit()
        self.confirm_pwd_input.setPlaceholderText("确认密码")
        self.confirm_pwd_input.setEchoMode(QLineEdit.Password)

        self.new_user_input.setMinimumHeight(30)
        self.new_pwd_input.setMinimumHeight(30)
        self.confirm_pwd_input.setMinimumHeight(30)

        # Use percentage of width instead of fixed width
        input_width = min(400, self.width() // 3)

        self.new_user_input.setMinimumWidth(input_width)
        self.new_pwd_input.setMinimumWidth(input_width)
        self.confirm_pwd_input.setMinimumWidth(input_width)

        # Maximum width for good readability
        self.new_user_input.setMaximumWidth(400)
        self.new_pwd_input.setMaximumWidth(400)
        self.confirm_pwd_input.setMaximumWidth(400)

        register_btn = QPushButton("注册")
        register_btn.setStyleSheet(
            "background-color: #28A745; color: white; padding: 5px; border-radius: 5px;"
        )
        register_btn.clicked.connect(self.register_user)

        form_layout.addWidget(title)
        form_layout.addWidget(self.new_user_input)
        form_layout.addWidget(self.new_pwd_input)
        form_layout.addWidget(self.confirm_pwd_input)
        form_layout.addWidget(register_btn)

        layout.addWidget(form_container)
        self.setLayout(layout)

    def register_user(self):
        username = self.new_user_input.text()
        password = self.new_pwd_input.text()
        confirm_password = self.confirm_pwd_input.text()

        if not username or not password:
            QMessageBox.warning(self, "注册失败", "用户名和密码不能为空")
            return

        if password != confirm_password:
            QMessageBox.warning(self, "注册失败", "两次输入的密码不一致")
            return

        if username in user_data:
            QMessageBox.warning(self, "注册失败", "用户名已存在")
            return

        user_data[username] = password
        QMessageBox.information(self, "注册成功", "注册成功，请返回登录")
        self.close()
