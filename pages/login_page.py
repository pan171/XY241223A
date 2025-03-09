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
        self.setFixedSize(900, 600)

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

        self.user_input.setFixedHeight(30)
        self.pwd_input.setFixedHeight(30)

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

        self.user_input.setFixedWidth(self.width() // 3)
        self.pwd_input.setFixedWidth(self.width() // 3)
        login_btn.setFixedWidth(self.width() // 3)
        register_btn.setFixedWidth(self.width() // 3)

        layout.addWidget(title)
        layout.addWidget(self.user_input)
        layout.addWidget(self.pwd_input)
        layout.addWidget(login_btn)
        layout.addWidget(register_btn)

        self.setLayout(layout)

    def open_main(self):
        username = self.user_input.text()
        password = self.pwd_input.text()

        if username in user_data and user_data[username] == password:
            self.main_window.showFullScreen()
            self.close()
        else:
            QMessageBox.warning(self, "登录失败", "用户名或密码错误")

    def open_register(self):
        self.register_window = RegisterWindow()
        self.register_window.show()


class RegisterWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("裂缝通道评价系统")
        self.setFixedSize(900, 600)

        # 在 RegisterWindow.initUI() 的开头（在设置布局之前）添加以下代码：
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

        self.new_user_input.setFixedHeight(30)
        self.new_pwd_input.setFixedHeight(30)
        self.confirm_pwd_input.setFixedHeight(30)

        register_btn = QPushButton("注册")
        register_btn.setStyleSheet(
            "background-color: #28A745; color: white; padding: 5px; border-radius: 5px;"
        )
        register_btn.clicked.connect(self.register_user)

        self.new_user_input.setFixedWidth(self.width() // 3)
        self.new_pwd_input.setFixedWidth(self.width() // 3)
        self.confirm_pwd_input.setFixedWidth(self.width() // 3)
        register_btn.setFixedWidth(self.width() // 3)

        layout.addWidget(title)
        layout.addWidget(self.new_user_input)
        layout.addWidget(self.new_pwd_input)
        layout.addWidget(self.confirm_pwd_input)
        layout.addWidget(register_btn)

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
