from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QStyledItemDelegate, QProxyStyle


class CenterDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index):
        option.displayAlignment = Qt.AlignCenter
        super().paint(painter, option, index)


class CenteredComboBoxStyle(QProxyStyle):
    """Custom ProxyStyle to center QComboBox text"""

    def drawItemText(self, painter, rect, flags, palette, enabled, text, textRole):
        flags |= Qt.AlignCenter
        super().drawItemText(painter, rect, flags, palette, enabled, text, textRole)
