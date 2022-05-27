import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import calculator

import io
from PIL import Image
from PyQt5.QtGui import QImage
from PyQt5.QtCore import QBuffer


class MyApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.image = QImage(QSize(332, 213), QImage.Format_RGB32)
        self.image.fill(Qt.white)
        self.drawing = False
        self.brush_size = 5
        self.brush_color = Qt.black
        self.last_point = QPoint()
        self.initUI()

    def initUI(self):
        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)
        filemenu = menubar.addMenu('File')

        save_action = QAction('calc', self)
        save_action.setShortcut('Ctrl+S')
        save_action.triggered.connect(self.calc)

        clear_action = QAction('Clear', self)
        clear_action.setShortcut('Ctrl+C')
        clear_action.triggered.connect(self.clear)

        exit_action = QAction('Exit', self)
        exit_action.setShortcut('Esc')
        exit_action.triggered.connect(self.exit)

        filemenu.addAction(save_action)
        filemenu.addAction(clear_action)
        filemenu.addAction(exit_action)

        self.setWindowTitle('Simple Painter')
        self.setGeometry(300, 300, 332, 213)
        self.show()

    def paintEvent(self, e):
        canvas = QPainter(self)
        canvas.drawImage(self.rect(), self.image, self.image.rect())

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = e.pos()

    def mouseMoveEvent(self, e):
        if (e.buttons() & Qt.LeftButton) & self.drawing:
            painter = QPainter(self.image)
            painter.setPen(QPen(self.brush_color, self.brush_size, Qt.SolidLine, Qt.RoundCap))
            painter.drawLine(self.last_point, e.pos())
            self.last_point = e.pos()
            self.update()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.drawing = False

    def calc(self):
        img = self.image
        buffer = QBuffer()
        buffer.open(QBuffer.ReadWrite)
        img.save(buffer, "PNG")
        pil_img = Image.open(io.BytesIO(buffer.data()))
        for i in range(0, pil_img.size[0]):
            for j in range(0, pil_img.size[1]):
                rgb = pil_img.getpixel((i,j))
                rgb_r = (255-rgb[0], 255-rgb[1], 255-rgb[2])
                pil_img.putpixel((i,j), rgb_r)
        calculator.main(pil_img)

    def clear(self):
        self.image.fill(Qt.white)
        self.update()

    def exit(self):
        exit()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())