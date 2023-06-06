import sys
from PyQt5.QtGui import QPainter, QColor, QPen, QFont, QImage, QIcon, QPixmap, QFontMetrics, QMovie
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QWidget, QLabel, QGraphicsOpacityEffect
from PyQt5.QtCore import QTimer, Qt, QPoint, QRect, QSize
from get_time import get_time
from dataclasses import dataclass

screen = [1920, 1080]


class TextUI:
    def __init__(self, x, y, windows, text="None", disable=True, font_size=20, color="(0, 0, 0, 1)"):
        self.x = x
        self.y = y
        self.text = text
        self.font_size = font_size
        self.disable = disable
        self.windows = windows
        self.color = color

    def initUI(self):
        # create image
        self.label = QLabel(self.windows)
        # self.label.setStyleSheet("background: rgba(0,0,0,0);")  # set transparent
        self.label.setStyleSheet("background: rgba(0,0,0,0); color:rgba" + self.color + ";")

        self.label.setFont(QFont('Arial', self.font_size))
        self.label.setText(self.text)
        self.move(int(self.x), int(self.y))
        # self.label.setAlignment(Qt.AlignCenter)
        # x, y, w, h = self.get_xywh_of_screen_align_center()
        # self.label.setGeometry(x, y, w, h)

        if self.disable:
            self.set_un_visible()

    def set_txt(self, txt: str):
        self.text = txt
        self.label.setText(self.text)
        self.move(int(self.x), int(self.y))

    def get_txt(self):
        return self.text

    def set_visible(self):
        self.label.setVisible(True)

    def set_un_visible(self):
        self.label.setVisible(False)

    def move(self, x, y):
        font = self.label.font()
        fm = QFontMetrics(font)
        self.label.setGeometry(int(x), int(y), int(fm.width(self.text)), int(fm.height()))
        # print(fm.width(self.text))

    def get_xywh_of_screen_align_center(self):  # get center of screen
        font = self.label.font()
        fm = QFontMetrics(font)
        w = fm.width(self.text)
        h = fm.height()
        return [(screen[0] - w) // 2, (screen[1] - h) // 2, w, h]


class TimerText:
    def __init__(self, x, y, disable=True, font_size=16):
        self.x = x
        self.y = y
        self.font_size = font_size
        self.disable = disable

    def set_visible(self):
        self.disable = True

    def set_un_visible(self):
        self.disable = False

    def paint(self, painter):
        if not self.disable:
            # draw txt on acupuncture
            painter.setPen(QColor(255, 255, 255))
            painter.setFont(QFont('Arial', self.font_size))
            h, w, ampm = get_time()
            painter.drawText(QPoint(self.x, self.y), h)
            painter.setFont(QFont('Arial', self.font_size - 3))
            painter.drawText(QPoint(self.x + 38, self.y - 3), ":")
            painter.setFont(QFont('Arial', self.font_size))
            painter.drawText(QPoint(self.x + 50, self.y), w)
            painter.setFont(QFont('Arial', self.font_size - 5))
            txt = ampm.lower()
            painter.drawText(QPoint(self.x + 85, self.y - 2), txt)


class ImageGIF:
    def __init__(self, x, y, w, h, image_path, windows, opacity_level=1.0, disable=True):
        self.x = x
        self.y = y
        self.image_path = image_path
        self.w = w
        self.h = h
        self.opacity_level = opacity_level
        self.disable = disable
        self.windows = windows

    def initUI(self):
        # create image
        self.back_image = QLabel(self.windows)
        self.back_image.setGeometry(int(self.x), int(self.y), int(self.w), int(self.h))

        movie = QMovie(self.image_path)
        movie.setSpeed(100)
        movie.start()
        # Set the movie to the label
        self.back_image.setMovie(movie)
        self.back_image.setStyleSheet("background: rgba(0,0,0,0);")
        self.back_image.setScaledContents(True)

        opacity_effect = QGraphicsOpacityEffect()
        opacity_effect.setOpacity(self.opacity_level)
        self.back_image.setGraphicsEffect(opacity_effect)

        if self.disable:
            self.set_un_visible()

    def set_visible(self):
        self.back_image.setVisible(True)

    def set_un_visible(self):
        self.back_image.setVisible(False)

    def move(self, x, y):
        self.x = x
        self.y = y
        self.back_image.setGeometry(self.x, self.y, self.w, self.h)


class ImageUI:
    def __init__(self, x, y, w, h, image_path, windows, opacity_level=1.0, disable=True):
        self.x = x
        self.y = y
        self.image_path = image_path
        self.w = w
        self.h = h
        self.opacity_level = opacity_level
        self.disable = disable
        self.windows = windows

    def initUI(self):
        # create image
        self.back_image = QLabel(self.windows)
        self.back_image.setGeometry(int(self.x), int(self.y), int(self.w), int(self.h))
        self.back_image.setStyleSheet("background: rgba(0,0,0,0);")
        image = QImage(self.image_path).scaledToHeight(int(self.h))
        # Create an opacity effect object
        opacity_effect = QGraphicsOpacityEffect()
        opacity_effect.setOpacity(self.opacity_level)
        self.back_image.setGraphicsEffect(opacity_effect)
        self.back_image.setPixmap(QPixmap.fromImage(image))

        if self.disable:
            self.set_un_visible()

    def set_visible(self):
        self.back_image.setVisible(True)

    def set_un_visible(self):
        self.back_image.setVisible(False)

    def move(self, x, y):
        self.x = x
        self.y = y
        self.back_image.setGeometry(self.x, self.y, self.w, self.h)

    def reset_img(self, img_path):
        image = QImage(img_path).scaledToHeight(int(self.h))
        if image:
            self.back_image.setPixmap(QPixmap.fromImage(image))


class PlayableImageUI:
    def __init__(self, x, y, w, h, image_path, b_index, n_index, windows, opacity_level=1.0, disable=True):
        self.x = x
        self.y = y
        self.image_path = image_path
        self.w = w
        self.h = h
        self.b_index = b_index
        self.n_index = n_index
        self.opacity_level = opacity_level
        self.disable = disable
        self.windows = windows

    def initUI(self):
        # create image
        self.back_image = QLabel(self.windows)
        self.back_image.setGeometry(int(self.x), int(self.y), int(self.w), int(self.h))
        self.back_image.setStyleSheet("background: rgba(0,0,0,0);")
        image = QImage(self.image_path).scaledToHeight(int(self.h))
        # Create an opacity effect object
        opacity_effect = QGraphicsOpacityEffect()
        opacity_effect.setOpacity(self.opacity_level)
        self.back_image.setGraphicsEffect(opacity_effect)
        self.back_image.setPixmap(QPixmap.fromImage(image))

        if self.disable:
            self.set_un_visible()

    def set_visible(self):
        self.back_image.setVisible(True)

    def set_un_visible(self):
        self.back_image.setVisible(False)

    def get_back_image_index(self):
        return self.b_index

    def get_next_image_index(self):
        return self.n_index

    def move(self, x, y):
        self.x = x
        self.y = y
        self.back_image.setGeometry(self.x, self.y, self.w, self.h)


class ImageWidget(QWidget):
    def __init__(self, ui_buttons):
        super().__init__()
        self.ui_buttons = ui_buttons

    def paintEvent(self, event):
        painter = QPainter(self)
        for ui_button in self.ui_buttons:
            ui_button.paint(painter)


class DotMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.ui_images = [
            ImageUI(screen[0] // 3.5, screen[1] // 1.8, screen[0] // 24, screen[0] // 24, 'menu.png', self,
                    disable=True),
            ImageUI(-(screen[0] // 4), screen[1] // 22, 1280 * 2, 64 * 2, 'border.png', self, disable=False,
                    opacity_level=0.5)]
        self.ui_txts = [TextUI(screen[0] // 3, screen[1] // 11, self, text="all", disable=False),
                        TextUI(screen[0] // 1.1, screen[1] // 11, self, text="無手勢", disable=False, )]
        self.image_widget = ImageWidget()
        self.init_ui_before_widget()  # layer01

        self.setCentralWidget(self.image_widget)  # layer02  # Update every frame
        self.init_ui_after_widget()  # layer03
        # self.ui_txts[1].set_txt("取消動作")
        # self.ui_txts[1].set_txt("無手勢")

    def init_ui_after_widget(self):
        for ui_txt in self.ui_txts:
            ui_txt.initUI()
        for ui_button in self.ui_buttons:
            ui_button.set_image_widget(self.image_widget)
            ui_button.initUI()  # set UI in front
        # self.ui_button[0].move(1000, 200)  # move UI button

    def init_ui_before_widget(self):
        for ui_image in self.ui_images:
            ui_image.initUI()  # set UI in front


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DotMainWindow()
    window.showFullScreen()  # Show the widget
    screen.append(window.width())
    screen.append(window.height())
    window.setStyleSheet("background-color: grey;")
    sys.exit(app.exec_())
