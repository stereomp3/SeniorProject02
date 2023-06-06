from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget
from PyQt5.QtGui import QImage, QPainter, QPen, QFont, QColor, QTextOption, QStaticText, QPixmap
from PyQt5.QtCore import Qt, QPoint, QThread, pyqtSignal, QMutex, QPointF
import sys
import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass
from DB.DBconnector import get_sql_data, get_acupuncture_point_data  # mark as source root
from UI.UI import TextUI, ImageUI, PlayableImageUI, TimerText, ImageGIF
from models.model import get_predictions
import models.FaceModel as FM
from BlueTooth_Qrcode import QRcodeReader, BlueToothConnection
from UI.UI_util import pre_txt2real_txt, SetGestureImageOn, playNextImage, menu_visible, menu_un_visible, \
    loading_visible, loading_un_visible, classifyArea_visible, classifyArea_un_visible, update_acupuncture_txt, \
    close_acupuncture_txt
import time_profile

# menu 重新訓練 手放上面一點
data = {}
# screen = [1920, 1080]
# screen = [1280, 1024]
screen = [1024, 768]
dots = []
screen_x_offset = 0
max_point = 42
types = {}
gesture_time_frame = 30  # 10

playable_index = {"introduction": 0, "menu": 7, "acupuncture": 11}  # introduction、menu
image_dict = {"phone": 6, "back": 7, "menu": 8, "Qrcode": 9, "Blutoothsuccess": 10, "Blutoothfail": 11,
              "classifyArea": 12}
gif_dict = {"loading": 0}
txt_dict = {"classify_txt": 0, "loading": 1, "acupuncture_title": 2, "line1": 3, "line2": 4, "line3": 5, "line4": 6,
            "line5": 7, "line6": 8, "line7": 9, "line8": 9}  # line5 ignore in first two


@dataclass
class Acupuncture:
    name: str
    rel_point: int
    offset_x: int
    offset_y: int
    handness: int
    types: list  # the disease types of acupuncture


class Dot:
    def __init__(self, x, y, acupuncture):
        self.x = x
        self.y = y
        self.disable = True
        self.acupuncture = acupuncture
        self.option = QTextOption()
        txt = self.acupuncture.name
        self.static_text = QStaticText(txt)
        cx = abs(int(self.x - screen[0] * 0.001))
        self.option.setAlignment(Qt.AlignVCenter)
        self.static_text.setTextWidth(cx)
        self.static_text.setTextOption(self.option)

    def paint_circle(self, painter):
        if not self.disable:
            painter.drawEllipse(self.x, self.y, 10, 10)

    def paint_font(self, painter):
        if not self.disable:
            painter.drawStaticText(QPointF(self.x - 10, self.y + 20), self.static_text)

    def move_dot(self, x, y):
        self.x = x
        self.y = y

    def get_types(self):  # clickable object types
        return 0

    def get_acupuncture(self):
        return self.acupuncture

    def set_active(self):
        self.disable = False

    def un_active(self):
        self.disable = True


get_sql_data(data, "New_acupuncture_point")

# dot = Dot(-50, -50)
# dot.set_acupuncture(Acupuncture("測試", 8, 0, 0, 4))
# dots.append(dot)

for k, v in data.items():
    dot = Dot(-50, -50, Acupuncture(k, v[0], v[1], v[2], v[4], v[5]))
    dots.append(dot)
    if types.__contains__(v[4]):
        types[v[4]] += 1
    else:
        types[v[4]] = 1
dots.sort(key=lambda x: x.get_acupuncture().handness)
mutex = QMutex()


class WorkerThread(QThread):
    trigger = pyqtSignal(np.ndarray)
    finished = pyqtSignal()

    def __init__(self, windows):
        super().__init__()
        self.windows = windows
        # bluetooth
        self.bt = BlueToothConnection()
        self.bt.finished.connect(self.bt_is_finish)
        self.bt.send_message.connect(self.bt_get_message)
        self.bt_txt = ""
        self.Qrcode_is_read = False  # for bluetooth Qrcode reader
        self.bt_is_receive = False
        self.main_profile = time_profile.TimeProfile(True)

    def run(self):
        mp_hands = mp.solutions.hands
        mp_face_mesh = mp.solutions.face_mesh
        mp_face_detection = mp.solutions.face_detection
        with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh, \
                mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection, \
                mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
            cap = cv2.VideoCapture(0)
            print("start")
            if not cap.isOpened():
                print("Cannot open camera")
                exit()
            timer = 0
            is_introduction = False
            is_menu = False
            is_acupuncture = False  # show the certain acupuncture
            acupuncture_txt = ""  # show the certain acupuncture
            acupuncture_show = False  # show the acupuncture point
            img_index = 0  # for playable img
            now_img = None  # for img
            ap_filter = None
            acupuncture_part = "all"
            bt_connected = False

            while True:
                self.main_profile.begin("main update")

                self.main_profile.label_begin("read webcam")
                ret, img = cap.read()
                if not ret:
                    print("Cannot receive frame")
                    break
                self.main_profile.label_end()

                self.main_profile.label_begin("parse webcam image")
                # img = cap_resize(img, 256, 100, 512, 384)  # fit the mirror
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 將 BGR 轉換成 RGB
                img3 = cv2.flip(img, 1)
                self.main_profile.label_end()

                self.main_profile.label_begin("set acupoints activeself")
                for dot in dots:
                    dot.un_active()
                self.main_profile.label_end()

                self.main_profile.label_begin("read Qrcode")
                if self.Qrcode_is_read:
                    info = QRcodeReader(img)
                    if info:  # connecting ...
                        if self.bt.isRunning():
                            self.bt.close()
                        self.bt.setValue(info)
                        self.bt.start()
                        self.Qrcode_is_read = False
                        bt_connected = True
                self.main_profile.label_end()

                self.main_profile.label_begin("receive bluetooth")
                if self.bt_is_receive:  # 判斷傳送字符
                    sender, txt = self.bt_txt.split("@@@")
                    if sender == "BlueTooth":  # to control UI element
                        print("BlueTooth:" + txt)
                        if txt == "loading":
                            loading_visible(self.windows.ui_gifs, self.windows.ui_txts, gif_dict["loading"],
                                            txt_dict["loading"])
                        elif txt == "success":
                            loading_un_visible(self.windows.ui_gifs, self.windows.ui_txts, gif_dict["loading"],
                                               txt_dict["loading"])
                            self.windows.ui_images[image_dict["phone"]].set_visible()
                            now_img.set_un_visible()
                            now_img = self.windows.ui_images[image_dict["Blutoothsuccess"]]
                            now_img.set_visible()
                            bt_connected = False
                        elif txt == "fail":
                            loading_un_visible(self.windows.ui_gifs, self.windows.ui_txts, gif_dict["loading"],
                                               txt_dict["loading"])
                            now_img.set_un_visible()
                            now_img = self.windows.ui_images[image_dict["Blutoothfail"]]
                            now_img.set_visible()
                            bt_connected = False

                    elif sender == "acupuncture":  # show the acupuncture info
                        print("acupuncture show:" + txt)
                        acupuncture_txt = txt
                        acupuncture_info = get_acupuncture_point_data(acupuncture_txt, "New_acupuncture_point")
                        if acupuncture_info is not None:
                            img_index = playable_index["acupuncture"]
                            is_acupuncture = True
                            self.windows.ui_playable_images[img_index].set_visible()
                            update_acupuncture_txt(self.windows.acupuncture_img[0], self.windows.ui_txts, txt_dict,
                                                   acupuncture_info, playable_index["acupuncture"], img_index,
                                                   acupuncture_txt)

                    elif sender == "part":  # show the body part of acupuncture
                        print("part:" + txt)
                        if txt == "all":
                            acupuncture_part = "all"
                        elif txt == "hand":
                            acupuncture_part = "hand"
                        elif txt == "head":
                            acupuncture_part = "head"
                        elif txt == "face_mesh":
                            acupuncture_part = "face_mesh"

                    elif sender == "symptom":  # show the certain acupuncture
                        print("symptom:" + txt)
                        acupuncture_show = True
                        ap_filter = acupuncture_list_by_disease(dots, txt)
                        txt = txt + " (" + str(len(ap_filter)) + "個穴位)"
                        # show UI
                        classifyArea_visible(self.windows.ui_images, self.windows.ui_txts, image_dict["classifyArea"],
                                             txt_dict["classify_txt"], txt)

                        if txt == "顯示全部穴位":
                            classifyArea_un_visible(self.windows.ui_images, self.windows.ui_txts,
                                                    image_dict["classifyArea"],
                                                    txt_dict["classify_txt"])
                    self.bt_is_receive = False
                self.main_profile.label_end()

                # head face mesh
                if acupuncture_show and acupuncture_part == "face_mesh":
                    self.main_profile.label_begin("face mesh process")
                    face_results = face_mesh.process(img)
                    self.main_profile.label_end()
                    if face_results.multi_face_landmarks:
                        self.main_profile.label_begin("lock")
                        mutex.lock()
                        self.main_profile.label_end()
                        self.main_profile.label_begin("face acupoints process")
                        for face_landmarks in face_results.multi_face_landmarks:
                            degree, base_length = landmark_to_face_acupunncture(face_landmarks.landmark)
                            # print("face")
                            lower_limit = 0
                            upper_limit = lower_limit + types[0]
                            set_acupuncture_point(face_landmarks.landmark, lower_limit, upper_limit, base_length,
                                                  degree, is_face=True, ap_filter=ap_filter)
                        self.main_profile.label_end()
                        self.main_profile.label_begin("mutex unlock()")
                        mutex.unlock()
                        self.main_profile.label_end()

                # head face box
                if acupuncture_show and (acupuncture_part == "all" or acupuncture_part == "head"):
                    self.main_profile.label_begin("face box process")
                    face_results = face_detection.process(img)
                    self.main_profile.label_end()
                    if face_results.detections:
                        self.main_profile.label_begin("lock")
                        mutex.lock()
                        self.main_profile.label_end()
                        self.main_profile.label_begin("face acupoints process")
                        for detection in face_results.detections:
                            output = FM.get_predictions(detection.location_data.relative_keypoints)
                            set_face_box_ap(output, ap_filter=ap_filter)  # 穴位以固定，順序或其他做更動模型也要動
                        self.main_profile.label_end()
                        self.main_profile.label_begin("mutex unlock()")
                        mutex.unlock()
                        self.main_profile.label_end()

                self.main_profile.label_begin("hand process")
                # hand
                hand_results = hands.process(img3)
                self.main_profile.label_end()
                if hand_results.multi_hand_landmarks:
                    count = 0
                    hand_label = []
                    for handness in hand_results.multi_handedness:
                        for hand in handness.classification:
                            hand_label.append(hand.label)
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        if acupuncture_show and (acupuncture_part == "all" or acupuncture_part == "hand"):
                            self.main_profile.label_begin("lock")
                            mutex.lock()
                            self.main_profile.label_end()

                            self.main_profile.label_begin("hand acupoints process")
                            degree, base_length, v1nv2 = landmark_to_hand_acupunncture(hand_landmarks.landmark)
                            lower_limit, upper_limit = HandRecognition(hand_label[count], v1nv2)
                            set_acupuncture_point(hand_landmarks.landmark, lower_limit, upper_limit, base_length,
                                                  degree, ap_filter=ap_filter)
                            self.main_profile.label_end()

                            self.main_profile.label_begin("mutex unlock()")
                            mutex.unlock()
                            self.main_profile.label_end()

                        if not timer % gesture_time_frame:
                            self.main_profile.label_begin("hand gesture process")
                            timer = 0
                            pre_txt = get_predictions(hand_landmarks.landmark)
                            real_txt = pre_txt2real_txt(pre_txt)
                            real_txt = pre_txt
                            # self.windows.ui_txts[0].set_txt(real_txt)

                            if real_txt == "menu":
                                if not is_introduction and not self.Qrcode_is_read and not is_menu and not bt_connected and not is_acupuncture:  # call menu
                                    self.windows.ui_playable_images[img_index].set_un_visible()
                                    is_menu = True
                                    img_index = playable_index["menu"]
                                    menu_visible(self.windows.ui_images, self.windows.ui_playable_images, image_dict,
                                                 img_index)

                            if real_txt == "right":
                                if is_introduction or is_menu or is_acupuncture:
                                    img_index = playNextImage("next", self.windows.ui_playable_images, img_index)
                                    if is_acupuncture:
                                        update_acupuncture_txt(self.windows.acupuncture_img[0], self.windows.ui_txts,
                                                               txt_dict, acupuncture_info,
                                                               playable_index["acupuncture"], img_index,
                                                               acupuncture_txt)
                            if real_txt == "left":
                                if is_introduction or is_menu or is_acupuncture:
                                    img_index = playNextImage("back", self.windows.ui_playable_images, img_index)
                                    if is_acupuncture:
                                        update_acupuncture_txt(self.windows.acupuncture_img[0], self.windows.ui_txts,
                                                               txt_dict, acupuncture_info,
                                                               playable_index["acupuncture"], img_index,
                                                               acupuncture_txt)

                            if real_txt == "choose":  # choose menu function
                                if is_menu:
                                    is_menu = False
                                    menu_un_visible(self.windows.ui_images, self.windows.ui_playable_images, image_dict,
                                                    img_index)
                                    menu_pi = 0
                                    if img_index == playable_index["menu"]:  # introduction
                                        menu_pi = playable_index["menu"]
                                        img_index = playable_index["introduction"]
                                        is_introduction = True
                                        self.windows.ui_playable_images[img_index].set_visible()

                                    # bluetooth connect to phone with QRcode
                                    if img_index == playable_index["menu"] + 1:
                                        menu_pi = playable_index["menu"] + 1
                                        self.Qrcode_is_read = True
                                        now_img = self.windows.ui_images[image_dict["Qrcode"]]
                                        now_img.set_visible()

                                    # close acupuncture point
                                    if img_index == playable_index["menu"] + 2:
                                        menu_pi = playable_index["menu"] + 2
                                        acupuncture_show = False
                                        ap_filter = []
                                        classifyArea_un_visible(self.windows.ui_images, self.windows.ui_txts,
                                                                image_dict["classifyArea"],
                                                                txt_dict["classify_txt"])
                                    # close acupuncture point
                                    if img_index == playable_index["menu"] + 3:
                                        menu_pi = playable_index["menu"] + 3
                                        acupuncture_show = True
                                        ap_filter = None
                                    self.windows.ui_playable_images[menu_pi].set_un_visible()  # close txt

                            if real_txt == "cancel":
                                if is_acupuncture:
                                    is_acupuncture = False
                                    close_acupuncture_txt(self.windows.acupuncture_img[0], self.windows.ui_txts,
                                                          txt_dict)

                                if not is_introduction and is_menu:  # close menu
                                    is_menu = False
                                    menu_un_visible(self.windows.ui_images, self.windows.ui_playable_images, image_dict,
                                                    img_index)
                                if img_index == playable_index["menu"] - 1:  # close introduction
                                    is_introduction = False

                                if now_img:
                                    now_img.set_un_visible()
                                self.windows.ui_playable_images[img_index].set_un_visible()
                                self.Qrcode_is_read = False

                            SetGestureImageOn(real_txt, self.windows.ui_images)
                            self.main_profile.label_end()

                        timer += 1
                        count += 1

                self.trigger.emit(img3)
                self.main_profile.end()
        self.finished.emit()

    def bt_get_message(self, txt):
        self.bt_is_receive = True
        self.bt_txt = txt

    def bt_is_finish(self, is_finish):
        if is_finish:
            print("finish")
        else:
            print("please open phone server or phone bluetooth...")
            self.Qrcode_is_read = False
            self.bt.close()


def cap_resize(img, x, y, w, h):
    return img[y:y + h, x:x + w]


def acupuncture_list_by_disease(dots, disease):
    dot_list = []
    for dot in dots:
        acupuncture_types = dot.get_acupuncture().types
        for d in acupuncture_types:
            if str(disease) == str(d):
                dot_list.append(dot)
                break
    return dot_list


def landmark_to_hand_acupunncture(landmark):
    """
    use hand landmark to get rotation degree, base_length(use in normalize size), Vertical Normal(judge the font or back of hand)
    :param landmark: input mediapipe hand landmark
    :return:
        degree: the rotation angle of hand
        base_length: normalize length, use 0 and 1 hand point
        v1nv2: the Vertical Normal(垂直法向量) of [landmark0, landmark5] and [landmark0, landmark17]
    """
    v0to17 = np.array([landmark[0].x - landmark[17].x, landmark[0].y - landmark[17].y])
    v0to5 = np.array([landmark[0].x - landmark[5].x, landmark[0].y - landmark[5].y])
    v0to1 = np.array([landmark[0].x - landmark[1].x, landmark[0].y - landmark[1].y])
    v0to9 = np.array([landmark[0].x - landmark[9].x, landmark[0].y - landmark[9].y])
    degree = GetRotation(v0to9)
    # print(degree)
    base_length = np.linalg.norm(v0to1)
    # - RightHandFront  + RightHandBack,  + LeftHandFont,  - LeftHandBack
    v1nv2 = np.cross(v0to17, v0to5)
    return degree, base_length, v1nv2


def landmark_to_face_acupunncture(landmark):
    """
    use face mesh landmark to get rotation degree, base_length(use in normalize size)
    :param landmark: input mediapipe face mesh landmark
    :return:
        degree: the rotation angle of hand
        base_length: normalize length, use 10 and 151 face mesh point
    """
    v10to151 = np.array([landmark[10].x - landmark[151].x, landmark[10].y - landmark[151].y])
    v10to152 = np.array([landmark[10].x - landmark[152].x, landmark[10].y - landmark[152].y])
    degree = GetRotation(v10to152)
    # print(degree)
    base_length = np.linalg.norm(v10to151)
    return degree, base_length


def set_acupuncture_point(landmark, lower_limit, upper_limit, base_length, degree, is_face=False, ap_filter=None):
    """
    set the dot to acupuncture
    :param landmark: input landmark
    :param lower_limit: input the range of lower num of dots in case of Classification
    :param upper_limit: input the range of upper num of dots in case of Classification
    :param base_length: the base_length of landmark_to_face_acupunncture or landmark_to_hand_acupunncture
    :param degree: the degree of landmark_to_face_acupunncture or landmark_to_hand_acupunncture
    :param is_face: if the landmark is face, set it true!
    :param filter: show the certain point
    :return:
    """
    if ap_filter is None:
        for i in range(lower_limit, upper_limit):
            set_dot(dots[i], landmark, base_length, degree, is_face)
    else:
        for i in range(lower_limit, upper_limit):
            if len(ap_filter) == 0:
                break
            for d in ap_filter:
                if dots[i] == d:
                    set_dot(dots[i], landmark, base_length, degree, is_face)


def set_dot(dot, landmark, base_length, degree, is_face):
    acupuncture = dot.get_acupuncture()
    offset_x, offset_y = SetOffset(acupuncture.offset_x * base_length / 15,
                                   acupuncture.offset_y * base_length / 15, degree)
    x = (landmark[acupuncture.rel_point].x + offset_x) * (
            screen[0] - screen_x_offset * 2) + screen_x_offset  # x 座標
    if is_face:
        x = (1 - landmark[acupuncture.rel_point].x + offset_x) * (
                screen[0] - screen_x_offset * 2) + screen_x_offset  # x 座標
    y = (landmark[acupuncture.rel_point].y + offset_y) * screen[1]  # y 座標
    dot.move_dot(int(x), int(y))
    dot.set_active()


def set_face_box_ap(output, ap_filter=None):
    if ap_filter is None:
        for i in range(0, len(output), 2):
            x = (1 - output[i]) * (screen[0] - screen_x_offset * 2) + screen_x_offset  # x 座標
            y = (output[i + 1]) * screen[1]  # y 座標
            dots[i // 2].move_dot(int(x), int(y))
            dots[i // 2].set_active()
    else:
        for i in range(0, len(output), 2):
            for d in ap_filter:
                if dots[i // 2] == d:
                    x = (1 - output[i]) * (screen[0] - screen_x_offset * 2) + screen_x_offset  # x 座標
                    y = (output[i + 1]) * screen[1]  # y 座標
                    dots[i // 2].move_dot(int(x), int(y))
                    dots[i // 2].set_active()


def HandRecognition(label, n):
    """

    :param label: the hand is Right or Left
    :param n: the Vertical Normal of hand
    :return:
        lower_limit: input the range of lower num of dots in case of Classification
        upper_limit: input the range of upper num of dots in case of Classification
    """
    lower_limit = 0
    upper_limit = 0
    if label == "Right" and n < 0:  # RightHandFront 5
        # print("RightHandFront")
        lower_limit = types[0] + types[3] + types[4]
        upper_limit = lower_limit + types[5]
    elif label == "Right" and n > 0:  # RightHandBack 6
        # print("RightHandBack")
        lower_limit = types[0] + types[3] + types[4] + types[5]
        upper_limit = lower_limit + types[6]
    elif label == "Left" and n > 0:  # LeftHandFont 3
        # print("LeftHandFont")
        lower_limit = types[0]
        upper_limit = lower_limit + types[3]
    elif label == "Left" and n < 0:  # LeftHandBack 4
        # print("LeftHandBack")
        lower_limit = types[0] + types[3]
        upper_limit = lower_limit + types[4]
    elif label == "Left" and n < 0:  # face 0
        # print("LeftHandBack")
        lower_limit = 0
        upper_limit = lower_limit + types[0]
    return lower_limit, upper_limit


def GetRotation(v):
    """
    get degree of handRotation. finger is up: 180, finger is down: 0
    :param v: np vector of [x1-x2,y1-y2]
    :return: degree
    """
    # _baseHandRotation = np.array([-0.0376609, -0.3828629])
    _baseHandRotation = np.array([0.01302427, 0.39472041])
    V1_norm = np.linalg.norm(_baseHandRotation)  # 計算長度
    V2_norm = np.linalg.norm(v)
    dot_product = np.dot(_baseHandRotation, v)  # 計算內積
    cross_product = np.cross(_baseHandRotation, v)  # 計算外積
    angle = np.arccos(dot_product / (V1_norm * V2_norm))  # 計算弧度
    angle_degrees = np.degrees(angle)  # 把弧度轉為角度
    if cross_product < 0:
        angle_degrees = 360 - angle_degrees
    return angle_degrees


def get_direction(x, y):
    """
    get two point direction
    :param x: input point x
    :param y: input point y
    :return: direction
    """
    vec = np.array([x, y])
    length = np.linalg.norm(vec)
    direction = vec / length
    return direction


def SetOffset(offset_x, offset_y, degree):
    """
    set offset of landmark
    :param offset_x: data base offset_x
    :param offset_y: data base offset_y
    :param degree: GetRotation degree
    :return: offset_x, offset_y
    """
    vec = np.array([offset_x, offset_y])
    theta = np.deg2rad(degree)
    # 計算旋轉矩陣
    rot_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                           [np.sin(theta), np.cos(theta)]])
    rot_vec = rot_matrix.dot(vec)  # 計算旋轉後的向量
    return rot_vec[0], rot_vec[1]


class ImageWidget(QWidget):
    def __init__(self, image, timer_txt):
        super().__init__()
        self.image = image
        self.timer_txt = timer_txt
        self.profile = time_profile.TimeProfile(True)

    def setImage(self, image):
        self.image = image
        self.update()

    def paintEvent(self, event):
        self.profile.begin("Annotation paint")

        self.profile.label_begin("image scale")
        painter = QPainter(self)
        qt_image = QImage(self.image.data, self.image.shape[1], self.image.shape[0],
                          self.image.strides[0], QImage.Format_RGB888)
        scaled_image = qt_image.scaledToHeight(screen[1])
        global screen_x_offset
        screen_x_offset = (screen[0] - scaled_image.width()) // 2
        self.profile.label_end()
        self.profile.label_begin("paint image")
        painter.drawPixmap(screen_x_offset, 0, QPixmap.fromImage(scaled_image))
        self.profile.label_end()

        self.profile.label_begin("lock")
        mutex.lock()
        self.profile.label_end()

        self.profile.label_begin("paint acupoints")
        if len(dots) > 0:
            painter.setBrush(Qt.red)
            painter.setPen(Qt.red)
            for d in dots:
                d.paint_circle(painter)
            painter.setPen(QColor(255, 255, 255))
            painter.setFont(QFont('Arial', 20))
            for d in dots:
                d.paint_font(painter)
        self.profile.label_end()

        self.profile.label_begin("unlock")
        mutex.unlock()
        self.profile.label_end()

        self.profile.label_begin("paint timer")
        self.timer_txt.paint(painter)
        self.profile.label_end()

        self.profile.end()


class DotMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("background-color: black;")
        cap = cv2.VideoCapture(0)
        ret, img = cap.read()
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.ui_images = [
            ImageUI(0, 0, 1024, 768, 'UI/StaticBackground', self, disable=False),
            ImageUI(337, 704, 45, 43, 'UI/left-on.png', self, disable=False),  # 1
            ImageUI(444, 688, 45, 60, 'UI/choose-on.png', self, disable=False),  # 2
            ImageUI(548, 704, 47, 43, 'UI/right-on.png', self, disable=False),  # 3
            ImageUI(696, 704, 42, 43, 'UI/menu-on.png', self, disable=False),  # 4
            ImageUI(842, 688, 47, 59, 'UI/cancel-on.png', self, disable=False),  # 5
            # ImageUI(820, 43, 34, 26, 'UI/wifi-on.png', self, disable=False),         # 5
            ImageUI(45, 684, 41, 72, 'UI/phone-on.png', self, disable=True),  # 6
            ImageUI(0, 0, 1024, 768, 'UI/back.png', self, disable=True, opacity_level=0.9),  # 7
            ImageUI(320, 580, 337, 68, 'UI/menu.png', self, disable=True),  # 8
            ImageUI(0, 0, 1024, 768, 'UI/Qrcode_reader.png', self, disable=True, opacity_level=0.9),  # 9
            ImageUI(170, 150, 684, 408, 'UI/Blutoothsuccess.png', self, disable=True),  # 10
            ImageUI(170, 150, 684, 408, 'UI/Blutoothfail.png', self, disable=True),  # 11
            ImageUI(0, 20, 308, 87, 'UI/classifyArea.png', self, disable=True, opacity_level=0.9),  # 12
        ]

        self.ui_playable_images = [
            PlayableImageUI(100, 80, 802, 593, 'UI/instruction01.png', -1, 1, self,
                            disable=True),
            PlayableImageUI(100, 80, 802, 593, 'UI/instruction02.png', 0, 2, self,
                            disable=True),
            PlayableImageUI(100, 80, 802, 593, 'UI/instruction03.png', 1, 3, self,
                            disable=True),
            PlayableImageUI(100, 80, 802, 593, 'UI/instruction04.png', 2, 4, self,
                            disable=True),
            PlayableImageUI(100, 80, 802, 593, 'UI/instruction05.png', 3, 5, self,
                            disable=True),
            PlayableImageUI(100, 80, 802, 593, 'UI/instruction06.png', 4, 6, self,
                            disable=True),
            PlayableImageUI(100, 80, 802, 593, 'UI/instruction07.png', 5, -1, self,
                            disable=True),
            PlayableImageUI(307, 535, 96, 40, 'UI/instrctions-on.png', -1, 8, self, disable=True),  # 7
            PlayableImageUI(398, 535, 96, 41, 'UI/pairing-on.png', 7, 9, self, disable=True),  # 8
            PlayableImageUI(489, 535, 96, 40, 'UI/unshow-on.png', 8, 10, self, disable=True),  # 9
            PlayableImageUI(578, 535, 96, 41, 'UI/show-on.png', 9, -1, self, disable=True),  # 10
            PlayableImageUI(150, 100, 744, 550, 'UI/acupuncture01.png', -1, 12, self, disable=True),  # 11
            PlayableImageUI(150, 100, 744, 550, 'UI/acupuncture02.png', 11, 13, self, disable=True),  # 12
            PlayableImageUI(150, 100, 744, 550, 'UI/acupuncture03.png', 12, 14, self, disable=True),  # 13
            PlayableImageUI(150, 100, 744, 550, 'UI/acupuncture04.png', 13, 15, self, disable=True),  # 14
            PlayableImageUI(150, 100, 744, 550, 'UI/acupuncture05.png', 14, -1, self, disable=True),  # 15
        ]
        self.acupuncture_img = [
            ImageUI(610, 260, 205, 205, 'UI/acupuncture/acupuncture.png', self, disable=True), ]

        self.ui_txts = [TextUI(30, 65, self, text="頭痛", disable=True, font_size=14),
                        TextUI(907, 615, self, text="loading", disable=True, color="(255,255,255,1)"),
                        TextUI(425, 160, self, text="穴穴穴", disable=True, font_size=32),  # 2
                        TextUI(240, 310, self, text="我我我我我我我我我我我我我我我", disable=True, font_size=14),  # 3
                        TextUI(240, 345, self, text="我我我我我我我我我我我我我我我", disable=True, font_size=14),  # 4
                        TextUI(240, 380, self, text="我我我我我我我我我我我我我我我", disable=True, font_size=14),  # 5
                        TextUI(240, 415, self, text="我我我我我我我我我我我我我我我", disable=True, font_size=14),  # 6
                        TextUI(240, 455, self, text="我我我我我我我我我我我我我我我", disable=True, font_size=14),  # ignore 7
                        TextUI(240, 495, self, text="我我我我我我我我我我我我我我我我我我我我我我我", disable=True, font_size=14),  # 8
                        TextUI(240, 530, self, text="我我我我我我我我我我我我我我我我我我我我我我我", disable=True, font_size=14),  # 9
                        TextUI(240, 565, self, text="我我我我我我我我我我我我我我我我我我我我我我我", disable=True, font_size=14), ]

        self.ui_gifs = [ImageGIF(900, 650, 106, 85, 'UI/loading.gif', self, disable=True), ]
        self.timer_txt = TimerText(660, 65, disable=False)

        self.image_widget = ImageWidget(cv2.flip(img2, 1), self.timer_txt)

        self.init_ui_before_widget()  # layer01
        self.setCentralWidget(self.image_widget)  # layer02  # Update every frame
        self.init_ui_after_widget()  # layer03

        self.work = WorkerThread(self)
        self.startThread()

    def init_ui_after_widget(self):
        for ui_image in self.ui_images:
            ui_image.initUI()  # set UI in front
        for ui_image in self.ui_playable_images:
            ui_image.initUI()  # set UI in front
        for ui_image in self.acupuncture_img:
            ui_image.initUI()  # set UI in front
        for ui_image in self.ui_gifs:
            ui_image.initUI()  # set UI in front
        for ui_txt in self.ui_txts:
            ui_txt.initUI()
        # self.ui_button[0].move(1000, 200)  # move UI button

    def init_ui_before_widget(self):
        pass
        # for ui_image in self.ui_images:
        #     ui_image.initUI()  # set UI in front

    def startThread(self):
        self.work.start()
        self.work.trigger.connect(self.run_openCV)

    def run_openCV(self, img):
        self.image_widget.setImage(img)


def key(self):
    keycode = self.key()  # 取得該按鍵的 keycode
    if keycode == 16777216:
        QApplication.instance().quit()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DotMainWindow()
    # window.show()
    window.showFullScreen()
    window.keyPressEvent = key
    screen.append(window.width())
    screen.append(window.height())
    sys.exit(app.exec_())
