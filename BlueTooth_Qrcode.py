from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget
from PyQt5.QtGui import QImage, QPainter, QPen
from PyQt5.QtCore import Qt, QPoint, QThread, pyqtSignal
import sys
import cv2
import bluetooth


def QRcodeReader(img):  # get Qrcode
    detect = cv2.QRCodeDetector()
    value, points, straight_qrcode = detect.detectAndDecode(img)
    return value


class BlueToothConnection(QThread):
    finished = pyqtSignal(bool)
    send_message = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.value = []
        self.socket = None

    def run(self):
        if not self.perparing():
            self.sleep(2000)  # wait for close
        self.connected2Server()
        self.finished.emit(True)

    def setValue(self, value):  # the Mac and UUID of phone server
        self.value = str(value).split("@@@")
        print(self.value)

    def perparing(self):
        if len(self.value) != 2:
            self.send_message.emit("BlueTooth@@@fail")
            return 0
        NAME, UUID = self.value
        print("Name: {}, UUID: {}".format(NAME, UUID))
        self.send_message.emit("BlueTooth@@@loading")
        devices = bluetooth.discover_devices(lookup_names=True)
        isPaired = False
        for mac_address, name in devices:
            if NAME == name:
                print("MAC: {}, UUID: {}".format(mac_address, UUID))
                self.MAC = mac_address
                isPaired = True
        if not isPaired:
            print("not available")
            self.send_message.emit("BlueTooth@@@fail")
            self.finished.emit(False)
            return 0
        self.UUID = UUID
        return 1

    def connected2Server(self):
        print("starting connect to server...")
        service_matches = bluetooth.find_service(uuid=self.UUID, address=self.MAC)
        print(len(service_matches))
        if len(service_matches) == 0:
            print("Couldn't find the SampleServer service.")
            self.send_message.emit("BlueTooth@@@fail")
            sys.exit(0)
        first_match = service_matches[len(service_matches) - 1]
        port = first_match["port"]
        host = first_match["host"]

        print("Connecting to {}".format(host))

        self.sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
        self.sock.connect((host, port))
        self.send_message.emit("BlueTooth@@@success")
        while True:
            data = self.sock.recv(1024)
            if not data:
                break
            # print(data.decode("utf-8"))
            self.send_message.emit(data.decode("utf-8"))

        self.sock.close()
        self.send_message.emit("BlueTooth@@@close")

    def close(self):
        if self.sock != None:
            self.sock.close()
        self.terminate()
