# -*- coding: utf-8 -*-
import cv2, os
import sys, time
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import QMessageBox
from module import *
from ui import Ui_filedecoder


class Decoder(QMainWindow, Ui_filedecoder):
    def __init__(self):
        super(Decoder, self).__init__()
        self.inputfile = None
        self.outputflod = None
        self.setupUi(self)

        self.decode_Button.clicked.connect(self.decode_programer)
        self.choseinputfile_Button.clicked.connect(self.chose_inputfiler)
        self.choseoutputfolder_Button.clicked.connect(self.chose_outputfloder)

    def decode_programer(self):
        if self.inputfile is not None and self.outputflod is not None:
            start_time = time.time()
            imagelist = v2i(self.inputfile)
            end_time = time.time()
            print(f"v2i 执行时间: {end_time - start_time:.4f} 秒")

            start_time = time.time()
            textdict = decode_list(imagelist, self.outputflod)
            end_time = time.time()
            print(f"decode_list 执行时间: {end_time - start_time:.4f} 秒")

            start_time = time.time()
            write_file(textdict, self.outputflod, 0)
            end_time = time.time()
            print(f"write_file 执行时间: {end_time - start_time:.4f} 秒")

            QMessageBox.about(self, "消息提示", "解码完成")
        else:
            QMessageBox.about(self, "消息提示", "文件或文件夹选择失败,请重新选择")

    def chose_inputfiler(self):
        open_filename = QFileDialog.getOpenFileName(None, '选择文件', '', 'All files(*.*)')
        str = ""
        if open_filename[0] != '':
            if self.outputflod is not None:
                str += "输入路径： " + open_filename[0] + "\n"
                str += "输出路径： " + self.outputflod
                self.plainTextEdit.setPlainText(str)
                self.inputfile = open_filename[0]
                QMessageBox.about(self, "消息提示", "选择成功,请开始解码")
            else:
                str += "输入路径： " + open_filename[0] + "\n"
                self.plainTextEdit.setPlainText(str)
                self.inputfile = open_filename[0]
                QMessageBox.about(self, "消息提示", "选择成功,请选择输出文件夹")
        else:
            QMessageBox.about(self, "消息提示", "选择失败,请重新选择")

    def chose_outputfloder(self):
        str_path = QFileDialog.getExistingDirectory(None, "选取文件夹", "")
        str = ""
        if str_path is not None:
            if self.inputfile is not None:
                str = "输入路径： " + self.inputfile + "\n"
                str += "输出路径： " + str_path
                self.plainTextEdit.setPlainText(str)
                self.outputflod = str_path
                QMessageBox.about(self, "消息提示", "选择成功,请开始解码")
            else:
                str = str_path
                self.plainTextEdit.setPlainText(str)
                self.outputflod = str_path
                QMessageBox.about(self, "消息提示", "选择成功,请选择输入文件")


        else:
            QMessageBox.about(self, "消息提示", "选择失败,请重新选择")


app = QApplication(sys.argv)
decoder = Decoder()
decoder.show()
sys.exit(app.exec_())
