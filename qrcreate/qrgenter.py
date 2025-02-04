# -*- coding: utf-8 -*-
import cv2,os
import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import QMessageBox
from qrgenui import Ui_qrgenter
import qrcode

class Qrgen(QMainWindow,Ui_qrgenter):
    def __init__(self):
        super(Qrgen,self).__init__()
        self.outputflod=None

        self.setupUi(self)
        
        self.gen_Button.clicked.connect(self.qrgen)
        self.choseoutputfolder_Button.clicked.connect(self.chose_outputfloder)

    def qrgen(self):
        if self.outputflod is not None:
            # imagelist=create_qrCode(self.inputfile)
            string=str(self.lineEdit.text())
            create_qr(string,self.outputflod)
            QMessageBox.about(self,"消息提示","二维码生成完成")
        else:
             QMessageBox.about(self,"消息提示","请重新选择或输入")

    def chose_outputfloder(self):
        str_path = QFileDialog.getExistingDirectory(None,"选取文件夹","") 
        if str_path is not None:
            self.outputflod=str_path
            QMessageBox.about(self,"消息提示","选择成功,请开始生成")
        else:
            QMessageBox.about(self,"消息提示","选择失败,请重新选择")


def create_qr(str,filepath):

    error_correction_1 = qrcode.constants.ERROR_CORRECT_H
    qr = qrcode.QRCode(version=5,
                           error_correction=error_correction_1,
                           border=4,
                           box_size=10)

    qr.add_data(str)
    filenum=len(os.listdir(filepath))
    img = qr.make_image()#2维
    img.save(filepath+"/"+str+".png")

app=QApplication(sys.argv)
decoder=Qrgen()
decoder.show()
sys.exit(app.exec_())