import random
import sys

import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QTableWidgetItem, QPushButton
from sklearn import svm

from interface_ui import Ui_MainWindow


class MainGUI(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(QMainWindow, self).__init__()
        self.setupUi(self)

        self.random_bt = QPushButton("Random", self)
        self.random_bt.setFixedSize(100, 25)
        self.random_bt.move(120, 340)
        self.random_bt.clicked.connect(self.random_fc)
        self.result_text.move(240, 342)


        duongDan = 'Data_Gestational_diabetes.csv'
        # Dữ liệu đầu vào bao gồm 23 đặc trưng và đặc trưng thứ 23 là nhãn (0- Bình Thường, 1 - Mắc ĐTD)
        tenCot = ['Year_old', 'Occupatio', 'Education', 'gestational_age', 'Number_of_pregnancy', 'Para',
                  'Childbirth_history',
                  'Family_history', 'Height', 'Pre_gregnancy_weight', 'Current_Weight', 'Upper_blood_pressure',
                  'Lower_blood_pressure', 'Work_status', 'Work_performance', 'Nutrition',
                  'Sport_practice_day_per_month',
                  'Sport_day_per_week', 'Sport_last_month', 'Glucose1', 'Glucose2', 'Glucose3', 'Conclude']
        duLieu = pd.read_csv(duongDan, names=tenCot)
        # Chuyển dữ liệu đầu vào sang dạng Ma trận
        matrixDuLieu = duLieu.values
        # Lấy ra tập đặc trưng huấn luyện
        X = matrixDuLieu[1:, :-1]
        # Lấy nhãn chẩn đoán: 0 - Bình Thường, 1 - Mắc ĐTD
        y = matrixDuLieu[1:, -1]
        # duLieu.head()
        # Huấn luyện mô hình SVM
        self.svm1 = svm.SVC(kernel='rbf', C=100000, gamma=1e-06)
        self.svm1.fit(X, y)

        nghenghiep_list =  [
    "Tự do", "Nông dân", "Nhân viên", "Công nhân", "Kế toán",
    "Giáo viên", "Cán bộ", "Dược sĩ", "Luật sư", "Sinh viên",
    "Kinh doanh", "Cảnh sát", "Bác sĩ", "Điều dưỡng", "Kiến trúc sư",
    "Kỹ sư"
]
        self.nghenghiep_cb.addItems(nghenghiep_list)
        trinhdo_list = [
            "Đại học",
            "Cao đẳng",
            "Trung cấp",
            "Trung học phổ thông"
        ]
        self.trinhdohocvan_cb.addItems(trinhdo_list)
        gestational_age = [
            "Từ 24.0 đến 24.5", "Từ 24.6 đến 25.5", "Từ 25.6 đến 26.5",
            "Từ 26.6 đến 27.5", "Từ 27.6 đến 28"
        ]
        index_para = [
            "0000", "0010", "1001", "1011", "1031",
            "2002", "2012", "2022", "2032", "3013",
            "1021", "3003"
        ]
        self.tuoithai_cb.addItems(gestational_age)
        self.tiensusinhno_cb.addItems(['Không', 'Có'])
        self.solansinhno_cb.addItems(index_para)
        self.tiensugiadinhdtd_cb.addItems(['Không', 'Có'])
        self.tccv_truoccothai_cb.addItems(['Nhẹ', 'Vừa', 'Nặng'])
        self.tccv_saucothai_cb.addItems(['Nhẹ', 'Vừa', 'Nặng'])
        self.chedodinhduong_cb.addItems(['Kém hơn', 'Bằng nhau', 'Tốt hơn'])
        self.luyentapthethao1thang_cb.addItems(['Không', 'Có'])
        self.predict_one_bt.clicked.connect(self.predict_fc)

    def random_fc(self):
        try:
            nghenghiep_list = [
                "Tự do", "Nông dân", "Nhân viên", "Công nhân", "Kế toán",
                "Giáo viên", "Cán bộ", "Dược sĩ", "Luật sư", "Sinh viên",
                "Kinh doanh", "Cảnh sát", "Bác sĩ", "Điều dưỡng", "Kiến trúc sư",
                "Kỹ sư"
            ]

            trinhdo_list = [
                "Đại học",
                "Cao đẳng",
                "Trung cấp",
                "Trung học phổ thông"
            ]

            gestational_age = [
                "Từ 24.0 đến 24.5", "Từ 24.6 đến 25.5", "Từ 25.6 đến 26.5",
                "Từ 26.6 đến 27.5", "Từ 27.6 đến 28"
            ]

            index_para = [
                "0000", "0010", "1001", "1011", "1031",
                "2002", "2012", "2022", "2032", "3013",
                "1021", "3003"
            ]


            random_gestational_age = random.choice(gestational_age)
            random_index_para = random.choice(index_para)

            self.tuoi_txt.setText(str(np.random.randint(18, 40)))
            self.nghenghiep_cb.setCurrentText(random.choice(nghenghiep_list))
            self.trinhdohocvan_cb.setCurrentText(random.choice(trinhdo_list))
            self.solanmangthai_txt.setText(str(np.random.randint(0, 3)))

            self.glu1_cb.setText(str(round(np.random.uniform(3, 8), 2)))
            self.glu2_cb.setText(str(round(np.random.uniform(3, 8), 2)))
            self.glu3_cb.setText(str(round(np.random.uniform(3, 8), 2)))

            self.ngaytuan_txt.setText(str(np.random.randint(1, 6)))
            self.phutngay_txt.setText(str(np.random.randint(10, 60)))

            self.chieucao_txt.setText(str(np.random.randint(145, 170)))
            self.cannangtruocmangthai_txt.setText(str(np.random.randint(40, 60)))
            self.cannanghientai_txt.setText(str(np.random.randint(50, 70)))
            self.huyetaptren_txt.setText(str(np.random.randint(100, 130)))
            self.huyetapduoi_txt.setText(str(np.random.randint(60, 80)))

            self.tuoithai_cb.setCurrentText(random_gestational_age)
            self.tiensusinhno_cb.setCurrentText(random.choice(['Không', 'Có']))
            self.solansinhno_cb.setCurrentText(random_index_para)
            self.tiensugiadinhdtd_cb.setCurrentText(random.choice(['Không', 'Có']))
            self.tccv_truoccothai_cb.setCurrentText(random.choice(['Nhẹ', 'Vừa', 'Nặng']))
            self.tccv_saucothai_cb.setCurrentText(random.choice(['Nhẹ', 'Vừa', 'Nặng']))
            self.chedodinhduong_cb.setCurrentText(random.choice(['Kém hơn', 'Bằng nhau', 'Tốt hơn']))
            self.luyentapthethao1thang_cb.setCurrentText(random.choice(['Không', 'Có']))


        except Exception as ex:
            print(ex)
    def predict_fc(self):
        try:
            tuoithais = [24, 25, 26, 27, 28]

            data = [[int(self.tuoi_txt.text()), self.nghenghiep_cb.currentIndex(), self.trinhdohocvan_cb.currentIndex(),
                    tuoithais[self.tuoithai_cb.currentIndex()], int(self.solanmangthai_txt.text()), self.solansinhno_cb.currentIndex(),
                    self.tiensusinhno_cb.currentIndex(), self.tiensugiadinhdtd_cb.currentIndex(), float(self.chieucao_txt.text()),
                    float(self.cannangtruocmangthai_txt.text()), float(self.cannanghientai_txt.text()), float(self.huyetaptren_txt.text()),
                    float(self.huyetapduoi_txt.text()), self.tccv_truoccothai_cb.currentIndex(), self.tccv_saucothai_cb.currentIndex(),
                    self.chedodinhduong_cb.currentIndex(), self.luyentapthethao1thang_cb.currentIndex(), float(self.ngaytuan_txt.text()), float(self.phutngay_txt.text()),
                    float(self.glu1_cb.text()), float(self.glu2_cb.text()), float(self.glu3_cb.text())]]
            print(data)
            bn_pred = self.svm1.predict(data)
            print('Kết quả chẩn đoán SVM ', bn_pred)
            if bn_pred == "0":
                self.result_text.setText("Bệnh nhân không mắc tiểu đường thai kì")
            elif bn_pred == "1":
                self.result_text.setText("Bệnh nhân mắc tiểu đường thai kì")
            elif bn_pred == "2":
                self.result_text.setText("Bệnh nhân mắc tiểu đường thai kì nặng")

        except Exception as ex:
            self.result_text.setText("Vui lòng kiểm tra lại thông tin")




if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainGUI()
    main_window.setWindowTitle('Giao diện')
    main_window.show()
    sys.exit(app.exec_())