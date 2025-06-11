#Khai báo thư viện
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
# Đọc dữ liệu đầu vào
duongDan = 'Data_Gestational_diabetes.csv'
# Dữ liệu đầu vào bao gồm 23 đặc trưng và đặc trưng thứ 23 là nhãn (0 - Bình Thường, 1 - Mắc ĐTD)
tenCot = ['Year_old','Occupatio','Education','gestational_age','Number_of_pregnancy','Para','Childbirth_history',
              'Family_history','Height','Pre_gregnancy_weight','Current_Weight','Upper_blood_pressure',
              'Lower_blood_pressure','Work_status','Work_performance','Nutrition', 'Sport_practice_day_per_month',
              'Sport_day_per_week', 'Sport_last_month', 'Glucose1','Glucose2','Glucose3','Conclude']
duLieu = pd.read_csv(duongDan, names= tenCot)
# Chuyển dữ liệu đầu vào sang dạng Ma trận
matrixDuLieu = duLieu.values
# Lấy ra tập đặc trưng huấn luyện
X = matrixDuLieu[1:,:-1]
# Lấy nhãn chẩn đoán: 0 - Bình Thường, 1 - Mắc ĐTD
y = matrixDuLieu[1:,-1]
#duLieu.head()
# Thông tin bệnh nhân cần chẩn đoán
data_patients =bn =[[28,0,1,27,2,3,0,0,1.6,47,52,120,80,1,0,2,0,0,0,4.1,4.68,4.97], 
     [24,3,1,24,1,0,0,0,1.49,47,52,120,80,2,0,2,0,0,0,4.14,10.87,12.15 ]] 
# Huấn luyện mô hình SVM
svm1 = svm.SVC(kernel='rbf', C=100000, gamma=1e-06)
svm1.fit(X,y)
# Dự đoán
svm_patient_pred  = svm1.predict(data_patients)
print('Kết quả chuẩn đoánSVM ',svm_patient_pred )
# Gọi Random Forest 
rd = RandomForestClassifier(bootstrap=True, n_estimators=100, criterion='gini', max_features= 0.75)
rd.fit(X,y) 
bn_pred = rd.predict(data_patients) 
print('Kết quả chẩn đoán Random forest',bn_pred) 