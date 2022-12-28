#I am importing libraries, including KNNImputer and PolynomialFeatures.

import pandas as pd
import seaborn as sns
import numpy as np
import statistics
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

#I am downloading the data and using the 'info' function to show information about it. 
#The data has some missing values, so I am filling them with 0 and changing the data type to int for easier manipulation.


MPhones = pd.read_csv('C:/Users/..../Mobile-phones.csv', sep=';')

#MPhones.info()
#RangeIndex: 2000 entries, 0 to 1999
#Data columns (total 21 columns):
# Column Non-Null Count Dtype
#--- ------ -------------- -----
#0 battery_power 2000 non-null int64
#1 blue 2000 non-null int64
#2 clock_speed 2000 non-null float64
#3 dual_sim 2000 non-null int64
#4 fc 2000 non-null int64
#5 four_g 2000 non-null int64
#6 int_memory 2000 non-null int64
#7 m_dep 1324 non-null float64
#8 mobile_wt 2000 non-null int64
#9 n_cores 2000 non-null int64
#10 pc 2000 non-null int64
#11 px_height 1368 non-null float64
#12 px_width 1326 non-null float64
#13 ram 1821 non-null float64
#14 sc_h 1781 non-null float64
#15 sc_w 2000 non-null int64
#16 talk_time 2000 non-null int64
#17 three_g 1671 non-null float64
#18 touch_screen 2000 non-null int64
#19 wifi 2000 non-null int64
#20 price_range 2000 non-null int64
#dtypes: float64(7), int64(14)


MPhones = MPhones.fillna(0)

MPhones = MPhones.astype(int)


#I am using KNNImputer, an algorithm that takes into account the nearest data points and creates the mean between them to fill missing data.
#I am fitting the algorithm and transforming the data, which should now be filled with data.

imputer = KNNImputer()
MPhones.info()

#RangeIndex: 2000 entries, 0 to 1999
#Data columns (total 21 columns):
# Column Non-Null Count Dtype
#--- ------ -------------- -----
#0 battery_power 2000 non-null int32
#1 blue 2000 non-null int32
#2 clock_speed 2000 non-null int32
#3 dual_sim 2000 non-null int32
#4 fc 2000 non-null int32
#5 four_g 2000 non-null int32
#6 int_memory 2000 non-null int32
#7 m_dep 2000 non-null int32
#8 mobile_wt 2000 non-null int32
#9 n_cores 2000 non-null int32
#10 pc 2000 non-null int32
#11 px_height 2000 non-null int32
#12 px_width 2000 non-null int32
#13 ram 2000 non-null int32
#14 sc_h 2000 non-null int32
#15 sc_w 2000 non-null int32
#16 talk_time 2000 non-null int32
#17 three_g 2000 non-null int32
#18 touch_screen 2000 non-null int32
#19 wifi 2000 non-null int32
#20 price_range 2000 non-null int32
#dtypes: int32(21)


#I am plotting the correlation matrix to see the degree to which the variables correlate with the target feature. 
#If the correlation is low, the predictive data will be elastic and low on bias, so I will use it.

sns.heatmap(MPhones.corr())
plt.show()

#I am creating train and test data and fitting it to Linear Regression. 


X_train, X_test, y_train, y_test = train_test_split(MPhones.iloc[:,0:19], MPhones['price_range'], random_state=42, test_size=0.33)

LR = LinearRegression()
LR.fit(X_train, y_train)
Prediction_LR = LR.predict(X_test)
print(r2_score(y_test, Prediction_LR))

#The R^2 measure is 63%, which is not a perfect score, so I am using polynomial regression instead. 
#This is a type of regression that uses a 'curved' regression line. 


Poly = PolynomialFeatures()
Poly_final = Poly.fit_transform(X_train)
Poly_predict = Poly.fit_transform(X_test)
Poly_predict_Final = Poly.predict(Poly_predict)
print(r2_score(y_test, Poly_predict_Final))


#The R^2 measure for this model is 70%, which is a good score for predicting new values.






