# Heart-Disease
Dataset: https://github.com/vaezmasoud/Heart-Disease/blob/main/heart_cleveland.csv

feature:
1.	age: age in years
2.	sex: sex (1 = male; 0 = female)
3.	cp: chest pain type
-- Value 0: typical angina
-- Value 1: atypical angina
-- Value 2: non-anginal pain
-- Value 3: asymptomatic
4.	trestbps: resting blood pressure (in mm Hg on admission to the hospital)
5.	chol: serum cholestoral in mg/dl
6.	fbs: (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
7.	restecg: resting electrocardiographic results
-- Value 0: normal
-- Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
-- Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria
8.	thalach: maximum heart rate achieved
9.	exang: exercise induced angina (1 = yes; 0 = no)
10.	oldpeak = ST depression induced by exercise relative to rest
11.	slope: the slope of the peak exercise ST segment
-- Value 0: upsloping
-- Value 1: flat
-- Value 2: downsloping
12.	ca: number of major vessels (0-3) colored by flourosopy
13.	thal: 0 = normal; 1 = fixed defect; 2 = reversable defect
and the label
14.	condition: 0 = no disease, 1 = disease
----------------------------------
Heart Disease Prediction with
+ MLP Classifier - Accuracy: 0.5833333333333334 / AUC: 0.7723214285714286
+ Naive Bayes Gaussian - Accuracy: 0.7666666666666667 / AUC: 0.8415178571428572
+ Navie Bayes Multinomical - Accuracy: 0.6 / AUC: 0.7399553571428571
+ DecisionTree - Accuracy: 0.7333333333333333 / AUC: 0.734375
