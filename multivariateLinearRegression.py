import pandas as pd # file walin data ganna udau wenoo, data filter karanna udauweno, model ekata data pass karanna udauweno
from sklearn import preprocessing, model_selection # meya bavitha karanne data lable karanna
from sklearn import linear_model
import sklearn
import numpy as np


###LODE DATA###
print("-"*30);print("IMPORT DATA");print("-"*30);
data = pd.read_csv('houses_to_rent.csv', sep = ',')
data = data[['city', 'rooms', 'bathroom', 'parking spaces','fire insurance', 'furniture','rent amount']]
#print(data.head())  #data set eke mull 5 print karala pennai

### PROCESS DATA ###
############***** maching lerning model walata string theren nee number vitharai therenne ****#############
#************data wala " , " thiyennath bee
#*********** data wala type eka integer wenath ona*********

# rent amount kiayanne str ekakan mula RS thiyano meda "," thiyanoo, siyalla ewathkara integer ekaka karana ayuru
data['rent amount'] = data['rent amount'].map(lambda i: int(i[2:].replace(',','')))
data['fire insurance'] = data['fire insurance'].map(lambda i: int(i[2:].replace(',','')))


# samahara data sampuraba string, ewata ankayak labal gesiya yuthuya
le = preprocessing.LabelEncoder()
data['furniture'] = le.fit_transform((data['furniture'] ))
print(data.head())

print("-"*30);print("CHECKING NULL DATA");print("-"*30);
# null Data thiyanawada belima
print(data.isnull().sum())# kopamana null data sankawawak thibeda
print("-"*30);print("HEAD");print("-"*30);

##### SPLT DATA ####
print("-"*30);print("SPLIT DATA");print("-"*30)
#data set eka train saha testing data walata wenkirima
x = np.array(data.drop(['rent amount'],1))
y = np.array(data['rent amount'])
print('X',x.shape)
print('Y',y.shape)
xTrain, xTest, yTrain, yTest = sklearn.model_selection.train_test_split(x,y, test_size=0.2,random_state=10)

print('XTrain',xTrain.shape)
print('XTest',xTest.shape)

##### TRAINING ####
print("-"*30);print("TRAINING");print("-"*30)
#model eka train karanna hadanne
model = linear_model.LinearRegression()
model.fit(xTrain,yTrain)
accuracy = model.score(xTest,yTest)
print('Cofficients :',model.coef_)# anucramana pramanaya
print('intercept :', model.intercept_) # anucramanaya
print('Accuracy :',round(accuracy*100,3), '%')


#### EVALUATION ####
print("-"*30);print("TRAINING");print("-"*30)
# den karanna yanne api train karapu model eka test karana eka
testVals = model.predict(xTest)
print(testVals.shape)
error = []
# ape model ekata testing valu dala ena agayai, prectical apu agayai athara wenasa ganno
for i,testVal in enumerate(testVals):
    error.append(yTest[i]-testVal)
    print(f'Acutal :{yTest[i]} Prediction : {int(testVal)} Error:{int(error[i])}')













