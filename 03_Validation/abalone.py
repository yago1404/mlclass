import math
import pandas as pd
import seaborn as sn
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import requests

from sklearn import svm

from sklearn.naive_bayes import GaussianNB


sex_value = {
    'M': 1,
    'F': 2,
    'I': 3
}

feature_cols = [
    # 'sex',  # 45
    # 'length',  # 55
    # 'diameter',  # 55
    'height',  # 55
    'whole_weight',  # 59
    # 'shucked_weight',  # 52
    # 'viscera_weight',  # 55
    'shell_weight'  # 60
]

data = pd.read_csv('abalone_dataset.csv')

# position = 0
# for current in data['sex']:
#     data['sex'][position] = sex_value[data['sex'][position]]
#     position = position + 1

# plot correlation
# correlation = data.corr(min_periods=1, numeric_only=True)
# correlation.plot()
# plt.show()


X = data[feature_cols][0:2506]
y = data['type'][0:2506]

for i in feature_cols:
    position = 0
    for j in data[i].values:
        if i == "length":
            data[i][position] = round(data[i][position], 3)
        if i == "whole_weight" or i == 'shell_weight':
            data[i][position] = round(data[i][position], 4)
        position = position + 1


model = svm.SVC(C=1)
model.fit(X, y)

score = model.score(data[feature_cols][2506:3133], data['type'][2506:3133])

to_predict = pd.read_csv('abalone_app.csv')

for i in feature_cols:
    position = 0
    for j in to_predict[i].values:
        if i == "length":
            to_predict[i][position] = round(to_predict[i][position], 3)
        if i == "whole_weight" or i == 'shell_weight':
            to_predict[i][position] = round(to_predict[i][position], 4)
        position = position + 1

predictions = model.predict(to_predict[feature_cols])
print(predictions)
print(score)

URL = "https://aydanomachado.com/mlclass/03_Validation.php"
#
DEV_KEY = "Equipe Bayes"
#
json_data = {'dev_key': DEV_KEY, 'predictions': pd.Series(predictions).to_json(orient='values')}
#
# Enviando requisição e salvando o objeto resposta
r = requests.post(url=URL, data=json_data)
# Extraindo e imprimindo o texto da resposta
pastebin_url = r.text
print(" - Resposta do servidor:\n", r.text, "\n")


