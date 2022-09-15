#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Atividade para trabalhar o pré-processamento dos dados.

Criação de modelo preditivo para diabetes e envio para verificação de peformance
no servidor.

@author: Aydano Machado <aydano.machado@gmail.com>
"""

"""
(1) Substituindo os NaN por 0 => 55%

(2) 1 + Removendo SkinThickness => 57%

(3) Removendo SkinThickness + substituindo NaN pela moda => 57%

(4) Removendo SkinThickness + substituindo NaN pela media => 58%

(5) Removendo SkinThickness + normalizando DiabetesPedigreeFunction + substituindo NaN pela moda
 => 58%

(6) Removendo SkinThickness + normalizando DiabetesPedigreeFunction e Glucose e Insulina + utilizando a moda => 55%

(7) Removendo colunas de alta corelacao e normalizando os dados pela media => 60%
"""

import math
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import requests

print('\n - Lendo o arquivo com o dataset sobre diabetes')
data = pd.read_csv('diabetes_dataset.csv')

# Criando X and y par ao algorítmo de aprendizagem de máquina.\
print(' - Criando X e y para o algoritmo de aprendizagem a partir do arquivo diabetes_dataset')
# Caso queira modificar as colunas consideradas basta algera o array a seguir.
feature_cols = ['Pregnancies', 'Glucose', 'BMI', 'DiabetesPedigreeFunction', 'Age']

medias = []
for i in range(len(feature_cols)):
    medias.append(np.nanmean(data[feature_cols[i]]))

# Cleaning NaN cells
for i in feature_cols:
    position = 0
    for j in data[i].values:
        if math.isnan(j):
            data[i][position] = medias[feature_cols.index(i)]
        if i == "DiabetesPedigreeFunction":
            data[i][position] = round(data[i][position], 3)
        position = position + 1

X = data[feature_cols]
y = data.Outcome

# Ciando o modelo preditivo para a base trabalhada
print(' - Criando modelo preditivo')
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, y)

#realizando previsões com o arquivo de
print(' - Aplicando modelo e enviando para o servidor')
data_app = pd.read_csv('diabetes_app.csv')
data_app = data_app[feature_cols]
y_pred = neigh.predict(data_app)

# Enviando previsões realizadas com o modelo para o servidor
URL = "https://aydanomachado.com/mlclass/01_Preprocessing.php"

# #TODO Substituir pela sua chave aqui
DEV_KEY = "Equipe Bayes"

# json para ser enviado para o servidor
data = {'dev_key':DEV_KEY,
        'predictions':pd.Series(y_pred).to_json(orient='values')}

# Enviando requisição e salvando o objeto resposta
r = requests.post(url = URL, data = data)

# Extraindo e imprimindo o texto da resposta
pastebin_url = r.text
print(" - Resposta do servidor:\n", r.text, "\n")