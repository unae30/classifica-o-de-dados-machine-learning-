from ucimlrepo import fetch_ucirepo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
  
# fetch dataset 
spambase = fetch_ucirepo(id=94) #Utilização da função "fetch..."para baixar o dataset de ID 94
  
# data (as pandas dataframes) 
X = spambase.data.features # Objetos que são retornados contentando os dados 
y = spambase.data.targets 
  
# metadata 
#print(spambase.metadata) 
  
# variable information 
#print(spambase.variables) 
#-----------------------------------------------------------------------------------------------------------
print(X.describe())
print ("----------------------------------------------------------------")
print (y.value_counts())