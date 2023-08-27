#-------------------------------------------------------------------------------
#Cargando librerias#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#Preprocesamiento de los datos (diferente para cada dataframe)
df = pd.read_csv("TARP.csv") #Cargamos los datos
'''Anteroriormente ya había abierto en excel los datos y se que algunas features
no estan completas, por lo que eliminare los registros que no tengan valor en
todos los feature'''
'''Buscamos el primer indice que no contiene todos los feature, sin embargo
la función devuelve 0 si estan completos, por lo que no se tomara en cuenta'''
first_na_indices = df.apply(lambda col: col.isna().idxmax())
'''Quitamos los ceros y los repetidos'''
IDs_FnaValues = list(set(first_na_indices) - {0})
df= df.iloc[:min(IDs_FnaValues)]  #Filtramos los registros completos
'''El label de clasificacion se define como ON y OFF de acuerdo al crecimiento,
por lo que remplazaremos como O y 1 '''
#-------------------------------------------------------------------------------
#Procesamiento de datos (igual para los diferentes dataframe)
df["Status"] = df["Status"].replace({"ON": 1, "OFF": 0})
'''Ahora solo faltaría estandarizar los valores para acelerar la convergencia
y mejorar el desempeño del modelo'''
def Estandarizacion(x):
  X_estandarizado = (x-np.mean(x))/np.std(x)
  return X_estandarizado
features = df.columns.difference(["Status"])
df[features] = df[features].apply(lambda col: Estandarizacion(col))
#-------------------------------------------------------------------------------
#Seperación de los datos
'''Vamos a separar los datos en test y train y respectivamente cada uno features
y labels'''
def SepTrainTest(df,LabelName,Percent_Train):
  TrainSize = int(len(df)*Percent_Train/100) #Porcentaje de entrenamiento
  TestSize = len(df)-TrainSize #Porcentaje de prueba
  Sep = np.random.choice(range(0, len(df)), size=len(df), replace=False)
  TrainIds = Sep[:TrainSize] #Asignación random de identificadores entrenamiento
  TestIds = Sep[-TestSize:] #Asignación random de identificadores  de prueba
  features = df.columns.difference([LabelName]) #Columnas de features
  df_Train_X = df[features].loc[TrainIds] #Features entrenamiento
  df_Test_X = df[features].loc[TestIds] #Features prueba
  df_Train_y = df[LabelName].loc[TrainIds] #Label entrenamiento
  df_Test_y = df[LabelName].loc[TestIds] #Label prueba
  return df_Train_X, df_Test_X, df_Train_y, df_Test_y
Train_X, Test_X, Train_y, Test_y = SepTrainTest(df,"Status",80)
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#Entrenamiento del modelo
'''Como es un problema de clasificación, utilizaré un modelo de regresión
logistica con la función Sigmoide'''
def sigmoid_function(X):
  return 1/(1+np.exp(-X)) #Definiendo la función
'''Ahora definimos los valores iniciales de los pesos theta'''
#theta0 = np.random.randn(len(Train_X.columns) + 1, 1)
theta0 = np.zeros((len(Train_X.columns) + 1, 1)) #Misma convergencia
'''Finalmente, definimos la función de regresión logistica para entrenamiento'''
def log_regression_Train(X, y, theta, alpha, epochs):
  X_vect = np.c_[np.ones((len(X), 1)), X] #Agregamos vector de unos para el bias
  X = X.values #Convertimos a matriz para operaciones matriciales
  y = y.values
  N = len(X)
  y_ = np.reshape(y, (len(y), 1)) #Trasponemos 
  for epoch in range(epochs):
    sigmoid_x_theta = sigmoid_function(X_vect.dot(theta)) #Producto matricial
    grad = (1/N) * X_vect.T.dot(sigmoid_x_theta - y_) #Gradiente
    theta = theta - (alpha * grad) #Actualizamos theta
    hyp = sigmoid_function(X_vect.dot(theta)) # Calculamos la función logistica
    return theta #Retornamos los mejores parametros
newTheta = log_regression_Train(Train_X, Train_y, theta0, 1, 200000)
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#Probamos el modelo#
'''El paso final es probar el modelo, sabiendo que para cada valor que arroje, 
se clasificara como 1 si es mayor a 0.5 y 0 si es menor. Como herramienta de 
medición de error utilizare la matriz de confusión y la precisión'''
#Función para matriz de confusión
def MatrizConfusion(X_real, X_predict):
  TP = 0; #TruePositive
  TN = 0; #TrueNegative 
  FP = 0; #FalsePositive
  FN = 0; #FalseNegative
  for i in range(len(X_real)):
    if (X_real[i] == 1 and X_predict[i] == 1):
      TP = TP + 1
    elif (X_real[i] == 0 and X_predict[i] == 0):
      TN = TN + 1
    elif (X_real[i] == 0 and X_predict[i] == 1):
      FP = FP + 1
    else:
      FN = FN + 1
  MC = [[TP,FP],[FN,TN]]
  return MC
#-------------------------------------------------------------------------------
#Función de prueba
def transformar_valor(valor): #Criterio de clasificación
    if valor > 0.5:
        return 1
    else:
        return 0
'''La siguiente función realiza lo mismo que la función de entrenamiento pero solo
una vez para los valores de entrenamiento y retorna la matriz de confusión'''
def log_regression_Test(X, y, theta): 
  X_vect = np.c_[np.ones((len(X), 1)), X]
  X = X.values
  y = y.values
  y_ = np.reshape(y, (len(y), 1)) # shape (150,1)
  N = len(X)
  sigmoid_x_theta = pd.DataFrame(sigmoid_function(X_vect.dot(theta)), columns=['y_testing'])
  #Lo hacemos dataframe para usar applymap
  y_predict = sigmoid_x_theta.applymap(transformar_valor)
  Results = MatrizConfusion(y,y_predict.values)
  return Results
#-------------------------------------------------------------------------------
#Prueba del modelo
Test_Results = log_regression_Test(Test_X, Test_y, newTheta)
#-------------------------------------------------------------------------------
#Resultados del modelo
matriz_confusion = [list(row) for row in Test_Results]
'''Calculamos la precisión'''
vp = matriz_confusion[0][0]
fp = matriz_confusion[0][1]
fn = matriz_confusion[1][0]
vn = matriz_confusion[1][1]
precision = (vp+vn) / (vp + fp +vn +fn)
'''Finalmente solo mostramos la matriz de confusión'''
plt.matshow(matriz_confusion, cmap='Blues')
# Agregar los valores de la matriz como texto en las celdas
for i in range(len(matriz_confusion)):
    for j in range(len(matriz_confusion[i])):
        plt.text(j, i, str(matriz_confusion[i][j]), ha='center', va='center', color='black')
plt.colorbar()  # Agregar una barra de color para referencia
titulo = f"Database 1: (14x2100): Matriz de Confusión\nPrecisión: {precision:.4f}"
plt.title(titulo)
plt.xlabel("Valores Predichos")
plt.ylabel("Valores Reales")
plt.show()

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------


"""Un segundo aprovechamiento para los datos, sería eliminar los feature para los
cuales los registros solo completan hasta 2100. Con ellos se tendrían ahora 8913 
registros con lo cual podria tener mejores resultados"""
df = pd.read_csv("TARP.csv") #Cargamos los datos
features = df.columns
df = df.drop(features[len(features)-6:len(features)-1], axis=1)
first_na_indices = df.apply(lambda col: col.isna().idxmax())
IDs_FnaValues = list(set(first_na_indices) - {0})
df= df.iloc[:min(IDs_FnaValues)]  #Filtramos los registros completos
#-------------------------------------------------------------------------------
#Procesamiento de datos (igual para los diferentes dataframe)
df["Status"] = df["Status"].replace({"ON": 1, "OFF": 0})
'''Ahora solo faltaría estandarizar los valores para acelerar la convergencia
y mejorar el desempeño del modelo'''
features = df.columns.difference(["Status"])
df[features] = df[features].apply(lambda col: Estandarizacion(col))
#-------------------------------------------------------------------------------
#Seperación de los datos
Train_X, Test_X, Train_y, Test_y = SepTrainTest(df,"Status",80)
#-------------------------------------------------------------------------------
#Entrenamiento del modelo
'''Ahora definimos los valores iniciales de los pesos theta'''
#theta0 = np.random.randn(len(Train_X.columns) + 1, 1)
theta0 = np.zeros((len(Train_X.columns) + 1, 1)) #Misma convergencia
'''Finalmente, definimos la función de regresión logistica para entrenamiento'''
#Me robe el codigo de los del profe
newTheta = log_regression_Train(Train_X, Train_y, theta0, 1, 200000)
#-------------------------------------------------------------------------------
#Prueba del modelo
Test_Results = log_regression_Test(Test_X, Test_y, newTheta)
#-------------------------------------------------------------------------------
#Resultados del modelo
matriz_confusion = [list(row) for row in Test_Results]
'''Calculamos la precisión'''
vp = matriz_confusion[0][0]
fp = matriz_confusion[0][1]
fn = matriz_confusion[1][0]
vn = matriz_confusion[1][1]
precision = (vp+vn) / (vp + fp +vn +fn)
'''Finalmente solo mostramos la matriz de confusión'''
plt.matshow(matriz_confusion, cmap='Blues')
# Agregar los valores de la matriz como texto en las celdas
for i in range(len(matriz_confusion)):
    for j in range(len(matriz_confusion[i])):
        plt.text(j, i, str(matriz_confusion[i][j]), ha='center', va='center', color='black')
plt.colorbar()  # Agregar una barra de color para referencia
titulo = f"Database 2 (9x8913) : Matriz de Confusión\nPrecisión: {precision:.4f}"
plt.title(titulo)
plt.xlabel("Valores Predichos")
plt.ylabel("Valores Reales")
plt.show()
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------



"""Un tercer aprovechamiento sería utilizar solo los feature que tengan registros
completos. Con ello tendria un total de 100,000"""
df = pd.read_csv("TARP.csv") #Cargamos los datos otra vez
features = df.columns
df = df.drop(features[len(features)-11:len(features)-1], axis=1)
#-------------------------------------------------------------------------------
#Procesamiento de datos (igual para los diferentes dataframe)
df["Status"] = df["Status"].replace({"ON": 1, "OFF": 0})
'''Ahora solo faltaría estandarizar los valores para acelerar la convergencia
y mejorar el desempeño del modelo'''
features = df.columns.difference(["Status"])
df[features] = df[features].apply(lambda col: Estandarizacion(col))
#-------------------------------------------------------------------------------
#Seperación de los datos
"Vamos a separar los datos "
Train_X, Test_X, Train_y, Test_y = SepTrainTest(df,"Status",80)
#-------------------------------------------------------------------------------
#Entrenamiento del modelo
'''Ahora definimos los valores iniciales de los pesos theta'''
#theta0 = np.random.randn(len(Train_X.columns) + 1, 1)
theta0 = np.zeros((len(Train_X.columns) + 1, 1)) #Misma convergencia
'''Finalmente, definimos la función de regresión logistica para entrenamiento'''
#Me robe el codigo de los del profe
newTheta = log_regression_Train(Train_X, Train_y, theta0, 1, 200000)
#-------------------------------------------------------------------------------
#Prueba del modelo
Test_Results = log_regression_Test(Test_X, Test_y, newTheta)
#-------------------------------------------------------------------------------
#Resultados del modelo
matriz_confusion = [list(row) for row in Test_Results]
'''Calculamos la precisión'''
vp = matriz_confusion[0][0]
fp = matriz_confusion[0][1]
fn = matriz_confusion[1][0]
vn = matriz_confusion[1][1]
precision = (vp+vn) / (vp + fp +vn +fn)
'''Finalmente solo mostramos la matriz de confusión'''
plt.matshow(matriz_confusion, cmap='Blues')
# Agregar los valores de la matriz como texto en las celdas
for i in range(len(matriz_confusion)):
    for j in range(len(matriz_confusion[i])):
        plt.text(j, i, str(matriz_confusion[i][j]), ha='center', va='center', color='black')
plt.colorbar()  # Agregar una barra de color para referencia
titulo = f"Database 3 (4x8913) : Matriz de Confusión\nPrecisión: {precision:.4f}"
plt.title(titulo)
plt.xlabel("Valores Predichos")
plt.ylabel("Valores Reales")
plt.show()

'''En el codigo mostrado, se resuelvo un problema de clasificación de estado de crecimiento 
de plantas, basado en factores que afectan su crecimiento. Tengo originalmente 14 feature
y 100,000 registros, sin embargo, no todos los feature tienen sus registros completos, siendo
5 que solo tienen 2100 registros, otros 5 que tienen poco más de 8900 y 4 que tienen los 
registros completos. Realice 3 modelos, considerando 4, 9 y 14 feature.

Lo primero que realice fue un cambio a los valores del label a 0 y 1 de acuerdo a su estado
de crecimiento. Despues estandarice los datos para acelerar convergencia y mejorar su desem
peño. Luego separe los datos en train y test de forma aleatoria asi como su label.
Posteriormente entrene un modelo de regresión logistica para un total de 200,000 epochs y 
una taza de aprendizaje de 1. Inicialice los valores de thetas en 0. Inmediatamente, probe 
el modelo y utilice como metrica de error la matriz de confusión y la precisión.

A pesar de la cantidad tan diferente de registros en los 3 aprovechamientos de los datos, no
encontre un cambio considerable en la precisión del modelo. En los 3 halle una precisión de
70(+-)5%, lo cual consideraría buena, si no es porque se trata de un problema de clasificación
binaria, que podría tener una precisión de 50% con solo clasificar todos con el mismo valor.
Para proximas entregas consideraría usar otra función de clasificación y limpiar la base de datos '''