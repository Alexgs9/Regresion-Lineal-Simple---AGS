# -*- coding: utf-8 -*-

#Hecho por Alexandro Gutierrez Serna

import pandas as pd
import matplotlib.pyplot as plt

# Lee el archivo CSV en un DataFrame de pandas
df = pd.read_csv('Salary_dataset.csv')

# YearsExperience = x
# Salary = y

#Toma todos los valores de la primera columna
x_df = df[['YearsExperience']]
x = x_df.values.reshape(-1, 1)
y = df[['Salary']]
y = y.values.reshape(-1, 1)

print("Equis vale:", x)
print("Ye vale:", y)

# Funcion para calcular la pendiente y la ordenada al origen
def regresion_lineal(x, y):
    n = len(x)
    x_prom = x.mean()
    y_prom = y.mean()
    xy = (x*y).sum()
    xx = (x**2).sum()
    m = (xy - n*x_prom*y_prom)/(xx - n*x_prom**2)
    b = y_prom - m*x_prom
    #print("Pendiente (m):", m)
    #print("Ordenada (b):", b)
    return m, b

# Función para predecir valores de y (variable dependiente)
def predict(x, m, b):
    y_pred = []
    for i in x:
        # Para cada conjunto de características (fila en X), calcula la predicción
        y_pred.append(m * i[0] + b)
    return y_pred

pendiente, ordenada = regresion_lineal(x, y)
print("Pendiente:", pendiente)
print("Ordenada:", ordenada)

# Calcular las predicciones
y_pred = predict(x, pendiente, ordenada)

# Imprimir las predicciones
print("Predicciones:", y_pred)

# Visualizar los resultados
plt.scatter(x, y, color='blue', label='Datos de reales')
plt.scatter(x, y_pred, color='orange', label='Datos de predicción')
plt.plot(x, y_pred, color='red', linewidth=2)
plt.title('Regresión Lineal Simple' + '\n' + 'Pendiente: ' + str(pendiente) + ', Ordenada: ' + str(ordenada))
plt.xlabel('Variable independiente (X)')
plt.ylabel('Variable dependiente (y)')
plt.legend()
plt.show()