"""
Programa para realizar una regresión lineal multivariable a partir de datos SIN INCERTIDUMBRE. Solo se requiere cargar los datos y generar un data frame a partir de ellos
"""

from numpy import linspace, meshgrid, array
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn import linear_model as lm


#---------------------------------------------------------
# ENTRADA DE LOS DATOS 
#---------------------------------------------------------

#  Títulos de la gráfica:
nombre_ejex = 'vd1'
nombre_ejey = 'vd2'
nombre_ejez = 'vi'
titulo = 'Regresion lineal multivariable'

# Inputs de nombres:
x1_nombre = 'vd1'
x2_nombre = 'vd2'
x3_nombre = 'vd3'
y_nombre ='vi'

# Datos de prueba
x1 = [2600, 3000, 3200, 3600, 4000]
x2 = [3, 4, 3, 3, 5]
x3 = [20, 15, 18, 30, 8]
y = [550000, 565000, 610000, 595000, 760000]

# Datos a predecir:
x1_p = [3000, 2800, 2500, 2800, 3100]
x2_p = [3, 1, 4, 5, 6]
x3_p = [10, 14, 20, 25, 5]


df = DataFrame({y_nombre:y, x1_nombre:x1, x2_nombre:x2})
columnas = df.columns[1:]

df_p = DataFrame({x1_nombre:x1, x2_nombre:x2_p})


#---------------------------------------------------------
# SE HACE EL AJUSTE
#---------------------------------------------------------

regresion = lm.LinearRegression()
regresion.fit(df[columnas].values, df[[y_nombre]].values)

w = regresion.coef_[0]
b = regresion.intercept_[0]

# Predicciones:
y_p = regresion.predict(df_p[columnas].values)
df_p[y_nombre] = y_p

# Se imprime la información obtenida:
print(f'\ncoeficientes:\n{w}\n\nOrdenada al origen:\n {b}')
print('\nPredicciones:\n', df_p.set_index(y_nombre))

# Cuando se tienen 3 dimensiones:
if len(columnas) == 2:
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    x_aux = df[columnas[0]]
    y_aux = df[columnas[1]]

    X = linspace(min(x_aux), max(x_aux),5)
    Y = linspace(min(y_aux), max(y_aux),5)
    X,Y = meshgrid(X,Y)
    Z = w[0]*X + w[1]*Y + b


    ax.scatter(df.iloc[:,[1]],df.iloc[:,[2]], y, color='r', label='Datos')
    ax.scatter(df_p.iloc[:,[0]],df_p.iloc[:,[1]], df_p.iloc[:,[2]] , color='b', label='Predicciones')
    ax.plot_surface(X,Y,Z, color='cyan', alpha=0.5)

    # Estética de la gráfica:
    ax.grid(linestyle='--', color='black', alpha=0.5, lw=0.6)
    ax.set_xlabel(f'{nombre_ejex}', fontsize='15')
    ax.set_ylabel(f'{nombre_ejey}', fontsize='15')
    ax.set_zlabel(f'{nombre_ejez}', fontsize='15')
    ax.set_title(f'{titulo}', fontsize='20')
    ax.legend()

    plt.show()

