"""
Programa para realizar una regresión lineal multivariable a partir de datos SIN INCERTIDUMBRE. Solo se requiere cargar los datos y generar un data frame a partir de ellos
"""

from pandas import DataFrame
import matplotlib.pyplot as plt
from numpy import linspace, meshgrid
from sklearn.linear_model import LinearRegression

#---------------------------------------------------------
# AUXILIARES
#---------------------------------------------------------

empty = DataFrame([])

#---------------------------------------------------------
# FUNCION QUE HACE EL AJUSTE
#---------------------------------------------------------

def Ajuste_lineal_multi(df, df_p, tit, y_exp=empty):

    """
    df:    datos al que se le realiza el ajuste. Tiene que ser tal que la primera columna es la de la variable dependiente. 
    df_p:  datos a los cuales se les quiere predecir el valor de la variable independiente. 
    y_exp: valores "experimentales" de la variable dependiente del conjunto de puntos df_p.
    """
    empty = y_exp.empty 

    # Se obtienen los nombres de las columnas 
    columnas = df.columns[1:]
    columna_dep = df.columns[0]

    #Se define el modelo
    regresion = LinearRegression()
    regresion.fit(df[columnas].values, df[[columna_dep]].values)

    w = regresion.coef_[0]
    b = regresion.intercept_[0]

    # Presicion del modelo:
    if empty == False :
        presicion = regresion.score(df_p.values, y_exp.values)*100
    presicion = 'No se puede calcular la presicion'

    # Predicciones:
    y_p = regresion.predict(df_p[columnas].values)
    df_p[columna_dep] = y_p

    # Se imprime la información obtenida:
    print(f'\ncoeficientes:\n{w}\n\nOrdenada al origen:\n {b}')
    print('\nPredicciones:\n', df_p.set_index(columna_dep))
    if empty == False:
        print(f'\nLa presicion del ajuste es del: {presicion}%' )

 
    # Cuando se tienen 3 dimensiones:
    if len(columnas) == 2:
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        x_aux = df[columnas[0]]
        y_aux = df[columnas[1]]

        X = linspace(min(x_aux), max(x_aux),5)
        Y = linspace(min(y_aux), max(y_aux),5)
        X,Y = meshgrid(X,Y)
        Z = w[0]*X + w[1]*Y + b

        if empty == False :
            ax.scatter(df_p.iloc[:,[0]],df_p.iloc[:,[1]], y_exp , color='g', label='Prueba')
        ax.scatter(df.iloc[:,[1]],df.iloc[:,[2]], df.iloc[:,[0]], color='r', label='Entrenamiento')
        ax.scatter(df_p.iloc[:,[0]],df_p.iloc[:,[1]], df_p.iloc[:,[2]] , color='b', label='Prueba-prediccion')
        ax.plot_surface(X,Y,Z, color='cyan', alpha=0.5)

        # Estética de la gráfica:
        ax.grid(linestyle='--', color='black', alpha=0.5, lw=0.6)
        ax.set_xlabel(columnas[0], fontsize='15')
        ax.set_ylabel(columnas[1], fontsize='15')
        ax.set_zlabel(columna_dep, fontsize='15')
        ax.set_title(f'{tit}', fontsize='20')
        ax.legend()

        plt.show()

    return w, b, presicion

#---------------------------------------------------------
# DATOS DE PRUEBA
#---------------------------------------------------------

"""
#  Títulos de la gráfica:
titulo = 'Regresion lineal multivariable'

# Datos de prueba
x1_nombre = 'vi1'
x2_nombre = 'vi2'
x3_nombre = 'vi3'
y_nombre ='vd'

x1 = [2600, 3000, 3200, 3600, 4000]
x2 = [3, 4, 3, 3, 5]
x3 = [20, 15, 18, 30, 8]
y = [550000, 565000, 610000, 595000, 760000]

df = DataFrame({y_nombre:y, x1_nombre:x1, x2_nombre:x2})

# Datos a predecir:
x1_p = [3000, 2800, 2500, 2800, 3100]
x2_p = [3, 1, 4, 5, 6]
x3_p = [10, 14, 20, 25, 5]

yexp=DataFrame({'y_exp':[550040, 565050, 610010, 595090, 760010]})

df_p = DataFrame({x1_nombre:x1_p, x2_nombre:x2_p})


# Aplicacion del ajuste a los datos de prueba
Ajuste_lineal_multi(df=df, df_p=df_p, tit=titulo)
"""
