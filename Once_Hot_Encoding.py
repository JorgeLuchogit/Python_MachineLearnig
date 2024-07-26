"""
el Once Hot Encoding sirve para tratar casos en la que la variable principal depende de una variable no numérica. Por ejemplo, el precio de una casa en función de la región en la que se ubica.

El tratamiento consiste en considerar cada valor de la variable no numerica como una nueva variable (dummy variable) que toma el valor 1 o 0 según corresponda el resto de variables numéricas independientes. 

el conjunto de dummy variables es linealmente dependiente por lo que una de estas variables debe eliminarse. Por ejemplo si se tienen 3 variables es equivalente (0,0) a (0,0,1).

Este programa realiza un proceso de Once Hot Encoding usando Pandas. El data frame con los datos debe ser tal, que la primera columna es la variable dependiente, seguida de las columnas de las variables no numéricas y finalmente las columnas de las variables independientes numéricas. 
"""

import pandas as pd 
from sklearn.linear_model import LinearRegression

#from sklearn import linear_model

#-----------------------------------------------------------------
# ENTRADA DE DATOS
#-----------------------------------------------------------------

# Datos de prueba:
dependiente = [550000, 565000, 610000, 680000, 725000, 585000, 615000, 650000, 710000, 575000, 600000, 620000, 695000]
numerica_in = [2600, 3000, 3200, 3600, 4000, 2600, 2800, 3300, 3600, 2600, 2900, 3100, 3600, ]
no_numeric  = ['val_1', 'val_1', 'val_1', 'val_1', 'val_1', 'val_2', 'val_2', 'val_2', 'val_2', 'val_3', 'val_3', 'val_3', 'val_3']

df = pd.DataFrame({'dependiente':dependiente, 'No_numerica':no_numeric, 'numerica_in':numerica_in})

# Datos a predecir:


# Número de variables no numéricas:
no_numeric_cantidad = 1

#-----------------------------------------------------------------
# MANEJO DE LOS DATOS
#-----------------------------------------------------------------

# Nombres de las columnas no numéricas:
columnas_no_numeric = df.columns[1:no_numeric_cantidad+1]

# Obtención de dummy variables:
for i in range(no_numeric_cantidad):
    df_dummy = pd.get_dummies(df[columnas_no_numeric[i]])

    # Se agregan al data frame original:
    df = pd.concat([df, df_dummy], axis='columns')

# Se quitan las variables no numericas:
df = df.drop(columnas_no_numeric, axis='columns')

# Se quita la última dummmy variable:
columna_final = df.columns[-1]
df = df.drop(columna_final, axis='columns')

print('\n Data frame  limpio\n', df)

#-----------------------------------------------------------------
# APLICACION DEL AJUSTE MULTIVARIABLE
#-----------------------------------------------------------------

x = df.drop(df.columns[0], axis='columns')
y = df.drop(df.columns[1:], axis='columns')

modelo = LinearRegression()

modelo.fit(x,y)
presicion_ajuste = modelo.score(x,y)

w = modelo.coef_[0]
b = modelo.intercept_[0]

print(f'\ncoeficientes:\n{w}\n\nOrdenada al origen:\n {b}\n\n Presicion del ajuste: {presicion_ajuste}')