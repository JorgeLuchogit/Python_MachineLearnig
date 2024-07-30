"""
En el entrenamiento de un modelo a partir de un conjunto de datos de prueba, a menudo dicho conjunto se divide en dos grupos, uno grande y uno pequeño. El grupo grande se utiliza para entrenar al algoritmo (grupo de entrenamiento), en el caso de las regresiones lineales, es al grupo al que se le ajusta la recta (o el plano). El resto se utiliza para comparar sus valores dependientes (obtenidos experimentalemente) con los valores dependientes obtenidos con el ajuste al evaluar en los valores independientes de dicho grupo de datos (grupo de prueba). 

El presente programa es una rutina que, dado un conjunto de datos arbitrario, se selecciona el grupo de prueba y el grupo de entrenamiento. El data frame con la informacion debe ser tal que la primera columna es la correspondiente a la variable dependiente.

Para el ejemplo del final se requiere contar con el modulo Regresion_Lineal_Multivariable que se encuentra en el mismo repositorio 
"""

from numpy import array
from numpy.random import normal
from pandas import DataFrame, concat
from sklearn.model_selection import train_test_split
from Regresion_Lineal_Multivariable import Ajuste_lineal_multi as ajlm

#--------------------------------------------------------------
# INPUT DATOS
#--------------------------------------------------------------

# Se generan datos de prueba:

cantidad_datos = 100
v_d = array([i for i in range (cantidad_datos)])
vi1 = v_d + normal(loc=0, scale=5, size=cantidad_datos)
vi2 = 3*v_d + normal(loc=0, scale=10, size=cantidad_datos)

# Datos de prueba:
df = DataFrame({'var_dep':v_d, 'var_i1':vi1, 'var_i2':vi2})

#--------------------------------------------------------------
# OBTENCION DE LOS GRUPOS
#--------------------------------------------------------------

# Se separan las variables dependientes e independientes:
y = df[df.columns[0]]
x = df[df.columns[1:]]

# Se obtiene el grupo de entrenamiento y el de prueba (este últino del 20%):
x_en, x_pru, y_en, y_pru = train_test_split(x, y, test_size=0.2)

"""Nota: Los datos se eligen aleatoriamente, si se desea siempre obtener el mismo resultado, se debe colocar en train_test_split el argumento random_state=n, con n una semilla de números aleatorios"""

# Se ordenan los datos:
df_en = concat([y_en, x_en], axis='columns')
df_pru = concat([y_pru, x_pru], axis='columns')

print(f'Grupo de entrenamiento ({len(x_en)} muestras) :\n', df_en )
print(f'Grupo de prueba ({len(x_pru)} muestras):\n', df_pru)

# Como ejemplo se aplica una regresión lineal
ajlm(df=df_en, df_p=df_pru.iloc[:,1:], tit='Entrenamiento y prueba\n regresión lineal', y_exp=df_pru.iloc[:,0])




