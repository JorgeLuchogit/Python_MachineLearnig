"""
Programa que simula un conjunto de datos de decisión a los cuales se les ajustará una curva sigmoide.
"""

from pandas import DataFrame
from numpy.random import randint
from math import exp

def simulacion_desicion_binaria(num_mues, salto=0.5):
    """
    num_muest: numero de datos que se quieren obtener.
    salto:     Punto en donde se da el salto de los datos. Por defecto se tiene 0.5, por lo que el salto dse da a la mitad de la distribución.
    """
    if salto <= 0 or salto >= 1:

        print('Fraccion no valida. Coloque un número entre 0 y 1')
        exit()

    decision = randint(0,11, num_mues)

    for i in range(num_mues):

        if i < num_mues*salto:
            if decision[i] == 1: 
                decision[i] = 1
            else:
                decision[i] = 0
        else:
            if decision[i] != 1 : 
                decision[i] = 1
            else:
                decision[i] = 0

    return DataFrame(decision)


