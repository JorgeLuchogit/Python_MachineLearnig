"""
Programa que simula un conjunto de datos de decisión a los cuales se les ajustará una curva sigmoide.
"""

from pandas import DataFrame
from numpy.random import randint


def simulacion_desicion_binaria(num_mues):
    """
    num_muest: numero de datos que se quieren obtener.
    """
    decision = randint(0,50,num_mues)

    for i in range(num_mues):
        if decision[i] < i:
            decision[i] = 1
        else:
            decision[i] =0
    return DataFrame(decision)


