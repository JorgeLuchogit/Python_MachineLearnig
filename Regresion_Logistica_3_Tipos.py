"""
Programa que hace una regresión logísta a un conjunto de datos de clasificación binaria. El programa requiere que las categorias sean 0 y 1. Se realizan 3 ajustes, uno sin considerar incertidumbres (ajuste s), otro considerando la incertidumbre en x (ajuste i) y otro considerando incertidumbres tanto en x como en y (ajuste o).  
"""

from numpy import sqrt as sqrt, diag, linspace, array, exp
from numpy.random import normal, randint
from scipy.odr import Model, Data, ODR
from pandas import DataFrame, concat
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

"""
#Para el ejemplo
import Simulacion_Decision as sm
"""

#------------------------------------------------------------------------------
# DEFINICIÓN DE FUNCIONES #
#------------------------------------------------------------------------------

# Para el ajuste o:
def recta(A, x):     
   return 1/(1+exp(-A[0]*(x-A[1])))  
#                   C       X0

# Para los ajustes i y s:
def recta_simple(x, C, x0):
   return  1/(1+exp(-C*(x-x0)))

# Ajuste:
def recta_odr(y, dy, x, dx, p0):

   # Pesos
   wy = 1/dy**2
   wx = 1/dx**2

   mod   = Model(recta)
   dat   = Data(x, y, wd=wx, we=wy)
   myodr = ODR(dat, mod, beta0=p0)

   return myodr.run()

# Incertidumbre al evaluar en la recta obtenida:
def recta_incertidumbre(x, A, dA, dx):   
   
   A_0 = A[0]; dA_0 = dA[0]  # C
   A_1 = A[1]; dA_1 = dA[1]  # X0

  
   return sqrt(A_0**2*dA_1**2*exp(-2*A_0*(-A_1 + x))/(1 + exp(-A_0*(-A_1 + x)))**4 + A_0**2*dx**2*exp(-2*A_0*(-A_1 + x))/(1 + exp(-A_0*(-A_1 + x)))**4 + dA_0**2*(A_1 - x)**2*exp(-2*A_0*(-A_1 + x))/(1 + exp(-A_0*(-A_1 + x)))**4)

# Incertidumbre relativa al evaluar en la recta:
def recta_incertidumbre_rel(x, A, dA, dx):   
   
   A_0 = A[0]; dA_0 = dA[0]  # C
   A_1 = A[1]; dA_1 = dA[1]  # x0
 
   return (1 + exp(-A_0*(-A_1 + x)))*sqrt(A_0**2*dA_1**2*exp(-2*A_0*(-A_1 + x))/(1 + exp(-A_0*(-A_1 + x)))**4 + A_0**2*dx**2*exp(-2*A_0*(-A_1 + x))/(1 + exp(-A_0*(-A_1 + x)))**4 + dA_0**2*(A_1 - x)**2*exp(-2*A_0*(-A_1 + x))/(1 + exp(-A_0*(-A_1 + x)))**4)

# Array auxiliar:
no_pred=array([])

def Ajuste_lineal_completo(df, titulo, x_p=no_pred, dx_p=no_pred, tipo_ajuste=[True, True, True], p0=[1,1]):

   """
   df:           dataframe con las columnas x, dx (incertidumbre en x), y, dy, en ese orden.
   titulo:       titulo de la gráfica.
   x_p:          valores en x a predecir. Necesariamente deben ser un np.array.
   dx_p:         incertidumbre de los valores x_p. Necesariamente deben ser un np.array.
   tipo_ajuste:  para elegir si se hacen (True) o no (false) los ajustes o, i, s, en ese orden. 
   p0:           valores iniciales de los coeficientes del ajuste 
   """

   # Título y ejes de las graficas:
   nombre_ejex = df.columns[0]
   nombre_ejey = df.columns[2]
   # Seleccionar tipo de ajuste:
   Incert_xy, Incert_x, No_Incert = tipo_ajuste
   
   # Auxiliares:
   x_fit = linspace(min(df.iloc[:,0]), max(df.iloc[:,0]), 1000, True)
   Lista_df = []

   #------------------------------------------------------------------------------
   # APLICACION, OBTENCION DE INFORMACIÓN Y PREDICCION DE LOS AJUSTES #
   #------------------------------------------------------------------------------

   # Se hace el ajuste o:
   if Incert_xy == True:

      Ajuste_o_info = recta_odr(y=df.y, dy=df.dy, x=df.x, dx=df.dx, p0=p0)
      Ajuste_o_param = Ajuste_o_info.beta[:]                                  
      Ajuste_o_dparam = sqrt(diag(Ajuste_o_info.cov_beta))
      
      # Información del ajuste:
      Lista_df.append(DataFrame({'Ajuste':['Incert-xy'], 'm':[Ajuste_o_param[0]], 'b':[Ajuste_o_param[1]], 'dm':[Ajuste_o_dparam[0]], 'db':[Ajuste_o_dparam[1]]}))
      # Se grafica del ajuste:
      plt.plot(x_fit, recta(Ajuste_o_param, x_fit), color='#04E1FF', lw=2, alpha=0.8, label='Incertidumbre en $x$ y en $y$ (o)')

      # Predicción:
      y_p  = recta(Ajuste_o_param, x_p)
      dy_p = recta_incertidumbre(x=x_p, A=Ajuste_o_param, dA=Ajuste_o_dparam, dx=dx_p)
      # Se grafica de la predicción:
      plt.errorbar(x=x_p, xerr=dx_p, y=y_p, yerr=dy_p, color='#55FF00', alpha=0.8, fmt='x', capsize=3, label='Predicciones (o)')

      # Información de la predicción:
      p_o = DataFrame({'x_p':x_p, 'dx_p':dx_p, 'y_p':y_p, 'dy_p':dy_p})
      print('\nPredicciones con icertidumbres en xy: \n',p_o)


   # Se hace el ajuste i:
   if Incert_x == True: 

      Ajuste_i_param, Ajuste_i_dparam = curve_fit(recta_simple, df.x, df.y, sigma=df.dy, p0=p0)
      Ajuste_i_dparam = sqrt(diag(Ajuste_i_dparam))

      Lista_df.append(DataFrame({'Ajuste':['Incert-x'],  'm':[Ajuste_i_param[0]], 'b':[Ajuste_i_param[1]], 'dm':[Ajuste_i_dparam[0]], 'db':[Ajuste_i_dparam[1]]}))
      plt.plot(x_fit, recta(Ajuste_i_param, x_fit), color='#FF04C2', lw=2, ls='--', alpha=0.8, label='Incertidumbre en $x$ (i)')

      y_p  = recta(Ajuste_i_param, x_p)
      dy_p = recta_incertidumbre(x=x_p, A=Ajuste_i_param, dA=Ajuste_i_dparam, dx=dx_p)
      plt.errorbar(x=x_p, xerr=dx_p, y=y_p, yerr=dy_p, color='#FFD100', alpha=0.8, fmt='^', capsize=3, label='Predicciones (i)')

      p_i = DataFrame({'x_p':x_p, 'dx_p':dx_p, 'y_p':y_p, 'dy_p':dy_p})
      print('\nPredicciones con incertidumbres en y:\n', p_i)


   # Se hace el ajuste s:
   if No_Incert == True:

      Ajuste_s_param, Ajuste_s_dparam = curve_fit(recta_simple, df.x, df.y, p0=p0)
      Ajuste_s_dparam = sqrt(diag(Ajuste_s_dparam))

      Lista_df.append(DataFrame({'Ajuste':['No-incert'], 'm':[Ajuste_s_param[0]], 'b':[Ajuste_s_param[1]], 'dm':[Ajuste_s_dparam[0]], 'db':[Ajuste_s_dparam[1]]}))
      plt.plot(x_fit, recta(Ajuste_s_param, x_fit), color='#6304FF', lw=1.5, alpha=0.8, label='Sin incertidumbre (s)')

      dx_p = array([0 for i in range(len(x_p))])
      y_p  = recta(Ajuste_s_param, x_p)
      dy_p = recta_incertidumbre(x=x_p, A=Ajuste_s_param, dA=Ajuste_s_dparam, dx=dx_p)
      plt.errorbar(x=x_p, y=y_p, yerr=dy_p, color='red', alpha=0.8, fmt='.', capsize=3, label='Predicciones (s)')

      p_s = DataFrame({'x_p':x_p, 'y_p':y_p, 'dy_p':dy_p})
      print('\nPredicciones sin ncertidumbres:\n', p_s)


   # Si no se elige un ajuste:
   if Incert_xy!=True and Incert_x !=True and No_Incert !=True:
      print('Falta elegir ajuste.')
      exit()


   # Se une e imprime la información de los ajustes: 
   info = concat(Lista_df).set_index('Ajuste')
   print('\nInformacion de los ajustes:\n', info)

   # Se grafican puntos experimentales:
   plt.errorbar(x=df.x, xerr=df.dx, y=df.y, yerr=df.dy, color='black', alpha=0.5, fmt='*', capsize=3, label='Datos')

   # Estética de la gráfica
   plt.grid(linestyle='--', color='#A47200', alpha=0.5, lw=0.6)
   plt.xlabel(f'{nombre_ejex}', fontsize='18')
   plt.ylabel(f'{nombre_ejey}', fontsize='18')
   plt.title(f'{titulo}', fontsize='20')
   plt.xticks(fontsize='16')
   plt.yticks(fontsize='16')
   plt.legend()

   #------------------------------------------------------------------------------
   # GRÁFICA DE LA INCERTIDUMBRE #
   #------------------------------------------------------------------------------

   plt.figure()

   if Incert_xy == True:
      plt.plot(x_fit, recta_incertidumbre_rel(x=x_fit, A=Ajuste_o_param, dA=Ajuste_o_dparam, dx=x_fit/10), color='#04E1FF', lw=2, alpha=0.8, label='Incertidumbre (o)')

   if Incert_x == True:
      plt.plot(x_fit, recta_incertidumbre_rel(x=x_fit, A=Ajuste_i_param, dA=Ajuste_i_dparam, dx=x_fit/10), color='#FF04C2', lw=2, ls='--', alpha=0.8, label='Incertidumbre (i)')

   if No_Incert == True:
      plt.plot(x_fit, recta_incertidumbre_rel(x=x_fit, A=Ajuste_s_param, dA=Ajuste_s_dparam, dx=x_fit/10), color='#6304FF', lw=1.5,  alpha=0.8, label='Incertidumbre (s)')

   # Estética de la gráfica de incertidumbre:
   plt.grid(linestyle='--', color='#A47200', alpha=0.5, lw=0.6)
   plt.xlabel(f'{nombre_ejex}', fontsize='18')
   plt.ylabel('Incertidumbre relativa (%)', fontsize='18')
   plt.title('Incertidumbre de los ajustes', fontsize='20')
   plt.xticks(fontsize='16')
   plt.yticks(fontsize='16')
   plt.legend()


   plt.show()

   return info

"""
#------------------------------------------------------------------------------
# EJEMPLO DE APLICACION:
#------------------------------------------------------------------------------

# Datos de prueba:

# Numero de datos:
num_dat = 20

df = sm.simulacion_desicion_binaria(num_dat, 0.4)
ddf = DataFrame([0.01 for i in range(num_dat)])
indices = DataFrame([i for i in range(num_dat)])
dindices = ddf*10
df = concat([indices, dindices, df, ddf], axis='columns')
df.columns=['x', 'dx', 'y', 'dy']

#aplicacion del ajuste
Ajuste_lineal_completo(df=df, titulo='log reg', tipo_ajuste=[True, True, True], p0=[5,num_dat/2])
"""
