"""
Programa que realiza regresiones lineales. Se realizan 3 ajustes, uno sin considerar incertidumbres (s), otro considerando la incertidumbre en x (i) y otro considerando la incertidumbre tanto en x como en y (o).  
"""

from numpy import sqrt as npsqrt, diag, linspace, array
from scipy.odr import Model, Data, ODR
from pandas import DataFrame, concat
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------
# DEFINICIÓN DE FUNCIONES #
#------------------------------------------------------------------------------

# Para el ajuste o:
def recta(A, x):     
   return A[0]*x + A[1]  

# Para los ajustes i y s:
def recta_simple(x, A, B):
   return A*x + B

# Ajuste:
def recta_odr(y, dy, x, dx):

   # Pesos
   wy = 1/dy**2
   wx = 1/dx**2

   mod   = Model(recta)
   dat   = Data(x, y, wd=wx, we=wy)
   myodr = ODR(dat, mod, beta0=[1, 0])

   return myodr.run()

# Incertidumbre al evaluar en la recta obtenida:
def recta_incertidumbre(x, A, dA, dx):   
   
   A_0 = A[0]; dA_0 = dA[0]
   A_1 = A[1]; dA_1 = dA[1] 
   
   return npsqrt(A_0**2*dx**2 + dA_0**2*x**2 + dA_1**2)

# Incertidumbre relativa al evaluar en la recta:
def recta_incertidumbre_rel(x, A, dA, dx):   
   
   A_0 = A[0]; dA_0 = dA[0]
   A_1 = A[1]; dA_1 = dA[1] 
   
   return npsqrt(A_0**2*dx**2 + dA_0**2*x**2 + dA_1**2)/(A_0*x + A_1)*100


#------------------------------------------------------------------------------
# ENTRADA DE LOS DATOS #
#------------------------------------------------------------------------------

x  = [0,1,2,3,4,5,6,7,8,9]
y  = [-1,1.5,2.1,3.6,4.8,5.9,6,7.3,8.7,9.1]
dx = [0.1,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
dy = [0.7,0.1,1.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

df = DataFrame({'x':x, 'dx':dx,'y':y, 'dy':dy,})

# Título y ejes de las graficas:
nombre_ejex = 'Datos en $x$'
nombre_ejey = 'Datos en $y$'
titulo = 'Regresión lineal'

# Seleccionar tipo de ajuste:
Incert_xy = True;     Incert_x  = True;     No_Incert = True

# Valores a predecir (se requiere que sean arrays):
x_p = array([4.5, 6.8, 11, 15, 19, 19.5])
dx_p = array([0.45, 0.68, 0.1, 0.15, 0.19, 0.15])
 
# Auxiliares:
x_fit = linspace(min(df.x), max(df.x), 1000, True)
Lista_df = []


#------------------------------------------------------------------------------
# APLICACION, OBTENCION DE INFORMACIÓN Y PREDICCION DE LOS AJUSTES #
#------------------------------------------------------------------------------

# Se hace el ajuste o:
if Incert_xy == True:

   Ajuste_o_info = recta_odr(y=df.y, dy=df.dy, x=df.x, dx=df.dx)
   Ajuste_o_param = Ajuste_o_info.beta[:]                                  
   Ajuste_o_dparam = npsqrt(diag(Ajuste_o_info.cov_beta))
   
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

   Ajuste_i_param, Ajuste_i_dparam = curve_fit(recta_simple, df.x, df.y, sigma=df.dy)
   Ajuste_i_dparam = npsqrt(diag(Ajuste_i_dparam))

   Lista_df.append(DataFrame({'Ajuste':['Incert-x'],  'm':[Ajuste_i_param[0]], 'b':[Ajuste_i_param[1]], 'dm':[Ajuste_i_dparam[0]], 'db':[Ajuste_i_dparam[1]]}))
   plt.plot(x_fit, recta(Ajuste_i_param, x_fit), color='#FF04C2', lw=2, ls='--', alpha=0.8, label='Incertidumbre en $x$ (i)')

   y_p  = recta(Ajuste_i_param, x_p)
   dy_p = recta_incertidumbre(x=x_p, A=Ajuste_i_param, dA=Ajuste_i_dparam, dx=dx_p)
   plt.errorbar(x=x_p, xerr=dx_p, y=y_p, yerr=dy_p, color='#FFD100', alpha=0.8, fmt='^', capsize=3, label='Predicciones (i)')

   p_i = DataFrame({'x_p':x_p, 'dx_p':dx_p, 'y_p':y_p, 'dy_p':dy_p})
   print('\nPredicciones con incertidumbres en y:\n', p_i)


# Se hace el ajuste s:
if No_Incert == True:

   Ajuste_s_param, Ajuste_s_dparam = curve_fit(recta_simple, df.x, df.y)
   Ajuste_s_dparam = npsqrt(diag(Ajuste_s_dparam))

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


# Se une la información de los ajustes 
print('\nInformacion de los ajustes:\n', concat(Lista_df).set_index('Ajuste'))

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