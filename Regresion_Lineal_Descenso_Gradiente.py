"""
Programa que calcula la pendiente, m, y la oordenada al origen, b, de la recta que mejor se ajusta a un conjunto de datos, (x,y). En una regresión lineal, m y b se obtienen resolviendo el sistema de ecuaciones que surge al minimizar la funcion del error (o de costo):

err=  (\sum_{i=1}^n [y_1-(mxi+b)]^2)/n                       (1)

En este caso, m y b se aproximan por el método del descenso del gradiente. El código requiere que el peso, alpha, que controla el paso de m y b, se coloque a mano, por lo que de no hacer la elección correcta el programa puede romperse. 
"""

from numpy import array, linspace, meshgrid
import matplotlib.pyplot as plt

#--------------------------------------------------------------------
# PREELIMINARES
#--------------------------------------------------------------------

# Para almacenar los pasos en m, b y el error:
m_g = []; b_g = []; err_g = []

# Función que aplica el descenso del gradiente:
def descenso_gradiente(x,y, m0, b0, iteraciones, n):
    if n != len(y):
        print('x e y tienen tamaños distintos.')
    else:
        # Peso 
        alpha = 0.0001       

        for i in range(iteraciones):

            y_ajuste = m0*x + b0

            # Parciales del error respecto a m y b:
            parcial_m = -(2/n)*sum((y-y_ajuste)*x)
            parcial_b = -(2/n)*sum(y-y_ajuste)
            err = (1/n)*sum((y-y_ajuste)**2)


            # Avance de m y b:
            m0 = m0 - alpha*parcial_m
            b0 = b0 - alpha*parcial_b

            # Almacén de m, b y el error:
            m_g.append(m0)
            b_g.append(b0)
            err_g.append(err)

            # Tolerancia o variación de m y b:
            Tol_m = abs(alpha*parcial_m*100/m0)
            Tol_b = abs(alpha*parcial_b*100/b0)

            if Tol_m < 0.001 and Tol_b< 0.001 :
                break
            else:
                continue
        print(f' m: {round(m0, 2)}  b: {round(b0,2)} error: {round(err,2)} tolerancia m: {round(Tol_m,2)}% tolerancia b: {round(Tol_b,2)}% iteracion: {i}')

    return m0, b0

#--------------------------------------------------------------------
# ENTRADA DE LOS DATOS
#--------------------------------------------------------------------

# Títulos:
titulo_x = 'valores en x'
titulo_y = 'valores en y'
titulo_grafica = 'Ajuste lineal por descenso\n del gradiente'

#--------------------------------------------------------------------------
# Datos de prueba:

# Alpha = 0.001 
x_exp = array([1,   2,   3,   4,   5,   6,   7,   8,   9,   10, 11])
y_exp = array([1.2, 2.1, 3.5, 3.9, 4.7, 6.1, 7.2, 7.9, 9.5, 9.9, 11])

# Alpha = 0.000001 
#x_exp = array([92,   58,   88,   70,   80,   49,   65,   35,   66])
#y_exp = array([98,   68,   81,   80,   83,   52,   66,   30,   68])
#--------------------------------------------------------------------------

# Auxiliar:
n = len(x_exp)

# Aplicación del ajuste
m, b = descenso_gradiente(x=x_exp, y=y_exp, m0=0, b0=0, iteraciones=2000, n=n)

# Gráfica del ajuste realizado
x_fit = linspace(min(x_exp), max(x_exp), 10)
y_fit = m*x_fit+b

plt.plot(x_fit, y_fit, color='r', label='ajuste', lw=0.9, alpha=0.8)
plt.errorbar(x=x_exp, y=y_exp, color='black', label='Datos', fmt='.', alpha=0.9)

# Estética de la gráfica del ajuste
plt.title(titulo_grafica, fontsize='20')
plt.grid(linestyle='--', color='#A47200', alpha=0.5, lw=0.6)
plt.xlabel(f'{titulo_x}', fontsize='18')
plt.ylabel(f'{titulo_y}', fontsize='18')
plt.xticks(fontsize='16')
plt.yticks(fontsize='16')
plt.legend()


# Gráfica del recorrido de m y b:
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

X = linspace(min(m_g)-0.5, max(m_g)+0.5, 20)
Y = linspace(min(b_g)-5, max(b_g)+5, 20)
X,Y = meshgrid(X,Y)

aux=0
for i in range(n):
    aux=aux+(y_exp[i]-(X*x_exp[i]+Y))**2
Z = aux/n

ax.plot_surface(X,Y,Z, color='cyan', alpha=0.5)
ax.plot(m_g, b_g, err_g, color='r', label='recorrido (m,b)')

# Estética de la gráfica del recorrido de m y b:
ax.set_title('Error del ajuste lineal\n en función de m y b')
ax.set_xlabel('Pendiente m')
ax.set_ylabel('Interseccion b')
ax.set_zlabel('Error')
plt.legend()


plt.show()






