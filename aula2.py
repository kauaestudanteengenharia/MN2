import numpy as np
import matplotlib.pyplot as plt

# Função do PVI: y' = 4 e^(0.8 t) - 0.5 y
def f(t, y):
    return 4 * np.exp(0.8 * t) - 0.5 * y

# Solução analítica
def analitica(t):
    return (4 / 1.3) * (np.exp(0.8 * t) - np.exp(-0.5 * t)) + 2 * np.exp(-0.5 * t)

# Parâmetros do problema
a= 0
b= 4
n= 4
h = (b-a)/n
t = np.linspace(0, 4, 5)  # 0, 1, 2, 3, 4
y = np.zeros(5)
Ept = np.zeros(n+1)

# Condição inicial
y[0] = 2


# Método de Runge-Kutta de 2 ordem
for i in range(0, 4):
    k1 = h * f(t[i], y[i])
    k2 = h * f(t[i] + h, y[i] + k1 * h)
    y[i + 1] = y[i] + 1/2*(k1 + k2)
    Ept[i + 1] = abs((analitica(t[i + 1]) - y[i + 1]) / analitica(t[i + 1]) * 100)
print (y)

# Solução exata
y_exata = analitica(t)

# Erro absoluto
erro = np.abs(y_exata - y)

# Gráfico
plt.plot(t, y_exata, 'o-', label='Solução Exata')
plt.plot(t, y, 's--', label='RK (h=1)')
plt.plot(t, erro, 'b--', label='erro')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.legend()
plt.grid(True)
plt.show()