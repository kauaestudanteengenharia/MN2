import numpy as np
import matplotlib.pyplot as plt

# Função do PVI: y' = 4 e^(0.8 t) - 0.5 y
def f(t, y):
    return 4 * np.exp(0.8 * t) - 0.5 * y

# Solução analítica
def analitica(t):
    return (4 / 1.3) * (np.exp(0.8 * t) - np.exp(-0.5 * t)) + 2 * np.exp(-0.5 * t)

# Parâmetros do problema
a = 0
b = 4
n = 1
h = (b - a) / n
t = np.linspace(a, b, n+1)  # agora com n+1 pontos
y = np.zeros(n+1)
Ept = np.zeros(n+1)

# Condição inicial
y[0] = 2

# Método de Euler
for i in range(n):
    y[i+1] = y[i] + h * f(t[i], y[i])
    Ept[i+1] = abs((analitica(t[i+1]) - y[i+1]) / analitica(t[i+1]) * 100)

# Solução exata
y_exata = analitica(t)

# Erro absoluto
erro = np.abs(y_exata - y)

# Gráfico
plt.plot(t, y_exata, 'o-', label='Solução Exata')
plt.plot(t, y, 's--', label=f'Euler (h={h:.2f})')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.legend()
plt.grid(True)
plt.show()

# Impressão dos resultados
print("Solução aproximada:", y)
print("Erro percentual (%):", Ept)
