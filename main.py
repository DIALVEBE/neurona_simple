import numpy as np

print("Estamos creando una neurona para que aprenda el funcionamiento de la compuerta AND")
# Cada fila contiene dos entradas.
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([[0], [0], [0], [1]], dtype=float)

print("Entradas:\n", X)
print("Respuestas esperadas:\n", y)

print("La neurona calcula `z = x1*w1 + x2*w2 + b` y aplica la función sigmoide para producir un número entre 0 y 1. Los pesos `w1`, `w2` y el sesgo `b` son los valores que aprenderá.")
def sigmoide(z):
    return 1 / (1 + np.exp(-z))

rng = np.random.default_rng(7)
pesos = rng.normal(size=(2, 1))
sesgo = 0.0
tasa_aprendizaje = 0.5
epocas = 10000

for epoca in range(epocas):
    z = X @ pesos + sesgo
    predicciones = sigmoide(z)

    # Error cuadrático medio y sus derivadas para esta única neurona.
    error = predicciones - y
    gradiente_z = 2 * error * predicciones * (1 - predicciones) / len(X)
    gradiente_pesos = X.T @ gradiente_z
    gradiente_sesgo = np.sum(gradiente_z)

    pesos -= tasa_aprendizaje * gradiente_pesos
    sesgo -= tasa_aprendizaje * gradiente_sesgo

    if epoca % 2000 == 0:
        perdida = np.mean(error ** 2)
        print(f"Época {epoca:5d} | error: {perdida:.4f}")

print("Pesos aprendidos:", pesos.ravel())
print("Sesgo aprendido:", sesgo)

print("Probamos lo aprendido: Si la salida es mayor o igual a 0.5, la interpretamos como 1; en caso contrario, como 0.")

probabilidades = sigmoide(X @ pesos + sesgo)
respuestas = (probabilidades >= 0.5).astype(int)

for entrada, probabilidad, respuesta in zip(X, probabilidades.ravel(), respuestas.ravel()):
    print(f"{entrada.astype(int)} -> probabilidad: {probabilidad:.3f} -> respuesta: {respuesta}")

assert np.array_equal(respuestas, y.astype(int))


## Para experimentar
# cambia `epocas` a 100 y vuelve a ejecutar desde la celda de entrenamiento.
# ¿Qué ocurre? Después prueba con `tasa_aprendizaje = 0.05`.
# La compuerta XOR (`[0,0]→0`, `[0,1]→1`, `[1,0]→1`, `[1,1]→0`)
# no puede aprenderse perfectamente con esta sola neurona: requiere otra estructura.
