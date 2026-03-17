"""
Descripción: Genera una imagen del conjunto de Mandelbrot utilizando Python.
El usuario puede ajustar la región, la resolución y el número de iteraciones.
Autor: Anonimo
Fecha: 13/03/2026
"""

import numpy as np
import matplotlib.pyplot as plt
import time

def calcular_mandelbrot(ancho, alto, x_min, x_max, y_min, y_max, max_iter):
    """
    Calcula el conjunto de Mandelbrot para una región dada.
    Retorna una matriz con el número de iteraciones para cada píxel.
    """
    # Crear arrays de coordenadas reales e imaginarias
    x = np.linspace(x_min, x_max, ancho)
    y = np.linspace(y_min, y_max, alto)
    
    # Inicializar matriz de resultados
    fractal = np.zeros((alto, ancho), dtype=np.int32)
    
    # Recorrer cada píxel
    for i in range(alto):
        for j in range(ancho):
            c = complex(x[j], y[i])
            z = 0.0j
            for k in range(max_iter):
                z = z*z + c
                if (z.real*z.real + z.imag*z.imag) > 4.0:
                    fractal[i, j] = k
                    break
            else:
                fractal[i, j] = max_iter  # pertenece al conjunto
    return fractal

def aplicar_paleta_colores(fractal, max_iter):
    """
    Aplica una paleta de colores a la matriz de iteraciones.
    Retorna una matriz RGB para mostrar con matplotlib.
    """
    # Normalizar valores entre 0 y 1
    normalizado = fractal / max_iter
    
    # Crear imagen en color usando un mapa de colores (hot)
    colores = plt.cm.hot(normalizado)
    return colores

def guardar_imagen(nombre_archivo, colores):
    """
    Guarda la imagen en un archivo PNG.
    """
    plt.imsave(nombre_archivo, colores)
    print(f"Imagen guardada como {nombre_archivo}")

def mostrar_imagen(colores):
    """
    Muestra la imagen en pantalla.
    """
    plt.imshow(colores)
    plt.title("Conjunto de Mandelbrot")
    plt.axis('off')
    plt.show()

def main():
    """
    Función principal.
    """
    print("=== GENERADOR DE FRACTAL DE MANDELBROT ===\n")
    
    # Parámetros configurables
    ancho = 800          # píxeles
    alto = 600
    x_min = -2.0
    x_max = 1.0
    y_min = -1.5
    y_max = 1.5
    max_iter = 256        # iteraciones máximas
    
    print(f"Resolución: {ancho} x {alto}")
    print(f"Región: x=[{x_min}, {x_max}], y=[{y_min}, {y_max}]")
    print(f"Máximo de iteraciones: {max_iter}")
    
    inicio = time.time()
    print("Calculando fractal...")
    fractal = calcular_mandelbrot(ancho, alto, x_min, x_max, y_min, y_max, max_iter)
    fin = time.time()
    print(f"Cálculo completado en {fin - inicio:.2f} segundos.")
    
    print("Aplicando paleta de colores...")
    imagen_color = aplicar_paleta_colores(fractal, max_iter)
    
    # Guardar imagen
    guardar_imagen("mandelbrot_generado.png", imagen_color)
    
    # Mostrar imagen
    mostrar_imagen(imagen_color)

if __name__ == "__main__":
    main()