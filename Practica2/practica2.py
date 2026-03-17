"""
Descripción: Lee una matriz de adyacencias desde un archivo de texto,
calcula el número de vértices (n), el número de aristas (m) y el grado de cada vértice.
La matriz se asume simétrica (grafo no dirigido).
Autor: Anonimo
Fecha: 13/03/2026
"""

import os

def leer_matriz_desde_archivo(nombre_archivo):
    """
    Lee una matriz de adyacencias desde un archivo de texto.
    Cada línea del archivo contiene los valores de una fila separados por espacios.
    Retorna una lista de listas (matriz) y el número de vértices.
    """
    matriz = []
    try:
        with open(nombre_archivo, 'r', encoding='utf-8') as archivo:
            for linea in archivo:
                linea = linea.strip()
                if linea:
                    fila = list(map(int, linea.split()))
                    matriz.append(fila)
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo '{nombre_archivo}'. Asegúrate de que está en la misma carpeta.")
        return None, 0
    except Exception as e:
        print(f"❌ Error al leer el archivo: {e}")
        return None, 0

    n = len(matriz)
    for fila in matriz:
        if len(fila) != n:
            print("❌ Error: La matriz no es cuadrada.")
            return None, 0
    return matriz, n

def verificar_simetria(matriz, n):
    for i in range(n):
        for j in range(i+1, n):
            if matriz[i][j] != matriz[j][i]:
                return False
    return True

def calcular_aristas(matriz, n):
    m = 0
    for i in range(n):
        for j in range(i+1, n):
            if matriz[i][j] == 1:
                m += 1
    return m

def calcular_grados(matriz, n):
    grados = []
    for i in range(n):
        grado = sum(matriz[i][j] for j in range(n))
        grados.append(grado)
    return grados

def mostrar_resultados(n, m, grados, nombres):
    print("\n" + "="*50)
    print("RESULTADOS DEL ANÁLISIS DE LA RED DE AMISTADES")
    print("="*50)
    print(f"Número de vértices (n): {n}")
    print(f"Número de aristas (m): {m}")
    print("\nGrados de cada vértice:")
    print("-" * 30)
    for i, nombre in enumerate(nombres):
        print(f"{nombre:10} : {grados[i]} conexiones")
    print("-" * 30)

def main():
    # Ruta al archivo en la misma carpeta que el script
    directorio_script = os.path.dirname(os.path.abspath(__file__))
    archivo_matriz = os.path.join(directorio_script, "matriz_amigos.txt")
    
    nombres = ["Mateo", "Marcos", "Juan", "Lucas", "Santiago", 
               "Miguel", "Jesús", "Cristian", "Daniel", "Esteban"]
    
    print("=== ANALIZADOR DE RED DE AMISTADES ===\n")
    print(f"Leyendo matriz desde el archivo: {archivo_matriz}")
    
    matriz, n = leer_matriz_desde_archivo(archivo_matriz)
    if matriz is None:
        return
    
    print(f"Matriz cargada correctamente. Dimensiones: {n}x{n}")
    
    if verificar_simetria(matriz, n):
        print("La matriz es simétrica (grafo no dirigido).")
    else:
        print("Advertencia: La matriz no es simétrica. Se esperaba un grafo no dirigido.")
    
    m = calcular_aristas(matriz, n)
    grados = calcular_grados(matriz, n)
    mostrar_resultados(n, m, grados, nombres)
    
    # Guardar resultados en el mismo directorio
    archivo_salida = os.path.join(directorio_script, "resultados_red.txt")
    with open(archivo_salida, "w", encoding="utf-8") as salida:
        salida.write("RESULTADOS DEL ANÁLISIS DE LA RED DE AMISTADES\n")
        salida.write(f"Número de vértices (n): {n}\n")
        salida.write(f"Número de aristas (m): {m}\n")
        salida.write("\nGrados de cada vértice:\n")
        for i, nombre in enumerate(nombres):
            salida.write(f"{nombre}: {grados[i]}\n")
    
    print("\nLos resultados también se han guardado en 'resultados_red.txt'.")

if __name__ == "__main__":
    main()