"""
Descripción: Implementa el algoritmo de clustering K-Means desde cero para segmentar
             clientes de un supermercado en función de su edad e ingresos.
Autor: Anonimo
Fecha: 13/03/2026
"""

# =============================================================================
# IMPORTACIÓN DE LIBRERÍAS
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.datasets import make_blobs  # Solo para generar datos sintéticos
from sklearn.preprocessing import StandardScaler
import time

# =============================================================================
# GENERACIÓN DE DATOS SINTÉTICOS (SIMULACIÓN DE CLIENTES)
# =============================================================================
def generar_datos_clientes():
    """
    Genera un conjunto de datos sintético que simula 200 clientes.
    Cada cliente tiene dos características: edad (años) e ingreso anual (dólares).
    Los datos se crean con 4 centros predefinidos para simular distintos perfiles.
    """
    print("Generando datos sintéticos de clientes...")
    
    # Semilla para reproducibilidad
    np.random.seed(42)
    random.seed(42)
    
    # Definimos los centros de los clusters (edad, ingreso)
    centros_reales = np.array([
        [25, 30000],   # Jóvenes con bajo ingreso
        [30, 100000],  # Jóvenes con alto ingreso
        [45, 60000],   # Adultos con ingreso medio
        [60, 120000]   # Adultos mayores con alto ingreso
    ])
    
    # Generamos 200 puntos con una desviación estándar base pequeña
    # Usamos make_blobs para mayor realismo (clusters gaussianos)
    X, etiquetas_reales = make_blobs(
        n_samples=200,
        centers=centros_reales,
        cluster_std=1.0,  # Desviación controlada que no afectará drásticamente la edad
        random_state=42
    )
    
    # Añadimos ruido aleatorio asimétrico, específico para la edad y el ingreso
    # para simular las diferencias de escala en la desviación
    ruido_edad = np.random.normal(0, 4, size=X.shape[0])       # +- 4 años de std
    ruido_ingreso = np.random.normal(0, 8000, size=X.shape[0]) # +- 8000 dólares de std
    X[:, 0] += ruido_edad
    X[:, 1] += ruido_ingreso
    
    # Aseguramos que la edad sea positiva y dentro de un rango razonable
    X[:, 0] = np.clip(X[:, 0], 18, 70)
    X[:, 1] = np.abs(X[:, 1])  # Evitamos ingresos negativos
    
    print(f"Se generaron {X.shape[0]} clientes con {X.shape[1]} características.")
    return X, etiquetas_reales

# =============================================================================
# FUNCIONES AUXILIARES PARA K-MEANS
# =============================================================================
def distancia_euclidiana(punto_a, punto_b):
    """
    Calcula la distancia euclidiana entre dos puntos (vectores).
    """
    suma_cuadrados = 0
    for i in range(len(punto_a)):
        suma_cuadrados += (punto_a[i] - punto_b[i]) ** 2
    return np.sqrt(suma_cuadrados)

def inicializar_centroides_aleatorio(X, k):
    """
    Selecciona k puntos aleatorios del conjunto como centroides iniciales.
    """
    indices = random.sample(range(X.shape[0]), k)
    centroides = X[indices].copy()
    return centroides

def inicializar_centroides_kmeans_pp(X, k):
    """
    Implementa el método K-Means++ para una mejor inicialización.
    """
    centroides = []
    # Primer centroide: elegido al azar
    primer_indice = random.randint(0, X.shape[0] - 1)
    centroides.append(X[primer_indice].copy())
    
    for _ in range(1, k):
        # Calcular distancias al cuadrado de cada punto al centroide más cercano
        distancias_cuad = np.zeros(X.shape[0])
        for i, punto in enumerate(X):
            dist_min = float('inf')
            for c in centroides:
                dist = distancia_euclidiana(punto, c)
                if dist < dist_min:
                    dist_min = dist
            distancias_cuad[i] = dist_min ** 2
        
        # Elegir nuevo centroide con probabilidad proporcional a la distancia
        probabilidades = distancias_cuad / np.sum(distancias_cuad)
        nuevo_indice = np.random.choice(X.shape[0], p=probabilidades)
        centroides.append(X[nuevo_indice].copy())
    
    return np.array(centroides)

def asignar_clusters(X, centroides):
    """
    Asigna cada punto al centroide más cercano.
    Retorna un arreglo con las etiquetas de cluster (0..k-1) para cada punto.
    """
    k = centroides.shape[0]
    etiquetas = np.zeros(X.shape[0], dtype=int)
    
    for i, punto in enumerate(X):
        distancias = np.zeros(k)
        for j, c in enumerate(centroides):
            distancias[j] = distancia_euclidiana(punto, c)
        etiquetas[i] = np.argmin(distancias)
    
    return etiquetas

def actualizar_centroides(X, etiquetas, k):
    """
    Recalcula los centroides como la media de los puntos asignados a cada cluster.
    Retorna los nuevos centroides.
    """
    nuevos_centroides = np.zeros((k, X.shape[1]))
    
    for j in range(k):
        puntos_cluster = X[etiquetas == j]
        if len(puntos_cluster) > 0:
            nuevos_centroides[j] = puntos_cluster.mean(axis=0)
        else:
            # Si un cluster queda vacío, lo reiniciamos con un punto aleatorio
            indice_aleatorio = random.randint(0, X.shape[0] - 1)
            nuevos_centroides[j] = X[indice_aleatorio].copy()
    
    return nuevos_centroides

def calcular_inercia(X, etiquetas, centroides):
    """
    Calcula la suma de distancias al cuadrado de cada punto a su centroide (inercia).
    """
    inercia = 0
    for i, punto in enumerate(X):
        j = etiquetas[i]
        dist = distancia_euclidiana(punto, centroides[j])
        inercia += dist ** 2
    return inercia

def kmeans(X, k, max_iteraciones=300, tolerancia=1e-4, metodo_inicializacion='aleatorio'):
    """
    Ejecuta el algoritmo K-Means completo.
    
    Parámetros:
        X: matriz de datos (n_muestras x n_características)
        k: número de clusters
        max_iteraciones: número máximo de iteraciones
        tolerancia: cambio mínimo en centroides para considerar convergencia
        metodo_inicializacion: 'aleatorio' o 'kmeans++'
    
    Retorna:
        centroides_finales, etiquetas_finales, inercia_final, numero_iteraciones
    """
    # Inicialización
    if metodo_inicializacion == 'kmeans++':
        centroides = inicializar_centroides_kmeans_pp(X, k)
    else:
        centroides = inicializar_centroides_aleatorio(X, k)
    
    historial_centroides = [centroides.copy()]
    
    for iteracion in range(max_iteraciones):
        # Asignación
        etiquetas = asignar_clusters(X, centroides)
        
        # Actualización
        nuevos_centroides = actualizar_centroides(X, etiquetas, k)
        historial_centroides.append(nuevos_centroides.copy())
        
        # Verificar convergencia (cambio en centroides)
        cambios = np.linalg.norm(nuevos_centroides - centroides, axis=1).max()
        centroides = nuevos_centroides
        
        if cambios < tolerancia:
            #print(f"Convergencia alcanzada en {iteracion + 1} iteraciones.")
            break
    
    inercia = calcular_inercia(X, etiquetas, centroides)
    return centroides, etiquetas, inercia, iteracion + 1, historial_centroides

def ejecutar_kmeans_multiple(X, k, n_ejecuciones=10, metodo_inicializacion='aleatorio'):
    """
    Ejecuta K-Means varias veces y retorna el mejor resultado (menor inercia).
    """
    mejor_inercia = float('inf')
    mejor_centroides = None
    mejor_etiquetas = None
    mejor_iteraciones = 0
    mejor_historial = None
    
    for ejecucion in range(n_ejecuciones):
        centroides, etiquetas, inercia, iteraciones, historial = kmeans(
            X, k, metodo_inicializacion=metodo_inicializacion
        )
        #print(f"Ejecución {ejecucion + 1}: inercia = {inercia:.2f}, iteraciones = {iteraciones}")
        
        if inercia < mejor_inercia:
            mejor_inercia = inercia
            mejor_centroides = centroides
            mejor_etiquetas = etiquetas
            mejor_iteraciones = iteraciones
            mejor_historial = historial
    
    return mejor_centroides, mejor_etiquetas, mejor_inercia, mejor_iteraciones, mejor_historial

# =============================================================================
# FUNCIONES DE VISUALIZACIÓN
# =============================================================================
def graficar_datos(X, titulo="Datos de clientes"):
    """
    Muestra un gráfico de dispersión de los datos (edad vs ingreso).
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], alpha=0.6, edgecolors='k', linewidth=0.5)
    plt.xlabel('Edad (años)', fontsize=12)
    plt.ylabel('Ingreso anual (dólares)', fontsize=12)
    plt.title(titulo, fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def graficar_resultados(X, etiquetas, centroides, titulo="Resultado del clustering"):
    """
    Grafica los puntos coloreados por cluster y los centroides.
    """
    plt.figure(figsize=(10, 6))
    
    # Colores para los clusters
    etiquetas_unicas = np.unique(etiquetas)
    colores = plt.cm.tab10(np.linspace(0, 1, len(etiquetas_unicas)))
    
    for idx, color in zip(etiquetas_unicas, colores):
        puntos_cluster = X[etiquetas == idx]
        plt.scatter(puntos_cluster[:, 0], puntos_cluster[:, 1], 
                   color=color, alpha=0.6, label=f'Cluster {idx+1}', 
                   edgecolors='k', linewidth=0.5)
    
    # Graficar centroides
    plt.scatter(centroides[:, 0], centroides[:, 1], 
               marker='X', s=200, c='red', edgecolors='black', 
               linewidths=2, label='Centroides')
    
    plt.xlabel('Edad (años)', fontsize=12)
    plt.ylabel('Ingreso anual (dólares)', fontsize=12)
    plt.title(titulo, fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def graficar_evolucion_centroides(X, historial_centroides, etiquetas_finales):
    """
    Muestra cómo se mueven los centroides a lo largo de las iteraciones.
    """
    plt.figure(figsize=(10, 6))
    
    # Graficar puntos finales con transparencia
    etiquetas_unicas = np.unique(etiquetas_finales)
    colores = plt.cm.tab10(np.linspace(0, 1, len(etiquetas_unicas)))
    for idx, color in zip(etiquetas_unicas, colores):
        puntos_cluster = X[etiquetas_finales == idx]
        plt.scatter(puntos_cluster[:, 0], puntos_cluster[:, 1], 
                   color=color, alpha=0.3, edgecolors='k', linewidth=0.5)
    
    # Graficar evolución de centroides
    historial = np.array(historial_centroides)
    n_iter = historial.shape[0]
    for i in range(historial.shape[1]):  # Para cada centroide
        trayectoria = historial[:, i, :]
        plt.plot(trayectoria[:, 0], trayectoria[:, 1], 'o-', 
                markersize=4, linewidth=1, label=f'Centroide {i+1}')
    
    plt.xlabel('Edad (años)', fontsize=12)
    plt.ylabel('Ingreso anual (dólares)', fontsize=12)
    plt.title('Evolución de los centroides durante las iteraciones', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def metodo_codo(X, k_max=10, n_ejecuciones=5):
    """
    Ejecuta K-Means para diferentes valores de k y grafica la inercia.
    Ayuda a determinar el número óptimo de clusters (método del codo).
    """
    inercias = []
    k_valores = range(1, k_max + 1)
    
    for k in k_valores:
        _, _, inercia, _, _ = ejecutar_kmeans_multiple(X, k, n_ejecuciones=n_ejecuciones)
        inercias.append(inercia)
        print(f"k = {k}: inercia = {inercia:.2f}")
    
    # Graficar
    plt.figure(figsize=(10, 6))
    plt.plot(k_valores, inercias, 'bo-', markersize=8, linewidth=2)
    plt.xlabel('Número de clusters (k)', fontsize=12)
    plt.ylabel('Inercia (WCSS)', fontsize=12)
    plt.title('Método del codo para determinar k óptimo', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(k_valores)
    plt.tight_layout()
    plt.show()
    
    return inercias

# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================
def main():
    """
    Función principal que orquesta toda la práctica.
    """
    print("=" * 70)
    print("PRÁCTICA 3: SEGMENTACIÓN DE CLIENTES CON K-MEANS")
    print("=" * 70)
    
    # 1. Generar datos
    X, etiquetas_reales = generar_datos_clientes()
    
    # 2. Visualizar datos originales
    graficar_datos(X, "Distribución de clientes (edad vs ingreso)")
    
    # 3. Normalizar datos (importante cuando las escalas son diferentes)
    print("\nNormalizando datos (media=0, desviación=1)...")
    escalador = StandardScaler()
    X_norm = escalador.fit_transform(X)
    
    # 4. Determinar k óptimo mediante método del codo
    print("\n" + "-" * 50)
    print("APLICANDO MÉTODO DEL CODO")
    print("-" * 50)
    inercias = metodo_codo(X_norm, k_max=8, n_ejecuciones=5)
    
    # 5. Elegir k (según la gráfica, probablemente 4)
    k_elegido = 4
    print(f"\nSe elige k = {k_elegido} basado en el método del codo.")
    
    # 6. Ejecutar K-Means con el k elegido (usando K-Means++)
    print("\n" + "-" * 50)
    print(f"EJECUTANDO K-MEANS CON k = {k_elegido} (K-Means++)")
    print("-" * 50)
    
    inicio_tiempo = time.time()
    centroides, etiquetas, inercia, iteraciones, historial = ejecutar_kmeans_multiple(
        X_norm, k_elegido, n_ejecuciones=10, metodo_inicializacion='kmeans++'
    )
    tiempo_total = time.time() - inicio_tiempo
    
    print(f"Mejor inercia encontrada: {inercia:.2f}")
    print(f"Número de iteraciones: {iteraciones}")
    print(f"Tiempo de ejecución: {tiempo_total:.4f} segundos")
    
    # 7. Des-normalizar centroides para interpretación
    centroides_originales = escalador.inverse_transform(centroides)
    print("\nCentroides en escala original (edad, ingreso):")
    for i, c in enumerate(centroides_originales):
        print(f"  Cluster {i+1}: Edad = {c[0]:.1f} años, Ingreso = ${c[1]:.0f}")
    
    # 8. Visualizar resultados
    # Primero, necesitamos las etiquetas en el espacio original para graficar
    # Pero podemos graficar usando los datos normalizados y luego ajustar etiquetas
    graficar_resultados(X_norm, etiquetas, centroides, 
                       f"Clustering con k={k_elegido} (datos normalizados)")
    
    # Graficar evolución de centroides
    graficar_evolucion_centroides(X_norm, historial, etiquetas)
    
    # 9. Análisis de los clusters (interpretación)
    print("\n" + "-" * 50)
    print("ANÁLISIS DE LOS CLUSTERS OBTENIDOS")
    print("-" * 50)
    
    for i in range(k_elegido):
        puntos_cluster = X[etiquetas == i]
        edad_media = puntos_cluster[:, 0].mean()
        ingreso_medio = puntos_cluster[:, 1].mean()
        print(f"\nCluster {i+1}:")
        print(f"  Número de clientes: {len(puntos_cluster)}")
        print(f"  Edad media: {edad_media:.1f} años")
        print(f"  Ingreso medio: ${ingreso_medio:.0f}")
    
    print("\n" + "=" * 70)
    print("PROGRAMA FINALIZADO CON ÉXITO")
    print("=" * 70)

if __name__ == "__main__":
    main()