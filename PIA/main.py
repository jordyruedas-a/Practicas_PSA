# -*- coding: utf-8 -*-
"""
Programa principal: main.py
Descripción: Punto de entrada para ejecutar experimentos de comparación
entre PSO tradicional y PSO adaptativo.
Autor: Anonimo
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Importar nuestros módulos
from pso_adaptativo import PSOAdaptativo
from funciones_prueba import FuncionesPrueba


def ejecutar_experimento(nombre_problema, dimensiones, num_particulas, 
                        max_iteraciones, num_ejecuciones=10):
    """
    Ejecuta múltiples veces ambos algoritmos para comparación estadística.
    """
    print("\n" + "="*70)
    print(f" EXPERIMENTO: {nombre_problema} (dim={dimensiones})")
    print("="*70)
    
    # Obtener función y límites
    funcion = FuncionesPrueba.get_funcion(nombre_problema)
    limites = FuncionesPrueba.get_limites(nombre_problema)
    optimo_real = FuncionesPrueba.get_optimo(nombre_problema)
    
    # Almacenar resultados
    resultados_trad = []
    resultados_adapt = []
    
    for ejec in range(num_ejecuciones):
        print(f"\n--- Ejecución {ejec+1}/{num_ejecuciones} ---")
        
        # PSO TRADICIONAL (sin control difuso)
        pso_trad = PSOAdaptativo(
            funcion_objetivo=funcion,
            dimensiones=dimensiones,
            num_particulas=num_particulas,
            limites=limites,
            max_iteraciones=max_iteraciones,
            usar_control_difuso=False,
            nombre_problema=nombre_problema
        )
        mejor_valor_trad, _ = pso_trad.ejecutar(verbose=False)
        resultados_trad.append(mejor_valor_trad)
        
        # PSO ADAPTATIVO (con control difuso)
        pso_adapt = PSOAdaptativo(
            funcion_objetivo=funcion,
            dimensiones=dimensiones,
            num_particulas=num_particulas,
            limites=limites,
            max_iteraciones=max_iteraciones,
            usar_control_difuso=True,
            nombre_problema=nombre_problema
        )
        mejor_valor_adapt, _ = pso_adapt.ejecutar(verbose=False)
        resultados_adapt.append(mejor_valor_adapt)
    
    # Estadísticas
    media_trad = np.mean(resultados_trad)
    std_trad = np.std(resultados_trad)
    media_adapt = np.mean(resultados_adapt)
    std_adapt = np.std(resultados_adapt)
    
    print("\n" + "="*70)
    print(" RESULTADOS ESTADÍSTICOS")
    print("="*70)
    print(f"PSO TRADICIONAL:")
    print(f"  Media: {media_trad:.6f}")
    print(f"  Desviación: {std_trad:.6f}")
    print(f"  Mejor: {np.min(resultados_trad):.6f}")
    print(f"  Peor: {np.max(resultados_trad):.6f}")
    print()
    print(f"PSO ADAPTATIVO:")
    print(f"  Media: {media_adapt:.6f}")
    print(f"  Desviación: {std_adapt:.6f}")
    print(f"  Mejor: {np.min(resultados_adapt):.6f}")
    print(f"  Peor: {np.max(resultados_adapt):.6f}")
    
    # Calcular mejora porcentual
    mejora = ((media_trad - media_adapt) / media_trad) * 100 if media_trad != 0 else 0
    print(f"\n Mejora del adaptativo: {mejora:.2f}%")
    
    return {
        'problema': nombre_problema,
        'dimensiones': dimensiones,
        'tradicional': resultados_trad,
        'adaptativo': resultados_adapt,
        'mejora': mejora
    }


def graficar_comparacion(resultados):
    """
    Genera gráfica de caja comparando ambos algoritmos.
    """
    plt.figure(figsize=(10, 6))
    
    datos = [resultados['tradicional'], resultados['adaptativo']]
    
    bp = plt.boxplot(datos, labels=['PSO Tradicional', 'PSO Adaptativo'],
                    patch_artist=True)
    
    # Colorear
    bp['boxes'][0].set_facecolor('lightcoral')
    bp['boxes'][1].set_facecolor('lightgreen')
    
    plt.ylabel('Mejor valor encontrado')
    plt.title(f'Comparación en {resultados["problema"]} (dim={resultados["dimensiones"]})')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # Guardar
    nombre_archivo = f"comparacion_{resultados['problema']}.png"
    plt.savefig(nombre_archivo, dpi=150, bbox_inches='tight')
    print(f" Gráfica guardada: {nombre_archivo}")
    plt.show()


def ejecutar_demo_visual():
    """
    Ejecuta una demo visual para mostrar en el video.
    """
    print("\n" + "="*70)
    print(" DEMO VISUAL - PSO Adaptativo en 2D")
    print("="*70)
    
    # Ejecutar PSO adaptativo en función esfera (2D)
    pso_demo = PSOAdaptativo(
        funcion_objetivo=FuncionesPrueba.rosenbrock,
        dimensiones=2,
        num_particulas=25,
        limites=(-2, 2),
        max_iteraciones=100,
        usar_control_difuso=True,
        nombre_problema="rosenbrock_demo"
    )
    
    pso_demo.ejecutar(verbose=True)
    pso_demo.graficar_convergencia(guardar=True)
    
    return pso_demo


if __name__ == "__main__":
    print("="*70)
    print(" TRABAJO FINAL - PROGRAMACIÓN DE SISTEMAS ADAPTATIVOS")
    print(" PSO con Control Difuso Adaptativo")
    print(" Autor: Anonimo")
    print("="*70)
    
    # Crear carpeta para resultados si no existe
    if not os.path.exists('resultados'):
        os.makedirs('resultados')
        print(" Carpeta 'resultados' creada")
    
    # Ejecutar demo visual (para el video)
    pso_demo = ejecutar_demo_visual()
    
    # Ejecutar experimentos formales
    print("\n" + "="*70)
    print(" EXPERIMENTOS FORMALES")
    print("="*70)
    
    experimentos = [
        ('esfera', 2, 30, 100, 20),
        ('esfera', 10, 50, 500, 15),
        ('rosenbrock', 2, 30, 200, 20),
        ('rosenbrock', 10, 100, 1000, 10),
        ('rastrigin', 2, 30, 200, 15),
        ('rastrigin', 10, 100, 1000, 10),
    ]
    
    resultados_totales = []
    
    for prob, dim, part, iters, reps in experimentos:
        res = ejecutar_experimento(prob, dim, part, iters, reps)
        resultados_totales.append(res)
        graficar_comparacion(res)
    
    # Generar reporte final
    print("\n" + "="*70)
    print(" REPORTE FINAL - RESUMEN")
    print("="*70)
    
    for res in resultados_totales:
        print(f"\n{res['problema']} (dim={res['dimensiones']}):")
        print(f"  Tradicional: media={np.mean(res['tradicional']):.6f}")
        print(f"  Adaptativo: media={np.mean(res['adaptativo']):.6f}")
        print(f"  Mejora: {res['mejora']:.2f}%")
    
    print("\n" + "="*70)
    print(" TRABAJO COMPLETADO")
    print(" Resultados guardados en carpeta 'resultados/'")
    print("="*70)