# -*- coding: utf-8 -*-
"""
Módulo: funciones_prueba.py
Descripción: Define las funciones objetivo para evaluar el algoritmo.
Autor: Anonimo
"""

import numpy as np

class FuncionesPrueba:
    """
    Colección de funciones de prueba clásicas para optimización.
    Cada función tiene:
    - nombre: identificador
    - dimensiones: número de variables (puede ser variable)
    - limites: tupla (min, max) para cada dimensión
    - optimo_global: valor en el mínimo
    - posicion_optimo: coordenadas del mínimo
    """
    
    @staticmethod
    def esfera(x):
        """
        Función esfera (continuua, convexa, unimodal)
        Mínimo global: 0 en x = [0, 0, ..., 0]
        """
        return np.sum(x**2)
    
    @staticmethod
    def rosenbrock(x):
        """
        Función de Rosenbrock (valle estrecho, difícil para algunos algoritmos)
        Mínimo global: 0 en x = [1, 1, ..., 1]
        """
        n = len(x)
        suma = 0
        for i in range(n-1):
            suma += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
        return suma
    
    @staticmethod
    def rastrigin(x):
        """
        Función de Rastrigin (múltiples mínimos locales)
        Mínimo global: 0 en x = [0, 0, ..., 0]
        """
        n = len(x)
        return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))
    
    @staticmethod
    def ackley(x):
        """
        Función de Ackley (múltiples mínimos locales)
        Mínimo global: 0 en x = [0, 0, ..., 0]
        """
        n = len(x)
        sum1 = np.sum(x**2)
        sum2 = np.sum(np.cos(2 * np.pi * x))
        return -20 * np.exp(-0.2 * np.sqrt(sum1/n)) - np.exp(sum2/n) + 20 + np.e
    
    @staticmethod
    def griewank(x):
        """
        Función de Griewank (mínimos locales regularmente espaciados)
        Mínimo global: 0 en x = [0, 0, ..., 0]
        """
        n = len(x)
        sum1 = np.sum(x**2) / 4000
        prod1 = np.prod(np.cos(x / np.sqrt(np.arange(1, n+1))))
        return sum1 - prod1 + 1
    
    @staticmethod
    def sphere_ruidosa(x):
        """
        Función esfera con ruido gaussiano (prueba de robustez)
        """
        return np.sum(x**2) + np.random.normal(0, 0.01)
    
    @staticmethod
    def get_funcion(nombre):
        """
        Retorna la función correspondiente al nombre dado.
        """
        funciones = {
            'esfera': FuncionesPrueba.esfera,
            'rosenbrock': FuncionesPrueba.rosenbrock,
            'rastrigin': FuncionesPrueba.rastrigin,
            'ackley': FuncionesPrueba.ackley,
            'griewank': FuncionesPrueba.griewank,
            'sphere_ruidosa': FuncionesPrueba.sphere_ruidosa
        }
        return funciones.get(nombre, FuncionesPrueba.esfera)
    
    @staticmethod
    def get_limites(nombre):
        """
        Retorna los límites de búsqueda para cada función.
        """
        limites = {
            'esfera': (-5.12, 5.12),
            'rosenbrock': (-2.048, 2.048),
            'rastrigin': (-5.12, 5.12),
            'ackley': (-32.768, 32.768),
            'griewank': (-600, 600),
            'sphere_ruidosa': (-5.12, 5.12)
        }
        return limites.get(nombre, (-10, 10))
    
    @staticmethod
    def get_optimo(nombre):
        """
        Retorna el valor óptimo de cada función.
        """
        optimos = {
            'esfera': 0,
            'rosenbrock': 0,
            'rastrigin': 0,
            'ackley': 0,
            'griewank': 0,
            'sphere_ruidosa': 0
        }
        return optimos.get(nombre, 0)


# Prueba rápida
if __name__ == "__main__":
    print("--- Prueba de funciones ---")
    x = np.array([1.0, 1.0, 1.0])
    print(f"Esfera en [1,1,1]: {FuncionesPrueba.esfera(x)}")
    print(f"Rosenbrock en [1,1,1]: {FuncionesPrueba.rosenbrock(x)}")
    print(f"Rastrigin en [0,0,0]: {FuncionesPrueba.rastrigin(np.zeros(3))}")