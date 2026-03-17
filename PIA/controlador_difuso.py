# -*- coding: utf-8 -*-
"""
Módulo: controlador_difuso.py
Descripción: Implementa un sistema de inferencia difusa para ajustar
los parámetros del PSO en tiempo real.
Autor: Anonimo
Fecha: 15/03/2026
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class ControladorDifusoPSO:
    """
    Controlador difuso que ajusta los parámetros del PSO basado en
    la diversidad de la población y la mejora de la mejor solución global.
    
    Este es el corazón del sistema adaptativo. Usa reglas del tipo:
    SI diversidad es baja Y mejora es estancada ENTONCES w es alto
    """
    
    def __init__(self):
        """
        Inicializa el controlador difuso definiendo:
        - Variables de entrada (diversidad, mejora)
        - Variables de salida (w, c1, c2)
        - Funciones de membresía
        - Base de reglas
        """
        print(" Inicializando controlador difuso adaptativo...")
        self._crear_variables_entrada()
        self._crear_variables_salida()
        self._crear_funciones_membresia()
        self._crear_reglas()
        self._crear_sistema_control()
        
    def _crear_variables_entrada(self):
        """
        Define las variables de entrada del sistema difuso:
        - diversidad: qué tan dispersas están las partículas
        - mejora_gbest: qué tanto ha mejorado la mejor solución
        """
        # Rango de 0 a 1 (valores normalizados)
        self.diversidad = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'diversidad')
        self.mejora = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'mejora')
        
    def _crear_variables_salida(self):
        """
        Define las variables de salida (parámetros a ajustar):
        - w: factor de inercia (0.4 a 0.9)
        - c1: coeficiente cognitivo (1.0 a 2.5)
        - c2: coeficiente social (1.0 a 2.5)
        """
        self.w = ctrl.Consequent(np.arange(0.4, 0.91, 0.01), 'w')
        self.c1 = ctrl.Consequent(np.arange(1.0, 2.51, 0.01), 'c1')
        self.c2 = ctrl.Consequent(np.arange(1.0, 2.51, 0.01), 'c2')
        
    def _crear_funciones_membresia(self):
        """
        Define las funciones de membresía para cada variable.
        Usamos funciones triangulares y trapezoidales.
        """
        # ------------------------------------------------------------
        # Funciones para DIVERSIDAD (entrada)
        # ------------------------------------------------------------
        # Diversidad baja: las partículas están concentradas
        self.diversidad['baja'] = fuzz.trimf(self.diversidad.universe, [0, 0, 0.3])
        # Diversidad media: equilibrio
        self.diversidad['media'] = fuzz.trimf(self.diversidad.universe, [0.2, 0.45, 0.7])
        # Diversidad alta: partículas dispersas (explorando)
        self.diversidad['alta'] = fuzz.trimf(self.diversidad.universe, [0.6, 0.8, 1.0])
        
        # ------------------------------------------------------------
        # Funciones para MEJORA (entrada)
        # ------------------------------------------------------------
        # Estancado: no hay mejora reciente
        self.mejora['estancado'] = fuzz.trimf(self.mejora.universe, [0, 0, 0.2])
        # Mejora lenta: mejora moderada
        self.mejora['mejora_lenta'] = fuzz.trimf(self.mejora.universe, [0.1, 0.3, 0.5])
        # Mejora rápida: mejora significativa
        self.mejora['mejora_rapida'] = fuzz.trimf(self.mejora.universe, [0.4, 0.7, 1.0])
        
        # ------------------------------------------------------------
        # Funciones para W (salida)
        # ------------------------------------------------------------
        self.w['bajo'] = fuzz.trimf(self.w.universe, [0.4, 0.45, 0.55])
        self.w['medio'] = fuzz.trimf(self.w.universe, [0.5, 0.6, 0.7])
        self.w['alto'] = fuzz.trimf(self.w.universe, [0.65, 0.75, 0.9])
        
        # ------------------------------------------------------------
        # Funciones para C1 (salida)
        # ------------------------------------------------------------
        self.c1['bajo'] = fuzz.trimf(self.c1.universe, [1.0, 1.2, 1.5])
        self.c1['medio'] = fuzz.trimf(self.c1.universe, [1.3, 1.7, 2.0])
        self.c1['alto'] = fuzz.trimf(self.c1.universe, [1.8, 2.2, 2.5])
        
        # ------------------------------------------------------------
        # Funciones para C2 (salida)
        # ------------------------------------------------------------
        self.c2['bajo'] = fuzz.trimf(self.c2.universe, [1.0, 1.2, 1.5])
        self.c2['medio'] = fuzz.trimf(self.c2.universe, [1.3, 1.7, 2.0])
        self.c2['alto'] = fuzz.trimf(self.c2.universe, [1.8, 2.2, 2.5])
        
    def _crear_reglas(self):
        """
        Define la base de reglas del sistema difuso.
        Cada regla es del tipo: SI (condición) ENTONCES (acción)
        Las reglas se basan en el conocimiento experto sobre PSO.
        """
        reglas = []
        
        # Regla 1: Diversidad BAJA y Mejora ESTANCADA
        # -> Probablemente atrapado en mínimo local, necesitamos EXPLORAR
        # w alto (explorar), c1 bajo (menos experiencia personal), c2 alto (más social)
        regla1 = ctrl.Rule(
            self.diversidad['baja'] & self.mejora['estancado'],
            (self.w['alto'], self.c1['bajo'], self.c2['alto'])
        )
        reglas.append(regla1)
        
        # Regla 2: Diversidad BAJA y Mejora LENTA
        # -> Estamos convergiendo pero lento, equilibrio
        regla2 = ctrl.Rule(
            self.diversidad['baja'] & self.mejora['mejora_lenta'],
            (self.w['medio'], self.c1['medio'], self.c2['medio'])
        )
        reglas.append(regla2)
        
        # Regla 3: Diversidad BAJA y Mejora RÁPIDA
        # -> Encontramos buena región, hay que EXPLOTAR
        # w bajo (menos exploración), c1 alto (experiencia personal), c2 bajo (menos social)
        regla3 = ctrl.Rule(
            self.diversidad['baja'] & self.mejora['mejora_rapida'],
            (self.w['bajo'], self.c1['alto'], self.c2['bajo'])
        )
        reglas.append(regla3)
        
        # Regla 4: Diversidad MEDIA y Mejora ESTANCADA
        # -> Equilibrio pero sin mejora, explorar un poco más
        regla4 = ctrl.Rule(
            self.diversidad['media'] & self.mejora['estancado'],
            (self.w['medio'], self.c1['medio'], self.c2['medio'])
        )
        reglas.append(regla4)
        
        # Regla 5: Diversidad MEDIA y Mejora LENTA
        # -> Situación ideal, seguir así
        regla5 = ctrl.Rule(
            self.diversidad['media'] & self.mejora['mejora_lenta'],
            (self.w['medio'], self.c1['medio'], self.c2['medio'])
        )
        reglas.append(regla5)
        
        # Regla 6: Diversidad MEDIA y Mejora RÁPIDA
        # -> Vamos bien, intensificar explotación
        regla6 = ctrl.Rule(
            self.diversidad['media'] & self.mejora['mejora_rapida'],
            (self.w['bajo'], self.c1['alto'], self.c2['bajo'])
        )
        reglas.append(regla6)
        
        # Regla 7: Diversidad ALTA y Mejora ESTANCADA
        # -> Explorando pero sin encontrar nada bueno, seguir explorando
        regla7 = ctrl.Rule(
            self.diversidad['alta'] & self.mejora['estancado'],
            (self.w['alto'], self.c1['bajo'], self.c2['alto'])
        )
        reglas.append(regla7)
        
        # Regla 8: Diversidad ALTA y Mejora LENTA
        # -> Exploración dando resultados lentos, mantener exploración
        regla8 = ctrl.Rule(
            self.diversidad['alta'] & self.mejora['mejora_lenta'],
            (self.w['alto'], self.c1['bajo'], self.c2['alto'])
        )
        reglas.append(regla8)
        
        # Regla 9: Diversidad ALTA y Mejora RÁPIDA
        # -> Exploración exitosa, empezar a equilibrar
        regla9 = ctrl.Rule(
            self.diversidad['alta'] & self.mejora['mejora_rapida'],
            (self.w['medio'], self.c1['medio'], self.c2['medio'])
        )
        reglas.append(regla9)
        
        self.reglas = reglas
        
    def _crear_sistema_control(self):
        """
        Crea el sistema de control difuso a partir de las reglas.
        """
        self.sistema_ctrl = ctrl.ControlSystem(self.reglas)
        self.sistema_sim = ctrl.ControlSystemSimulation(self.sistema_ctrl)
        
    def calcular_parametros(self, diversidad_val, mejora_val):
        """
        Calcula los nuevos parámetros (w, c1, c2) dados los valores
        actuales de diversidad y mejora.
        
        Parámetros:
        -----------
        diversidad_val : float
            Valor normalizado de diversidad (0-1)
        mejora_val : float
            Valor normalizado de mejora (0-1)
            
        Retorna:
        --------
        tuple : (w_nuevo, c1_nuevo, c2_nuevo)
        """
        # Asignar entradas al sistema difuso
        self.sistema_sim.input['diversidad'] = diversidad_val
        self.sistema_sim.input['mejora'] = mejora_val
        
        # Computar las salidas
        self.sistema_sim.compute()
        
        # Obtener valores defusificados
        w_nuevo = self.sistema_sim.output['w']
        c1_nuevo = self.sistema_sim.output['c1']
        c2_nuevo = self.sistema_sim.output['c2']
        
        return w_nuevo, c1_nuevo, c2_nuevo


# Pequeña prueba para verificar que funciona
if __name__ == "__main__":
    # Crear controlador
    ctrl = ControladorDifusoPSO()
    
    # Probar algunos escenarios
    print("\n--- Prueba del controlador difuso ---")
    
    # Escenario 1: Diversidad baja, mejora estancada (atascado)
    w, c1, c2 = ctrl.calcular_parametros(0.1, 0.05)
    print(f"Escenario 1 (baja, estancado): w={w:.3f}, c1={c1:.3f}, c2={c2:.3f}")
    
    # Escenario 2: Diversidad media, mejora rápida (buena convergencia)
    w, c1, c2 = ctrl.calcular_parametros(0.4, 0.8)
    print(f"Escenario 2 (media, rápida): w={w:.3f}, c1={c1:.3f}, c2={c2:.3f}")
    
    # Escenario 3: Diversidad alta, mejora lenta (explorando)
    w, c1, c2 = ctrl.calcular_parametros(0.8, 0.2)
    print(f"Escenario 3 (alta, lenta): w={w:.3f}, c1={c1:.3f}, c2={c2:.3f}")