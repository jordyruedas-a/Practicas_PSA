# -*- coding: utf-8 -*-
"""
Módulo: pso_adaptativo.py
Descripción: Implementación del algoritmo PSO con control difuso adaptativo.
Autor: Anonimo
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

# Importaciones de nuestros módulos
from controlador_difuso import ControladorDifusoPSO
from funciones_prueba import FuncionesPrueba

class Particula:
    """
    Representa una partícula en el enjambre.
    
    Cada partícula tiene:
    - posición actual en el espacio de búsqueda
    - velocidad actual
    - mejor posición personal encontrada
    - valor de la función en esa mejor posición
    """
    
    def __init__(self, dimensiones, limites):
        """
        Inicializa una partícula con posición y velocidad aleatorias.
        
        Parámetros:
        -----------
        dimensiones : int
            Número de dimensiones del problema
        limites : tuple
            (min, max) para cada dimensión
        """
        self.dimensiones = dimensiones
        self.lim_min, self.lim_max = limites
        
        # Inicializar posición aleatoria dentro de los límites
        self.posicion = np.random.uniform(self.lim_min, self.lim_max, dimensiones)
        
        # Inicializar velocidad aleatoria (normalmente pequeña)
        self.velocidad = np.random.uniform(-1, 1, dimensiones) * (self.lim_max - self.lim_min) * 0.1
        
        # Mejor posición personal (inicialmente la posición actual)
        self.mejor_posicion_personal = self.posicion.copy()
        self.mejor_valor_personal = float('inf')
        
    def actualizar_mejor_personal(self, valor_actual):
        """
        Actualiza la mejor posición personal si el valor actual es mejor.
        """
        if valor_actual < self.mejor_valor_personal:
            self.mejor_valor_personal = valor_actual
            self.mejor_posicion_personal = self.posicion.copy()
            return True
        return False
    
    def __str__(self):
        return f"Partícula en {self.posicion[:3]}... (vel: {self.velocidad[:3]}...)"


class PSOAdaptativo:
    """
    Algoritmo Particle Swarm Optimization con control difuso adaptativo.
    
    El algoritmo ajusta dinámicamente sus parámetros (w, c1, c2) usando
    un controlador difuso que monitorea la diversidad del enjambre y
    la mejora de la mejor solución global.
    """
    
    def __init__(self, 
                 funcion_objetivo,
                 dimensiones=2,
                 num_particulas=30,
                 limites=(-5, 5),
                 max_iteraciones=100,
                 usar_control_difuso=True,
                 nombre_problema="esfera"):
        """
        Inicializa el algoritmo PSO.
        
        Parámetros:
        -----------
        funcion_objetivo : callable
            Función a minimizar
        dimensiones : int
            Número de variables del problema
        num_particulas : int
            Tamaño del enjambre
        limites : tuple
            (mínimo, máximo) para todas las dimensiones
        max_iteraciones : int
            Número máximo de iteraciones
        usar_control_difuso : bool
            Si True, usa control difuso adaptativo
        nombre_problema : str
            Identificador del problema (para visualización)
        """
        self.funcion_objetivo = funcion_objetivo
        self.dimensiones = dimensiones
        self.num_particulas = num_particulas
        self.lim_min, self.lim_max = limites
        self.max_iteraciones = max_iteraciones
        self.usar_control_difuso = usar_control_difuso
        self.nombre_problema = nombre_problema
        
        # Parámetros por defecto (para PSO tradicional)
        self.w = 0.7        # factor de inercia
        self.c1 = 1.5       # coeficiente cognitivo
        self.c2 = 1.5       # coeficiente social
        
        # Inicializar controlador difuso si se usa
        if self.usar_control_difuso:
            self.controlador = ControladorDifusoPSO()
            print(" Controlador difuso activado - parámetros se ajustarán dinámicamente")
        else:
            print(" Modo tradicional - parámetros fijos")
        
        # Inicializar poblaciones
        self.particulas = []
        self._inicializar_poblacion()
        
        # Mejor solución global
        self.mejor_posicion_global = None
        self.mejor_valor_global = float('inf')
        
        # Historial para análisis posterior
        self.historial_mejor_valor = []
        self.historial_diversidad = []
        self.historial_w = []
        self.historial_c1 = []
        self.historial_c2 = []
        
    def _inicializar_poblacion(self):
        """
        Crea todas las partículas del enjambre.
        """
        for i in range(self.num_particulas):
            particula = Particula(self.dimensiones, (self.lim_min, self.lim_max))
            self.particulas.append(particula)
            
    def _evaluar_particula(self, particula):
        """
        Evalúa la función objetivo en la posición de una partícula.
        """
        valor = self.funcion_objetivo(particula.posicion)
        return valor
    
    def _actualizar_mejor_global(self):
        """
        Actualiza la mejor posición global del enjambre.
        """
        for particula in self.particulas:
            if particula.mejor_valor_personal < self.mejor_valor_global:
                self.mejor_valor_global = particula.mejor_valor_personal
                self.mejor_posicion_global = particula.mejor_posicion_personal.copy()
                
    def _calcular_diversidad(self):
        """
        Calcula la diversidad del enjambre.
        
        La diversidad mide qué tan dispersas están las partículas.
        Se normaliza entre 0 y 1.
        """
        if self.num_particulas < 2:
            return 0.0
            
        # Calcular centroide del enjambre
        centroide = np.zeros(self.dimensiones)
        for p in self.particulas:
            centroide += p.posicion
        centroide /= self.num_particulas
        
        # Calcular distancia promedio al centroide
        distancias = 0
        for p in self.particulas:
            distancias += np.linalg.norm(p.posicion - centroide)
        dist_prom = distancias / self.num_particulas
        
        # Normalizar por el rango de búsqueda
        rango = self.lim_max - self.lim_min
        diversidad_norm = min(1.0, dist_prom / rango)
        
        return diversidad_norm
    
    def _calcular_mejora(self, ventana=10):
        """
        Calcula la tasa de mejora de la mejor solución global.
        
        Parámetros:
        -----------
        ventana : int
            Número de iteraciones para considerar
            
        Retorna:
        --------
        float : valor entre 0 y 1 (1 = mejora rápida, 0 = estancado)
        """
        if len(self.historial_mejor_valor) < 2:
            return 0.5  # valor neutro al inicio
        
        # Valor actual y valor hace 'ventana' iteraciones
        valor_actual = self.historial_mejor_valor[-1]
        idx_anterior = max(0, len(self.historial_mejor_valor) - ventana - 1)
        valor_anterior = self.historial_mejor_valor[idx_anterior]
        
        # Si no ha mejorado, mejora = 0
        if valor_anterior <= valor_actual:
            return 0.0
        
        # Calcular mejora relativa
        mejora_rel = (valor_anterior - valor_actual) / abs(valor_anterior) if valor_anterior != 0 else 1.0
        
        # Normalizar y acotar
        mejora_norm = min(1.0, max(0.0, mejora_rel))
        
        return mejora_norm
    
    def _ajustar_parametros(self, diversidad, mejora):
        """
        Ajusta los parámetros del PSO usando el controlador difuso.
        """
        if not self.usar_control_difuso:
            # Parámetros fijos
            return self.w, self.c1, self.c2
        
        # Usar controlador difuso
        w_nuevo, c1_nuevo, c2_nuevo = self.controlador.calcular_parametros(diversidad, mejora)
        
        return w_nuevo, c1_nuevo, c2_nuevo
    
    def ejecutar(self, verbose=True, guardar_historial=True):
        """
        Ejecuta el algoritmo PSO.
        
        Parámetros:
        -----------
        verbose : bool
            Si True, muestra progreso
        guardar_historial : bool
            Si True, guarda el historial para análisis
        """
        print(f"\n--- Ejecutando PSO en {self.nombre_problema} ---")
        print(f"Partículas: {self.num_particulas}, Dimensiones: {self.dimensiones}")
        print(f"Modo: {'ADAPTATIVO' if self.usar_control_difuso else 'TRADICIONAL'}")
        
        inicio_tiempo = time.time()
        
        for iteracion in range(self.max_iteraciones):
            # Evaluar cada partícula
            for particula in self.particulas:
                valor_actual = self._evaluar_particula(particula)
                particula.actualizar_mejor_personal(valor_actual)
            
            # Actualizar mejor global
            self._actualizar_mejor_global()
            
            # Guardar historial
            if guardar_historial:
                self.historial_mejor_valor.append(self.mejor_valor_global)
                diversidad_actual = self._calcular_diversidad()
                self.historial_diversidad.append(diversidad_actual)
            
            # Calcular diversidad y mejora para ajuste
            diversidad = self._calcular_diversidad()
            mejora = self._calcular_mejora()
            
            # Ajustar parámetros
            self.w, self.c1, self.c2 = self._ajustar_parametros(diversidad, mejora)
            
            if guardar_historial:
                self.historial_w.append(self.w)
                self.historial_c1.append(self.c1)
                self.historial_c2.append(self.c2)
            
            # Actualizar velocidad y posición de cada partícula
            for particula in self.particulas:
                # Generar números aleatorios
                r1 = np.random.random(self.dimensiones)
                r2 = np.random.random(self.dimensiones)
                
                # Componente cognitivo (hacia mejor personal)
                cognitivo = self.c1 * r1 * (particula.mejor_posicion_personal - particula.posicion)
                
                # Componente social (hacia mejor global)
                social = self.c2 * r2 * (self.mejor_posicion_global - particula.posicion)
                
                # Actualizar velocidad
                particula.velocidad = self.w * particula.velocidad + cognitivo + social
                
                # Actualizar posición
                particula.posicion = particula.posicion + particula.velocidad
                
                # Aplicar límites (si se sale, rebotar o truncar)
                for dim in range(self.dimensiones):
                    if particula.posicion[dim] < self.lim_min:
                        particula.posicion[dim] = self.lim_min
                        particula.velocidad[dim] *= -0.5  # rebote suave
                    elif particula.posicion[dim] > self.lim_max:
                        particula.posicion[dim] = self.lim_max
                        particula.velocidad[dim] *= -0.5
            
            # Mostrar progreso cada 10 iteraciones
            if verbose and (iteracion % 10 == 0 or iteracion == self.max_iteraciones - 1):
                print(f"Iter {iteracion:3d} | Mejor valor: {self.mejor_valor_global:.6f} | "
                      f"w={self.w:.3f}, c1={self.c1:.3f}, c2={self.c2:.3f} | "
                      f"Div={diversidad:.3f}, Mej={mejora:.3f}")
        
        tiempo_total = time.time() - inicio_tiempo
        print(f"\n Finalizado en {tiempo_total:.2f} segundos")
        print(f" Mejor valor encontrado: {self.mejor_valor_global:.10f}")
        print(f" Posición: {self.mejor_posicion_global}")
        
        return self.mejor_valor_global, self.mejor_posicion_global
    
    def graficar_convergencia(self, guardar=False, nombre_archivo=None):
        """
        Grafica la convergencia del algoritmo.
        """
        plt.figure(figsize=(12, 8))
        
        # Gráfica de convergencia
        plt.subplot(2, 2, 1)
        plt.plot(self.historial_mejor_valor, 'b-', linewidth=2)
        plt.xlabel('Iteración')
        plt.ylabel('Mejor valor')
        plt.title('Convergencia del algoritmo')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        # Gráfica de diversidad
        plt.subplot(2, 2, 2)
        plt.plot(self.historial_diversidad, 'g-', linewidth=2)
        plt.xlabel('Iteración')
        plt.ylabel('Diversidad')
        plt.title('Diversidad del enjambre')
        plt.grid(True, alpha=0.3)
        
        # Gráfica de parámetros (w, c1, c2)
        plt.subplot(2, 2, 3)
        plt.plot(self.historial_w, 'r-', label='w (inercia)', linewidth=2)
        plt.plot(self.historial_c1, 'g-', label='c1 (cognitivo)', linewidth=2)
        plt.plot(self.historial_c2, 'b-', label='c2 (social)', linewidth=2)
        plt.xlabel('Iteración')
        plt.ylabel('Valor del parámetro')
        plt.title('Evolución de parámetros adaptativos')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Título general
        plt.suptitle(f'PSO {("ADAPTATIVO" if self.usar_control_difuso else "TRADICIONAL")} - {self.nombre_problema}', 
                     fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if guardar:
            if nombre_archivo is None:
                nombre_archivo = f"convergencia_{self.nombre_problema}.png"
            plt.savefig(nombre_archivo, dpi=150, bbox_inches='tight')
            print(f" Gráfica guardada como '{nombre_archivo}'")
        
        plt.show()
    
    def animar_trayectoria(self, guardar=False):
        """
        Crea una animación de la trayectoria de las partículas.
        Solo para problemas de 2 dimensiones.
        """
        if self.dimensiones != 2:
            print("Animación solo disponible para 2 dimensiones")
            return
        
        # Crear malla para el contour
        x = np.linspace(self.lim_min, self.lim_max, 100)
        y = np.linspace(self.lim_min, self.lim_max, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = self.funcion_objetivo(np.array([X[i, j], Y[i, j]]))
        
        fig, ax = plt.subplots(figsize=(10, 8))
        contour = ax.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.7)
        plt.colorbar(contour, ax=ax, label='Valor función')
        
        # Puntos para las partículas
        scatter = ax.scatter([], [], c='red', s=50, alpha=0.7, edgecolors='white')
        best_scatter = ax.scatter([], [], c='yellow', s=200, marker='*', 
                                 edgecolors='black', linewidth=2, label='Mejor global')
        
        ax.set_xlim(self.lim_min, self.lim_max)
        ax.set_ylim(self.lim_min, self.lim_max)
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_title(f'Evolución del enjambre - {self.nombre_problema}')
        ax.legend()
        
        # Función de animación
        def update(frame):
            # Aquí necesitaríamos tener el historial de posiciones
            # Por simplicidad, solo mostramos la posición actual
            xs = [p.posicion[0] for p in self.particulas]
            ys = [p.posicion[1] for p in self.particulas]
            scatter.set_offsets(np.c_[xs, ys])
            
            if self.mejor_posicion_global is not None:
                best_scatter.set_offsets([self.mejor_posicion_global])
            
            ax.set_title(f'Iteración {frame}')
            return scatter, best_scatter
        
        ani = FuncAnimation(fig, update, frames=self.max_iteraciones, 
                          interval=100, blit=True, repeat=False)
        
        if guardar:
            ani.save(f'trayectoria_{self.nombre_problema}.gif', writer='pillow', fps=10)
            print(f" Animación guardada")
        
        plt.show()


# Prueba rápida
if __name__ == "__main__":
    from funciones_prueba import FuncionesPrueba
    
    # Crear y ejecutar PSO adaptativo en función esfera
    pso = PSOAdaptativo(
        funcion_objetivo=FuncionesPrueba.esfera,
        dimensiones=2,
        num_particulas=20,
        limites=(-5, 5),
        max_iteraciones=50,
        usar_control_difuso=True,
        nombre_problema="esfera_2d"
    )
    
    mejor_valor, mejor_pos = pso.ejecutar(verbose=True)
    pso.graficar_convergencia(guardar=True)