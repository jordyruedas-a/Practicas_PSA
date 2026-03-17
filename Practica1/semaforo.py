# Explicación detallada del código
import tkinter as tk
from collections import deque
import random
import time
import math

# Clase Auto
class Auto:
    """Representa un vehículo con un identificador único."""
    _id_counter = 0

    def __init__(self):
        Auto._id_counter += 1
        self.id = Auto._id_counter

    def __repr__(self):
        return f"Auto({self.id})"

# Clase Carril
class Carril:
    """Cola de autos que esperan para pasar."""
    def __init__(self):
        self.cola = deque()  # cola de objetos Auto

    def agregar_auto(self, auto):
        self.cola.append(auto)

    def quitar_auto(self):
        """Saca y retorna el primer auto si existe, si no retorna None."""
        if self.cola:
            return self.cola.popleft()
        return None

    def longitud(self):
        return len(self.cola)

    def vacio(self):
        return len(self.cola) == 0

# Clase Semaforo
class Semaforo:
    """Semáforo con estado y método para cambiar."""
    def __init__(self):
        self.estado = "rojo"  # "rojo", "verde", "amarillo"

    def cambiar_estado(self, nuevo_estado):
        self.estado = nuevo_estado

    def es_verde(self):
        return self.estado == "verde"

# Clase Movimiento
class Movimiento:
    """Un flujo de tráfico: desde un origen hacia un destino."""
    def __init__(self, nombre, tasa_llegada):
        self.nombre = nombre
        self.carril = Carril()
        self.semaforo = Semaforo()
        self.tasa_llegada = tasa_llegada  # probabilidad de llegada por tick

    def generar_auto(self):
        """Crea un auto y lo agrega al carril."""
        auto = Auto()
        self.carril.agregar_auto(auto)

    def liberar_auto(self):
        """Si el semáforo está verde, saca un auto del carril y lo retorna."""
        if self.semaforo.es_verde() and not self.carril.vacio():
            return self.carril.quitar_auto()
        return None

# Clase Interseccion
class Interseccion:
    """Modela la intersección con todos los movimientos y sus conflictos."""
    def __init__(self):
        # Definimos cuatro movimientos: NS, SN, EW, WE
        self.movimientos = {
            "NS": Movimiento("Norte -> Sur", 0.3),
            "SN": Movimiento("Sur -> Norte", 0.3),
            "EW": Movimiento("Este -> Oeste", 0.2),
            "WE": Movimiento("Oeste -> Este", 0.2)
        }

        # Matriz de conflictos: 1 si los movimientos no pueden estar verdes juntos
        # Orden: NS, SN, EW, WE
        self.matriz_conflictos = [
            [0, 0, 1, 1],  # NS conflictúa con EW y WE
            [0, 0, 1, 1],  # SN conflictúa con EW y WE
            [1, 1, 0, 0],  # EW conflictúa con NS y SN
            [1, 1, 0, 0]   # WE conflictúa con NS y SN
        ]
        self.lista_mov = list(self.movimientos.keys())

    def movimientos_conflictivos(self, mov1, mov2):
        """Retorna True si mov1 y mov2 están en conflicto."""
        i = self.lista_mov.index(mov1)
        j = self.lista_mov.index(mov2)
        return self.matriz_conflictos[i][j] == 1

    def fase_no_conflictiva(self, conjunto_movs):
        """Verifica si un conjunto de movimientos es no conflictivo entre sí."""
        for i in range(len(conjunto_movs)):
            for j in range(i+1, len(conjunto_movs)):
                if self.movimientos_conflictivos(conjunto_movs[i], conjunto_movs[j]):
                    return False
        return True

# Clase Controlador
class Controlador:
    """Asigna dinámicamente los tiempos de verde basado en las colas."""
    def __init__(self, interseccion, ciclo_base=20, amarillo=3):
        self.interseccion = interseccion
        self.ciclo_base = ciclo_base      # ticks base por fase
        self.amarillo = amarillo           # ticks de amarillo entre fases
        self.fase_actual = "NS"            # fase actual: "NS" o "EW"
        self.tiempo_restante = 0           # ticks restantes de la fase actual
        self.en_amarillo = False

    def calcular_tiempos(self):
        """Calcula los tiempos de verde para cada calle según colas."""
        colas = self.interseccion.movimientos
        # Suma de colas en cada calle
        suma_ns = colas["NS"].carril.longitud() + colas["SN"].carril.longitud()
        suma_ew = colas["EW"].carril.longitud() + colas["WE"].carril.longitud()
        total = suma_ns + suma_ew
        if total == 0:
            # Si no hay autos, repartir equitativamente
            return self.ciclo_base, self.ciclo_base
        # Proporcional a la demanda
        tiempo_ns = max(5, int((suma_ns / total) * 2 * self.ciclo_base))
        tiempo_ew = max(5, int((suma_ew / total) * 2 * self.ciclo_base))
        return tiempo_ns, tiempo_ew

    def actualizar(self, tick):
        """Actualiza el estado de los semáforos en cada tick."""
        # Si estamos en amarillo, contar y luego cambiar
        if self.en_amarillo:
            self.tiempo_restante -= 1
            if self.tiempo_restante <= 0:
                # Termina amarillo, pasamos a la siguiente fase
                self.en_amarillo = False
                self._aplicar_fase()
            return

        # Si no hay tiempo restante en la fase actual, hay que cambiar
        if self.tiempo_restante <= 0:
            # Poner todos en amarillo antes de cambiar
            self._poner_amarillo()
            self.en_amarillo = True
            self.tiempo_restante = self.amarillo
        else:
            # Restar un tick a la fase actual
            self.tiempo_restante -= 1

    def _poner_amarillo(self):
        """Pone todos los semáforos en amarillo."""
        for mov in self.interseccion.movimientos.values():
            mov.semaforo.cambiar_estado("amarillo")

    def _aplicar_fase(self):
        """Cambia a la siguiente fase con los tiempos calculados."""
        # Alternar fase
        if self.fase_actual == "NS":
            self.fase_actual = "EW"
            tiempo_ns, tiempo_ew = self.calcular_tiempos()
            self.tiempo_restante = tiempo_ew
            # Poner en verde los movimientos EW y WE
            self.interseccion.movimientos["EW"].semaforo.cambiar_estado("verde")
            self.interseccion.movimientos["WE"].semaforo.cambiar_estado("verde")
            self.interseccion.movimientos["NS"].semaforo.cambiar_estado("rojo")
            self.interseccion.movimientos["SN"].semaforo.cambiar_estado("rojo")
        else:
            self.fase_actual = "NS"
            tiempo_ns, tiempo_ew = self.calcular_tiempos()
            self.tiempo_restante = tiempo_ns
            self.interseccion.movimientos["NS"].semaforo.cambiar_estado("verde")
            self.interseccion.movimientos["SN"].semaforo.cambiar_estado("verde")
            self.interseccion.movimientos["EW"].semaforo.cambiar_estado("rojo")
            self.interseccion.movimientos["WE"].semaforo.cambiar_estado("rojo")

# Clase Simulacion
class Simulacion:
    """Maneja el bucle principal de la simulación y la comunicación con la GUI."""
    def __init__(self, interseccion, controlador, interfaz):
        self.interseccion = interseccion
        self.controlador = controlador
        self.interfaz = interfaz
        self.tick_actual = 0
        self.corriendo = False

    def iniciar(self):
        self.corriendo = True
        self._bucle()

    def detener(self):
        self.corriendo = False

    def _bucle(self):
        if not self.corriendo:
            return
        # Avanzar un tick
        self.tick_actual += 1

        # 1. Generar llegadas aleatorias
        for mov in self.interseccion.movimientos.values():
            if random.random() < mov.tasa_llegada:
                mov.generar_auto()

        # 2. Liberar autos según semáforos
        for mov in self.interseccion.movimientos.values():
            mov.liberar_auto()

        # 3. Actualizar controlador (cambios de semáforo)
        self.controlador.actualizar(self.tick_actual)

        # 4. Actualizar interfaz gráfica
        self.interfaz.actualizar()

        # Programar el siguiente tick (cada 200 ms)
        self.interfaz.root.after(200, self._bucle)

# Clase Interfaz (tkinter) - MODIFICADA para centrar semáforos y calles completas
class Interfaz:
    """Ventana principal con dibujo de la intersección."""
    def __init__(self, interseccion):
        self.interseccion = interseccion
        self.root = tk.Tk()
        self.root.title("Simulación de Semáforos Adaptativos")
        self.canvas = tk.Canvas(self.root, width=600, height=600, bg="white")
        self.canvas.pack()

        # Dibujar elementos estáticos
        self._dibujar_base()

        # Almacenar referencias a los objetos gráficos para actualizarlos
        self.textos_colas = {}
        self.semaforos_circulos = {}

        self._crear_elementos_dinamicos()

    def _dibujar_base(self):
        """Dibuja las calles extendidas hasta los bordes y la intersección."""
        # Calle horizontal (de borde a borde)
        self.canvas.create_rectangle(0, 250, 600, 350, fill="gray", outline="black")
        # Calle vertical (de borde a borde)
        self.canvas.create_rectangle(250, 0, 350, 600, fill="gray", outline="black")
        # Líneas de carriles (dash) también extendidas
        self.canvas.create_line(250, 0, 250, 600, fill="white", dash=(4, 2))
        self.canvas.create_line(350, 0, 350, 600, fill="white", dash=(4, 2))
        self.canvas.create_line(0, 250, 600, 250, fill="white", dash=(4, 2))
        self.canvas.create_line(0, 350, 600, 350, fill="white", dash=(4, 2))

        # Etiquetas de direcciones en los bordes
        self.canvas.create_text(300, 50, text="NORTE", font=("Arial", 10))
        self.canvas.create_text(300, 550, text="SUR", font=("Arial", 10))
        self.canvas.create_text(50, 300, text="OESTE", font=("Arial", 10))
        self.canvas.create_text(550, 300, text="ESTE", font=("Arial", 10))

    def _crear_elementos_dinamicos(self):
        """Crea los textos para las colas y los círculos de semáforos centrados."""
        # Posiciones de los semáforos (centrados en cada acceso)
        self.pos_semaforos = {
            "NS": (300, 230),   # norte (para autos que van al sur)
            "SN": (300, 370),   # sur (para autos que van al norte)
            "EW": (370, 300),   # este (para autos que van al oeste)
            "WE": (230, 300)    # oeste (para autos que van al este)
        }
        # Posiciones de los textos de cola (junto a cada semáforo)
        self.pos_colas = {
            "NS": (300, 200),   # arriba del semáforo norte
            "SN": (300, 400),   # abajo del semáforo sur
            "EW": (400, 300),   # derecha del semáforo este
            "WE": (200, 300)    # izquierda del semáforo oeste
        }

        for mov, (x, y) in self.pos_colas.items():
            # Texto que mostrará la longitud de la cola
            texto = self.canvas.create_text(x, y, text="0", font=("Arial", 14, "bold"), fill="blue")
            self.textos_colas[mov] = texto

        for mov, (x, y) in self.pos_semaforos.items():
            circulo = self.canvas.create_oval(x-10, y-10, x+10, y+10, fill="red", outline="black")
            self.semaforos_circulos[mov] = circulo

    def actualizar(self):
        """Actualiza los valores de las colas y colores de semáforos."""
        for mov in self.interseccion.movimientos:
            # Actualizar texto de cola
            longitud = self.interseccion.movimientos[mov].carril.longitud()
            self.canvas.itemconfig(self.textos_colas[mov], text=str(longitud))

            # Actualizar color del semáforo
            estado = self.interseccion.movimientos[mov].semaforo.estado
            color = {"verde": "green", "amarillo": "yellow", "rojo": "red"}[estado]
            self.canvas.itemconfig(self.semaforos_circulos[mov], fill=color)

        self.root.update()

    def ejecutar(self):
        self.root.mainloop()

# Programa principal
if __name__ == "__main__":
    # Crear intersección
    interseccion = Interseccion()

    # Crear controlador
    controlador = Controlador(interseccion, ciclo_base=20, amarillo=3)

    # Crear interfaz (con las mejoras gráficas)
    interfaz = Interfaz(interseccion)

    # Crear simulación
    simulacion = Simulacion(interseccion, controlador, interfaz)

    # Iniciar simulación al abrir la ventana
    interfaz.root.after(100, simulacion.iniciar)

    # Ejecutar interfaz
    interfaz.ejecutar()
