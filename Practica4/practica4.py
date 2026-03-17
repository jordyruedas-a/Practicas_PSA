"""
Descripción: Implementa un sistema multiagente para simular subastas con múltiples
             compradores autónomos y un subastador coordinador.
Autor: Anonimo
Fecha: 13/03/2026
"""

# =============================================================================
# IMPORTACIÓN DE LIBRERÍAS
# =============================================================================
import threading
import queue
import time
import random
import matplotlib.pyplot as plt
from collections import defaultdict

# =============================================================================
# CLASE MENSAJE (para comunicación entre agentes)
# =============================================================================
class Mensaje:
    """
    Representa un mensaje intercambiado entre agentes.
    Atributos:
        remitente: nombre del agente que envía
        destinatario: nombre del agente destino
        tipo: tipo de mensaje (PUJA, NOTIFICACION, etc.)
        contenido: datos adicionales (puede ser un diccionario)
    """
    def __init__(self, remitente, destinatario, tipo, contenido=None):
        self.remitente = remitente
        self.destinatario = destinatario
        self.tipo = tipo
        self.contenido = contenido if contenido else {}
    
    def __str__(self):
        return f"[{self.remitente} -> {self.destinatario}] {self.tipo}: {self.contenido}"

# =============================================================================
# CLASE BASE AGENTE
# =============================================================================
class Agente(threading.Thread):
    """
    Clase base abstracta para todos los agentes.
    Gestiona la cola de mensajes y el bucle de ejecución en un hilo separado.
    """
    
    # Contador estático para IDs únicos
    _contador_ids = 0
    _lock_contador = threading.Lock()
    
    def __init__(self, nombre):
        super().__init__()
        with Agente._lock_contador:
            Agente._contador_ids += 1
            self.id = Agente._contador_ids
        self.nombre = nombre
        self.cola_mensajes = queue.Queue()
        self.activo = True
        self.daemon = True  # Los hilos terminan cuando el programa principal termina
        
    def enviar_mensaje(self, destinatario, tipo, contenido=None):
        """
        Envía un mensaje a otro agente colocándolo en su cola.
        """
        mensaje = Mensaje(self.nombre, destinatario.nombre, tipo, contenido)
        destinatario.cola_mensajes.put(mensaje)
        
    def recibir_mensaje(self, timeout=0.1):
        """
        Intenta recibir un mensaje de la cola (no bloqueante con timeout).
        Retorna el mensaje o None si no hay.
        """
        try:
            return self.cola_mensajes.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def procesar_mensaje(self, mensaje):
        """
        Método a sobrescribir por las subclases.
        Define cómo reacciona el agente ante cada tipo de mensaje.
        """
        raise NotImplementedError("Las subclases deben implementar procesar_mensaje")
    
    def run(self):
        """
        Bucle principal del agente (se ejecuta en su hilo).
        """
        while self.activo:
            mensaje = self.recibir_mensaje()
            if mensaje:
                self.procesar_mensaje(mensaje)
            else:
                # Si no hay mensajes, el agente puede realizar otras tareas
                self.en_espera()
    
    def en_espera(self):
        """
        Comportamiento por defecto cuando no hay mensajes.
        Las subclases pueden sobrescribirlo.
        """
        time.sleep(0.05)  # Pequeña pausa para no saturar la CPU
    
    def detener(self):
        """
        Detiene el agente.
        """
        self.activo = False
    
    def __repr__(self):
        return f"Agente({self.nombre})"

# =============================================================================
# CLASE ARTÍCULO
# =============================================================================
class Articulo:
    """
    Representa un artículo a subastar.
    """
    def __init__(self, id_articulo, nombre, precio_base):
        self.id = id_articulo
        self.nombre = nombre
        self.precio_base = precio_base
        self.precio_actual = precio_base
        self.puja_ganadora = None
        self.comprador_ganador = None
        
    def __str__(self):
        return f"{self.nombre} (base: {self.precio_base})"

# =============================================================================
# CLASE AGENTE COMPRADOR
# =============================================================================
class AgenteComprador(Agente):
    """
    Agente que participa en la subasta pujando por los artículos.
    """
    
    def __init__(self, nombre, presupuesto, estrategia="conservadora"):
        super().__init__(nombre)
        self.presupuesto = presupuesto
        self.presupuesto_inicial = presupuesto
        self.estrategia = estrategia  # "conservadora", "agresiva", "aleatoria"
        self.articulos_comprados = []
        self.total_gastado = 0
        self.ultima_puja = 0
        self.articulo_actual = None
        self.precio_actual_subasta = 0
        
    def calcular_incremento(self, precio_actual):
        """
        Calcula el incremento a añadir a la puja según la estrategia.
        """
        if self.estrategia == "conservadora":
            # Incremento del 5% al 10%
            return precio_actual * random.uniform(0.05, 0.10)
        elif self.estrategia == "agresiva":
            # Incremento del 15% al 25%
            return precio_actual * random.uniform(0.15, 0.25)
        elif self.estrategia == "aleatoria":
            # Incremento completamente aleatorio entre 5% y 30%
            return precio_actual * random.uniform(0.05, 0.30)
        else:
            return precio_actual * 0.10  # valor por defecto
    
    def decidir_puja(self, precio_actual, id_articulo):
        """
        Decide si puja y por qué cantidad.
        Retorna el monto de la puja o None si no puja.
        """
        # No pujar si ya no tiene presupuesto
        if self.presupuesto <= 0:
            return None
        
        # La nueva puja debe ser superior a la actual
        incremento = self.calcular_incremento(precio_actual)
        nueva_puja = precio_actual + incremento
        
        # Redondear a 2 decimales
        nueva_puja = round(nueva_puja, 2)
        
        # Verificar que no exceda el presupuesto
        if nueva_puja > self.presupuesto:
            # Intentar con el presupuesto máximo (estrategia "todo o nada")
            if self.estrategia == "agresiva" and self.presupuesto > precio_actual:
                # El agresivo gasta todo lo que le queda si puede ganar
                return self.presupuesto
            else:
                return None
        
        return nueva_puja
    
    def procesar_mensaje(self, mensaje):
        """
        Procesa los mensajes recibidos.
        """
        tipo = mensaje.tipo
        contenido = mensaje.contenido
        
        if tipo == "INICIO_ARTICULO":
            # El subastador anuncia un nuevo artículo
            self.articulo_actual = contenido.get("articulo")
            self.precio_actual_subasta = contenido.get("precio_inicial", 0)
            print(f"  [Comprador {self.nombre}] Atento a subasta de {self.articulo_actual}")
            
        elif tipo == "ACTUALIZACION_PUJA":
            # Alguien ha pujado, actualizamos el precio
            nuevo_precio = contenido.get("precio_actual")
            pujador = contenido.get("pujador")
            if nuevo_precio:
                self.precio_actual_subasta = nuevo_precio
                # Si el pujador no soy yo, puedo contraatacar
                if pujador != self.nombre:
                    # Pequeña pausa para simular tiempo de reacción
                    time.sleep(random.uniform(0.1, 0.5))
                    self.evaluar_contraoferta()
        
        elif tipo == "RESULTADO_SUBASTA":
            # Se anuncia el ganador del artículo actual
            ganador = contenido.get("ganador")
            precio_final = contenido.get("precio_final")
            if ganador == self.nombre:
                print(f"  🏆 [Comprador {self.nombre}] GANÉ {self.articulo_actual} por {precio_final}")
                self.articulos_comprados.append({
                    "articulo": self.articulo_actual,
                    "precio": precio_final
                })
                self.total_gastado += precio_final
                self.presupuesto -= precio_final
            else:
                print(f"  [Comprador {self.nombre}] Perdí {self.articulo_actual} (ganó {ganador})")
        
        elif tipo == "FIN_SIMULACION":
            print(f"  [Comprador {self.nombre}] Simulación finalizada. Compré {len(self.articulos_comprados)} artículos.")
            self.detener()
    
    def evaluar_contraoferta(self):
        """
        Evalúa si debe realizar una contraoferta.
        Se llama cuando otro agente ha pujado.
        """
        if self.articulo_actual and self.presupuesto > self.precio_actual_subasta:
            monto_puja = self.decidir_puja(self.precio_actual_subasta, self.articulo_actual)
            if monto_puja:
                # Enviar puja al subastador
                # (Necesitaríamos tener referencia al subastador - se pasa al iniciar)
                # Por simplicidad, este método será llamado desde el bucle principal
                # con conocimiento del subastador.
                pass
    
    def en_espera(self):
        """
        Comportamiento cuando no hay mensajes.
        """
        # No hacer nada, solo esperar
        time.sleep(0.1)

# =============================================================================
# CLASE AGENTE SUBASTADOR
# =============================================================================
class AgenteSubastador(Agente):
    """
    Agente que coordina las subastas.
    """
    
    def __init__(self, nombre, articulos, tiempo_espera_max=3.0):
        super().__init__(nombre)
        self.articulos = articulos  # lista de objetos Articulo
        self.tiempo_espera_max = tiempo_espera_max  # segundos sin pujas para cerrar
        self.compradores = []  # lista de referencias a compradores
        self.subasta_activa = False
        self.articulo_actual = None
        self.precio_actual = 0
        self.ultimo_pujador = None
        self.ultimo_tiempo_puja = 0
        
    def registrar_compradores(self, lista_compradores):
        """
        Guarda referencias a los compradores para poder notificarles.
        """
        self.compradores = lista_compradores
    
    def iniciar_subasta(self):
        """
        Inicia el proceso de subasta para todos los artículos.
        """
        print(f"\n📢 [Subastador {self.nombre}] Comenzando subasta con {len(self.articulos)} artículos")
        
        for articulo in self.articulos:
            self.subastar_articulo(articulo)
        
        # Fin de todas las subastas
        self.notificar_fin()
    
    def subastar_articulo(self, articulo):
        """
        Realiza la subasta para un artículo específico.
        """
        self.articulo_actual = articulo
        self.precio_actual = articulo.precio_base
        self.ultimo_pujador = None
        self.subasta_activa = True
        
        print(f"\n🆕 [Subastador] Subastando: {articulo.nombre} (precio base: {articulo.precio_base})")
        
        # Notificar a todos los compradores
        self.notificar_inicio_articulo(articulo)
        
        # Bucle de recepción de pujas
        inicio_espera = time.time()
        while self.subasta_activa:
            mensaje = self.recibir_mensaje(timeout=0.5)
            if mensaje:
                if mensaje.tipo == "PUJA":
                    self.procesar_puja(mensaje)
                    inicio_espera = time.time()  # reiniciar contador
            else:
                # Verificar si ha pasado el tiempo máximo sin pujas
                if time.time() - inicio_espera > self.tiempo_espera_max:
                    self.cerrar_subasta_articulo()
        
        # Esperar un poco antes del siguiente artículo
        time.sleep(1)
    
    def procesar_puja(self, mensaje):
        """
        Procesa una puja recibida.
        """
        comprador = mensaje.remitente
        monto = mensaje.contenido.get("monto", 0)
        id_articulo = mensaje.contenido.get("id_articulo")
        
        # Verificar que corresponda al artículo actual
        if id_articulo != self.articulo_actual.id:
            print(f"  [Subastador] Puja de {comprador} para artículo incorrecto (ignorada)")
            return
        
        # Verificar que la puja sea superior al precio actual
        if monto <= self.precio_actual:
            print(f"  [Subastador] Puja de {comprador} por {monto} no válida (debe superar {self.precio_actual})")
            return
        
        # Aceptar puja
        self.precio_actual = monto
        self.ultimo_pujador = comprador
        self.ultimo_tiempo_puja = time.time()
        
        print(f"  ⚡ [Subastador] Nueva puja: {comprador} ofrece {monto} por {self.articulo_actual.nombre}")
        
        # Notificar actualización a todos los compradores
        self.notificar_actualizacion(comprador, monto)
    
    def cerrar_subasta_articulo(self):
        """
        Cierra la subasta del artículo actual y declara ganador.
        """
        self.subasta_activa = False
        
        if self.ultimo_pujador:
            # Hay ganador
            self.articulo_actual.precio_actual = self.precio_actual
            self.articulo_actual.puja_ganadora = self.precio_actual
            self.articulo_actual.comprador_ganador = self.ultimo_pujador
            
            print(f"  ✅ [Subastador] {self.articulo_actual.nombre} adjudicado a {self.ultimo_pujador} por {self.precio_actual}")
            
            # Notificar resultado a todos
            self.notificar_resultado(ganador=self.ultimo_pujador, precio=self.precio_actual)
        else:
            # Nadie pujó
            print(f"  ❌ [Subastador] {self.articulo_actual.nombre} no recibió pujas - desierto")
            self.notificar_resultado(ganador=None, precio=None)
    
    def notificar_inicio_articulo(self, articulo):
        """
        Envía mensaje de inicio a todos los compradores.
        """
        contenido = {
            "articulo": articulo.nombre,
            "id_articulo": articulo.id,
            "precio_inicial": articulo.precio_base
        }
        for comprador in self.compradores:
            self.enviar_mensaje(comprador, "INICIO_ARTICULO", contenido)
    
    def notificar_actualizacion(self, pujador, nuevo_precio):
        """
        Envía actualización de precio a todos los compradores.
        """
        contenido = {
            "pujador": pujador,
            "precio_actual": nuevo_precio,
            "id_articulo": self.articulo_actual.id
        }
        for comprador in self.compradores:
            self.enviar_mensaje(comprador, "ACTUALIZACION_PUJA", contenido)
    
    def notificar_resultado(self, ganador, precio):
        """
        Envía el resultado de la subasta a todos.
        """
        contenido = {
            "ganador": ganador,
            "precio_final": precio,
            "articulo": self.articulo_actual.nombre
        }
        for comprador in self.compradores:
            self.enviar_mensaje(comprador, "RESULTADO_SUBASTA", contenido)
    
    def notificar_fin(self):
        """
        Notifica que la simulación ha terminado.
        """
        for comprador in self.compradores:
            self.enviar_mensaje(comprador, "FIN_SIMULACION", {})
    
    def procesar_mensaje(self, mensaje):
        """
        Procesa mensajes dirigidos al subastador.
        """
        if mensaje.tipo == "PUJA":
            # Las pujas se procesan en el bucle principal de subasta
            # Aquí simplemente las reenviamos a la cola para ser procesadas
            # (ya están en la cola, este método se llama automáticamente)
            pass
    
    def run(self):
        """
        Bucle del subastador (inicia el proceso).
        """
        # Pequeña espera para que todos los compradores estén listos
        time.sleep(1)
        self.iniciar_subasta()
        # Al terminar, seguir vivo para posibles mensajes finales
        while self.activo:
            mensaje = self.recibir_mensaje(timeout=0.5)
            if mensaje:
                self.procesar_mensaje(mensaje)
            else:
                time.sleep(0.1)

# =============================================================================
# CLASE SISTEMA DE SUBASTAS (orquestador)
# =============================================================================
class SistemaSubastas:
    """
    Clase principal que configura y ejecuta la simulación.
    """
    
    def __init__(self, num_compradores=5, num_articulos=10):
        self.num_compradores = num_compradores
        self.num_articulos = num_articulos
        self.compradores = []
        self.subastador = None
        self.articulos = []
        
    def generar_articulos(self):
        """
        Genera una lista de artículos aleatorios.
        """
        nombres = [
            "Smartphone", "Laptop", "Tablet", "Auriculares", "Smartwatch",
            "Cámara", "Impresora", "Monitor", "Teclado", "Ratón",
            "Altavoz", "Disco SSD", "Memoria RAM", "Tarjeta gráfica", "Procesador",
            "Silla ergonómica", "Escritorio", "Lámpara LED", "Mochila", "Libro"
        ]
        self.articulos = []
        for i in range(self.num_articulos):
            nombre = random.choice(nombres) + f" #{i+1}"
            precio_base = round(random.uniform(100, 800), 2)
            self.articulos.append(Articulo(i+1, nombre, precio_base))
    
    def crear_compradores(self):
        """
        Crea los agentes compradores con presupuestos aleatorios.
        """
        estrategias = ["conservadora", "agresiva", "aleatoria"]
        self.compradores = []
        for i in range(self.num_compradores):
            nombre = f"Comprador_{i+1}"
            presupuesto = round(random.uniform(1000, 5000), 2)
            estrategia = random.choice(estrategias)
            comprador = AgenteComprador(nombre, presupuesto, estrategia)
            self.compradores.append(comprador)
    
    def ejecutar(self):
        """
        Ejecuta la simulación completa.
        """
        print("=" * 70)
        print("SISTEMA MULTIAGENTE DE SUBASTAS - SIMULACIÓN")
        print("=" * 70)
        
        # Generar datos
        self.generar_articulos()
        self.crear_compradores()
        
        # Crear subastador
        self.subastador = AgenteSubastador("Subastador", self.articulos, tiempo_espera_max=3.0)
        self.subastador.registrar_compradores(self.compradores)
        
        # Mostrar configuración
        print(f"\n📊 Configuración:")
        print(f"  - Compradores: {self.num_compradores}")
        for c in self.compradores:
            print(f"    * {c.nombre}: presupuesto ${c.presupuesto_inicial}, estrategia {c.estrategia}")
        print(f"  - Artículos: {self.num_articulos}")
        for a in self.articulos[:5]:  # mostrar solo los primeros 5
            print(f"    * {a.nombre}: base ${a.precio_base}")
        if self.num_articulos > 5:
            print(f"    * ... y {self.num_articulos - 5} más")
        
        # Iniciar hilos de compradores
        print(f"\n🚀 Iniciando agentes...")
        for comprador in self.compradores:
            comprador.start()
        
        # Iniciar subastador
        self.subastador.start()
        
        # Esperar a que termine el subastador
        self.subastador.join()
        
        # Detener compradores (por si acaso)
        for comprador in self.compradores:
            comprador.detener()
            comprador.join(timeout=1.0)
        
        print(f"\n✅ Simulación completada.\n")
        
        # Mostrar resultados
        self.mostrar_resultados()
        self.generar_graficas()
    
    def mostrar_resultados(self):
        """
        Muestra un resumen de los resultados de la subasta.
        """
        print("\n" + "=" * 70)
        print("RESULTADOS DE LA SUBASTA")
        print("=" * 70)
        
        # Artículos vendidos
        print("\n📦 Artículos subastados:")
        vendidos = 0
        total_recaudado = 0
        for articulo in self.articulos:
            if articulo.comprador_ganador:
                estado = f"VENDIDO a {articulo.comprador_ganador} por ${articulo.precio_actual}"
                vendidos += 1
                total_recaudado += articulo.precio_actual
            else:
                estado = "DESIERTO"
            print(f"  {articulo.nombre:30} | {estado}")
        
        print(f"\n📈 Resumen:")
        print(f"  - Artículos vendidos: {vendidos} de {self.num_articulos}")
        print(f"  - Recaudación total: ${total_recaudado:.2f}")
        
        # Rendimiento de compradores
        print("\n🛒 Rendimiento de compradores:")
        for comprador in self.compradores:
            print(f"\n  {comprador.nombre} (estrategia: {comprador.estrategia}):")
            print(f"    Presupuesto inicial: ${comprador.presupuesto_inicial:.2f}")
            print(f"    Presupuesto restante: ${comprador.presupuesto:.2f}")
            print(f"    Artículos comprados: {len(comprador.articulos_comprados)}")
            print(f"    Total gastado: ${comprador.total_gastado:.2f}")
            if comprador.articulos_comprados:
                print(f"    Artículos:")
                for art in comprador.articulos_comprados:
                    print(f"      - {art['articulo']} por ${art['precio']:.2f}")
    
    def generar_graficas(self):
        """
        Genera gráficas con los resultados.
        """
        # Gráfica 1: Artículos vendidos por precio
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        nombres = [a.nombre[:15] for a in self.articulos]  # truncar nombres largos
        precios_base = [a.precio_base for a in self.articulos]
        precios_final = [a.precio_actual if a.comprador_ganador else 0 for a in self.articulos]
        
        x = range(len(self.articulos))
        plt.bar(x, precios_base, width=0.4, label='Precio base', alpha=0.7)
        plt.bar([i+0.4 for i in x], precios_final, width=0.4, label='Precio final', alpha=0.7)
        plt.xlabel('Artículos')
        plt.ylabel('Precio ($)')
        plt.title('Comparación precios base vs final')
        plt.xticks([i+0.2 for i in x], nombres, rotation=45, ha='right')
        plt.legend()
        
        # Gráfica 2: Compras por comprador
        plt.subplot(1, 2, 2)
        nombres_comp = [c.nombre for c in self.compradores]
        gastos = [c.total_gastado for c in self.compradores]
        colores = ['green', 'blue', 'orange', 'red', 'purple'][:len(self.compradores)]
        plt.bar(nombres_comp, gastos, color=colores, alpha=0.7)
        plt.xlabel('Compradores')
        plt.ylabel('Total gastado ($)')
        plt.title('Gasto por comprador')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('resultados_subasta.png', dpi=150)
        # plt.show() # Comentado para evitar que bloquee la ejecución
        
        print("\n📊 Gráfica guardada como 'resultados_subasta.png'")

# =============================================================================
# PROGRAMA PRINCIPAL
# =============================================================================
if __name__ == "__main__":
    # Configurar semilla para reproducibilidad
    random.seed(42)
    
    # Crear y ejecutar sistema
    sistema = SistemaSubastas(num_compradores=5, num_articulos=10)
    sistema.ejecutar()