import time
"""
Geometric SIA Strategy
Este módulo implementa la estrategia Geometric para la identificación de biparticiones óptimas en sistemas de información, utilizando la métrica Earth Mover’s Distance (EMD) para comparar distribuciones marginales.
Funciones:
- _eval_delta_union(args): 
    Evalúa una bipartición candidata delta y su unión con omega, midiendo qué tan diferente es su distribución respecto al sistema original usando la métrica Earth Mover’s Distance (EMD).
    Esto es crucial para calcular la pérdida φ = emd_union - emd_delta, la base del criterio de selección de la mejor bipartición.
Clases:
- Geometric:
    Estrategia basada en SIA para encontrar la mejor bipartición de un subsistema dado, minimizando la pérdida de información medida por EMD. Utiliza paralelización para acelerar la evaluación de particiones y mantiene un registro de las mejores particiones encontradas durante la búsqueda.
    Métodos principales:
    - __init__(gestor: Manager): Inicializa la estrategia con el gestor y parámetros de configuración.
    - aplicar_estrategia(condicion: str, alcance: str, mecanismo: str): Ejecuta la estrategia Geometric sobre el subsistema definido por los parámetros dados.
    - _algorithm(vertices: list[tuple[int, int]]): Algoritmo principal de búsqueda de biparticiones óptimas.
    - _nodes_complement(nodes: list[tuple[int, int]]): Calcula el complemento de un conjunto de nodos respecto al conjunto total de vértices.
"""
import numpy as np
from typing import Union
from scipy.sparse import lil_matrix
from src.models.base.sia import SIA
from src.controllers.manager import Manager
from src.constants.base import EFECTO, ACTUAL, LAST_IDX, INFTY_POS
from src.funcs.base import emd_efecto
from src.models.core.solution import Solution
from src.middlewares.slogger import SafeLogger
from src.funcs.format import fmt_biparte_q
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import random

MAX_PROC_LIMIT = max(1, cpu_count() - 1)
MAX_RAM_MB = 3000
BYTES_PER_EVAL = 5 * 1024 * 1024
UMBRAL_SIMPLE_EVALUACION = 6
TIEMPO_MAXIMO_ITER = 0.005
_cache_resultados = {}



def _eval_delta_union(args):
    # Desempaqueta los argumentos: conjunto delta, conjunto omega, distribución marginal del subsistema y el subsistema en sí
    delta, omegas, dists_marginales, subsistema = args

    # Crea una clave única basada en delta y omega para almacenar/reutilizar resultados cacheados
    clave_cache = (tuple(delta), tuple(map(tuple, omegas)))
    if clave_cache in _cache_resultados:
        # Si ya se ha evaluado esta combinación antes, devuelve el resultado almacenado para evitar recalculo
        return _cache_resultados[clave_cache]

    # Inicializa dos vectores dispersos para representar alcance (EFECTO) y mecanismo (ACTUAL)
    temp = [lil_matrix((1, 128), dtype=np.int8), lil_matrix((1, 128), dtype=np.int8)]

    # Marca en el vector temp las posiciones activas del conjunto delta
    if isinstance(delta[0], tuple):  # Si delta es una lista de tuplas
        for d in delta:
            t, idx = d
            temp[int(t)][0, idx] = 1
    else:  # Si delta es una sola tupla
        t, idx = delta
        temp[int(t)][0, idx] = 1

    # Agrega al vector temp las posiciones activas de omega
    for omega in omegas:
        temp[int(omega[0])][0, omega[1]] = 1

    # Extrae los índices activados para alcance y mecanismo usando nonzero (posición de los 1s en las matrices dispersas)
    alcance = temp[EFECTO].nonzero()[1]
    mecanismo = temp[ACTUAL].nonzero()[1]

    # Crea la partición `delta` (solo los bits de delta) y calcula su distribución marginal
    part_delta = subsistema.bipartir(
        np.array(alcance, dtype=np.int8),
        np.array(mecanismo, dtype=np.int8),
    )
    dist_delta = part_delta.distribucion_marginal()

    # Calcula la EMD entre la distribución de la partición `delta` y la distribución del subsistema original
    emd_delta = emd_efecto(dist_delta, dists_marginales)

    # Crea la partición `union` (omega + delta) y calcula su distribución marginal
    part_union = subsistema.bipartir(
        np.array(alcance, dtype=np.int8),
        np.array(mecanismo, dtype=np.int8),
    )
    dist_union = part_union.distribucion_marginal()

    # Calcula la EMD entre la partición unión y el subsistema original
    emd_union = emd_efecto(dist_union, dists_marginales)

    # Empaqueta los resultados en una tupla: delta, emd de la unión, emd de delta y su distribución marginal
    resultado = (delta, emd_union, emd_delta, dist_delta)

    # Guarda los resultados en caché para evitar cálculos repetidos futuros
    _cache_resultados[clave_cache] = resultado

    # Retorna el resultado
    return resultado



class Geometric(SIA):
    def __init__(self, gestor: Manager):
        super().__init__(gestor)
        self.logger = SafeLogger("GEOMETRIC_SIA")
        self.memoria_particiones = dict()
        self.vertices: set[tuple]
        self.max_vertices_sampling = 20
        self.sample_size = 2000

    def aplicar_estrategia(self, condicion: str, alcance: str, mecanismo: str):
        # Paso 1: Descomponer en tensores elementales
        # Prepara el subsistema separando el sistema total en el subconjunto relevante de variables y distribuciones
        self.sia_preparar_subsistema(condicion, alcance, mecanismo)

        # Define los vértices: futuro (efecto) y presente (actual)
        futuro = [(EFECTO, i) for i in self.sia_subsistema.indices_ncubos]
        presente = [(ACTUAL, i) for i in self.sia_subsistema.dims_ncubos]
        vertices = list(presente + futuro)
        self.vertices = set(vertices)

        # Reinicio del algoritmo desde el primer vértice (puede ser aleatorio o fijo)
        inicio = vertices[0]
        
        # Limpia resultados anteriores para asegurar análisis limpio
        self.memoria_particiones.clear()

        # Paso 2 y 3: Calcular tabla de costos y generar biparticiones candidatas
        # Ejecuta el algoritmo con este orden inicial para explorar combinaciones
        mejor_particion = self._algorithm([inicio] + [v for v in vertices if v != inicio])

        # Paso 4: Evaluar biparticiones usando discrepancia tensorial
        # Recupera la pérdida mínima y la distribución marginal asociada a la mejor bipartición
        perdida, dist_marginal = self.memoria_particiones[mejor_particion]

        # Calcula los nodos complementarios a la partición ganadora
        complementaria = self._nodes_complement(mejor_particion)

        # Mide el tiempo total de ejecución
        tiempo_total = time.time() - self.sia_tiempo_inicio

        # Paso 5: Retornar la mejor bipartición formateada
        fmt_mip = fmt_biparte_q(list(mejor_particion), complementaria)

        # Empaqueta todos los resultados en una solución estándar
        return Solution(
            estrategia="GEOMETRIC",
            perdida=perdida,
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=dist_marginal,
            tiempo_total=tiempo_total,
            particion=fmt_mip,
        )


        """
        Ejecuta un algoritmo geométrico iterativo para particionar un conjunto de vértices minimizando una métrica de distancia (EMD).
        Args:
            vertices (list[tuple[int, int]]): Lista de vértices representados como tuplas de coordenadas (x, y).
        Proceso:
            - Inicializa las estructuras de trabajo con el primer vértice como punto de partida.
            - Itera por fases, donde en cada fase se selecciona el vértice que minimiza la diferencia de distancias EMD.
            - Utiliza procesamiento paralelo si el número de vértices supera un umbral definido.
            - Almacena en memoria las particiones y sus distancias mínimas encontradas.
            - Muestra el progreso de la ejecución y estima el tiempo restante.
            - Finaliza anticipadamente si se encuentra una partición con pérdida mínima (menor a 1e-6).
        Returns:
            tuple: Clave de la partición con la menor distancia EMD encontrada.
        Notas:
            - Utiliza variables y constantes externas como `UMBRAL_SIMPLE_EVALUACION`, `MAX_PROC_LIMIT`, `MAX_RAM_MB`, `BYTES_PER_EVAL`, `INFTY_POS`, `TIEMPO_MAXIMO_ITER`, `LAST_IDX`, y funciones auxiliares como `_eval_delta_union`.
            - Requiere que los atributos `self.sia_dists_marginales`, `self.sia_subsistema`, `self.memoria_particiones` y `self.logger` estén definidos en la clase.
        """
    def _algorithm(self, vertices: list[tuple[int, int]]):
        # Inicializa el conjunto omega con el primer vértice (punto de partida)
        omegas = [vertices[0]]
        # El resto se consideran candidatos delta
        deltas = vertices[1:]
        vertices_fase = vertices

        # Número de fases = número de vértices - 2 (uno para omega inicial, uno mínimo para evaluar)
        total_fases = len(vertices_fase) - 2

        # Marca el tiempo inicial para métricas de rendimiento
        start_time = time.time()

        # Decide si usar paralelización según la cantidad de vértices
        usar_paralelo = len(vertices) > UMBRAL_SIMPLE_EVALUACION

        # Iteración principal por fase (una expansión de omega en cada fase)
        for fase_idx in range(total_fases):
            # Reinicia omega y deltas para esta fase
            omegas_ciclo = [vertices_fase[0]]
            deltas_ciclo = vertices_fase[1:]

            # Variables para registrar la mejor pérdida y distribución marginal encontrada
            mejor_dist = None
            emd_particion = INFTY_POS

            # Número de iteraciones internas en esta fase
            total_iteraciones = len(deltas_ciclo) - 1

            # Barra de progreso para monitorear esta fase
            #progreso = tqdm(range(total_iteraciones), desc=f"Fase {fase_idx+1}/{total_fases}", ncols=100)
            progreso = tqdm(range(total_iteraciones))

            # Iteración interna: selección de mejor delta para expandir omega
            for iter_idx in progreso:
                # Prepara los argumentos para evaluar cada delta con el omega actual
                args = [
                    (delta, tuple(map(tuple, omegas_ciclo)), self.sia_dists_marginales, self.sia_subsistema)
                    for delta in deltas_ciclo
                ]

                iter_start = time.time()

                # Evalúa las combinaciones en paralelo o secuencialmente
                if usar_paralelo:
                    max_procs = min(len(args), MAX_PROC_LIMIT, int(MAX_RAM_MB * 1024 * 1024 / BYTES_PER_EVAL))
                    with Pool(processes=max_procs) as pool:
                        resultados = pool.map(_eval_delta_union, args)
                else:
                    resultados = list(map(_eval_delta_union, args))

                # Selecciona la mejor combinación según menor pérdida φ = emd_union - emd_delta
                mejor = min(resultados, key=lambda x: x[1] - x[2])
                delta, emd_union, emd_delta, dist_delta = mejor

                # Agrega el mejor delta a omega y lo elimina de los candidatos
                omegas_ciclo.append(delta)
                deltas_ciclo.remove(delta)

                # Actualiza si se encontró una mejor pérdida
                if emd_delta < emd_particion:
                    emd_particion = emd_delta
                    mejor_dist = dist_delta

                # Métricas de rendimiento en tiempo real
                elapsed = time.time() - start_time
                total_iters = fase_idx * total_iteraciones + iter_idx + 1
                iter_time = time.time() - iter_start

                # Desactiva paralelismo si una iteración es demasiado lenta
                if usar_paralelo and iter_time > TIEMPO_MAXIMO_ITER:
                    usar_paralelo = False

                # Actualiza barra de progreso con velocidad e ETA
                speed = total_iters / elapsed
                eta = (total_fases * total_iteraciones - total_iters) / speed if speed > 0 else float('inf')
                """ progreso.set_postfix({
                    "iter/s": f"{speed:.2f}",
                    "eta": f"{eta:.1f}s",
                    "done": f"{total_iters}/{total_fases * total_iteraciones}"
                }) """

            # Guarda el resultado final de esta fase en memoria
            clave = tuple(
                deltas_ciclo[LAST_IDX]
                if isinstance(deltas_ciclo[LAST_IDX], list)
                else deltas_ciclo
            )
            self.memoria_particiones[clave] = (emd_particion, mejor_dist)

            # Si la pérdida es casi cero, termina anticipadamente
            if emd_particion <= 1e-6:
                self.logger.info("Partición con pérdida mínima encontrada.")
                return min(self.memoria_particiones, key=lambda k: self.memoria_particiones[k][0])

            # Prepara la siguiente fase combinando último omega y último delta
            nuevo = (
                [omegas_ciclo[LAST_IDX]] if isinstance(omegas_ciclo[LAST_IDX], tuple)
                else omegas_ciclo[LAST_IDX]
            ) + (
                [deltas_ciclo[LAST_IDX]] if isinstance(deltas_ciclo[LAST_IDX], tuple)
                else deltas_ciclo[LAST_IDX]
            )
            omegas_ciclo.pop()
            omegas_ciclo.append(nuevo)
            vertices_fase = omegas_ciclo

        # Devuelve la mejor partición encontrada al final de todas las fases
        return min(self.memoria_particiones, key=lambda k: self.memoria_particiones[k][0])


    def _nodes_complement(self, nodes: list[tuple[int, int]]):
        """
        Devuelve una lista de los vértices que no están presentes en la lista de nodos proporcionada.
        Args:
            nodes (list[tuple[int, int]]): Lista de nodos representados como tuplas de enteros.
        Returns:
            list: Lista de vértices que no están en la lista de nodos dada.
        """

        return list(self.vertices - set(nodes))
