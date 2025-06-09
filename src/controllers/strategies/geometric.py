import time
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
    delta, omegas, dists_marginales, subsistema = args
    clave_cache = (tuple(delta), tuple(map(tuple, omegas)))
    if clave_cache in _cache_resultados:
        return _cache_resultados[clave_cache]

    temp = [lil_matrix((1, 128), dtype=np.int8), lil_matrix((1, 128), dtype=np.int8)]
    if isinstance(delta[0], tuple):
        for d in delta:
            t, idx = d
            temp[int(t)][0, idx] = 1
    else:
        t, idx = delta
        temp[int(t)][0, idx] = 1
    for omega in omegas:
        temp[int(omega[0])][0, omega[1]] = 1

    alcance = temp[EFECTO].nonzero()[1]
    mecanismo = temp[ACTUAL].nonzero()[1]

    part_delta = subsistema.bipartir(
        np.array(alcance, dtype=np.int8),
        np.array(mecanismo, dtype=np.int8),
    )
    dist_delta = part_delta.distribucion_marginal()
    emd_delta = emd_efecto(dist_delta, dists_marginales)

    part_union = subsistema.bipartir(
        np.array(alcance, dtype=np.int8),
        np.array(mecanismo, dtype=np.int8),
    )
    dist_union = part_union.distribucion_marginal()
    emd_union = emd_efecto(dist_union, dists_marginales)

    resultado = (delta, emd_union, emd_delta, dist_delta)
    _cache_resultados[clave_cache] = resultado
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
        self.sia_preparar_subsistema(condicion, alcance, mecanismo)
        futuro = [(EFECTO, i) for i in self.sia_subsistema.indices_ncubos]
        presente = [(ACTUAL, i) for i in self.sia_subsistema.dims_ncubos]
        vertices = list(presente + futuro)
        self.vertices = set(vertices)

        inicio = vertices[0]
        self.memoria_particiones.clear()
        mejor_particion = self._algorithm([inicio] + [v for v in vertices if v != inicio])
        perdida, dist_marginal = self.memoria_particiones[mejor_particion]
        complementaria = self._nodes_complement(mejor_particion)
        tiempo_total = time.time() - self.sia_tiempo_inicio
        fmt_mip = fmt_biparte_q(list(mejor_particion), complementaria)

        return Solution(
            estrategia="GEOMETRIC",
            perdida=perdida,
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=dist_marginal,
            tiempo_total=tiempo_total,
            particion=fmt_mip,
        )

    def _algorithm(self, vertices: list[tuple[int, int]]):
        omegas = [vertices[0]]
        deltas = vertices[1:]
        vertices_fase = vertices
        total_fases = len(vertices_fase) - 2
        start_time = time.time()
        usar_paralelo = len(vertices) > UMBRAL_SIMPLE_EVALUACION

        for fase_idx in range(total_fases):
            omegas_ciclo = [vertices_fase[0]]
            deltas_ciclo = vertices_fase[1:]
            mejor_dist = None
            emd_particion = INFTY_POS
            total_iteraciones = len(deltas_ciclo) - 1
            progreso = tqdm(range(total_iteraciones), desc=f"Fase {fase_idx+1}/{total_fases}", ncols=100)

            for iter_idx in progreso:
                args = [
                    (delta, tuple(map(tuple, omegas_ciclo)), self.sia_dists_marginales, self.sia_subsistema)
                    for delta in deltas_ciclo
                ]
                iter_start = time.time()

                if usar_paralelo:
                    max_procs = min(len(args), MAX_PROC_LIMIT, int(MAX_RAM_MB * 1024 * 1024 / BYTES_PER_EVAL))
                    with Pool(processes=max_procs) as pool:
                        resultados = pool.map(_eval_delta_union, args)
                else:
                    resultados = list(map(_eval_delta_union, args))

                mejor = min(resultados, key=lambda x: x[1] - x[2])
                delta, emd_union, emd_delta, dist_delta = mejor
                omegas_ciclo.append(delta)
                deltas_ciclo.remove(delta)

                if emd_delta < emd_particion:
                    emd_particion = emd_delta
                    mejor_dist = dist_delta

                elapsed = time.time() - start_time
                total_iters = fase_idx * total_iteraciones + iter_idx + 1
                iter_time = time.time() - iter_start

                if usar_paralelo and iter_time > TIEMPO_MAXIMO_ITER:
                    usar_paralelo = False

                speed = total_iters / elapsed
                eta = (total_fases * total_iteraciones - total_iters) / speed if speed > 0 else float('inf')
                progreso.set_postfix({
                    "iter/s": f"{speed:.2f}",
                    "eta": f"{eta:.1f}s",
                    "done": f"{total_iters}/{total_fases * total_iteraciones}"
                })

            clave = tuple(
                deltas_ciclo[LAST_IDX]
                if isinstance(deltas_ciclo[LAST_IDX], list)
                else deltas_ciclo
            )
            self.memoria_particiones[clave] = (emd_particion, mejor_dist)

            if emd_particion <= 1e-6:
                self.logger.info("Partición con pérdida mínima encontrada.")
                return min(self.memoria_particiones, key=lambda k: self.memoria_particiones[k][0])

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

        return min(self.memoria_particiones, key=lambda k: self.memoria_particiones[k][0])

    def _nodes_complement(self, nodes: list[tuple[int, int]]):
        return list(self.vertices - set(nodes))
