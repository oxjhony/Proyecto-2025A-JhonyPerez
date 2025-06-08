import time
import numpy as np
from typing import Union
from src.models.base.sia import SIA
from src.controllers.manager import Manager
from src.constants.base import EFECTO, ACTUAL, LAST_IDX, INFTY_POS
from src.funcs.base import emd_efecto
from src.models.core.solution import Solution
from src.middlewares.slogger import SafeLogger
from src.funcs.format import fmt_biparte_q


class Geometric(SIA):
    def __init__(self, gestor: Manager):
        super().__init__(gestor)
        self.logger = SafeLogger("GEOMETRIC_SIA")
        self.memoria_omega = dict()
        self.memoria_particiones = dict()
        self.vertices: set[tuple]

    def aplicar_estrategia(self, condicion: str, alcance: str, mecanismo: str):
        # 1. Descomponer en tensores elementales
        self.sia_preparar_subsistema(condicion, alcance, mecanismo)

        # 2. Calcular tabla de costos (con función t(i, j))
        futuro = [(EFECTO, i) for i in self.sia_subsistema.indices_ncubos]
        presente = [(ACTUAL, i) for i in self.sia_subsistema.dims_ncubos]
        vertices = list(presente + futuro)
        self.vertices = set(vertices)

        # 3. Identificar biparticiones candidatas
        # 4. Evaluar biparticiones usando discrepancia tensorial
        mejor_particion = self._algorithm(vertices)

        # 5. Retornar la mejor bipartición
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

        for _ in range(len(vertices_fase) - 2):
            omegas_ciclo = [vertices_fase[0]]
            deltas_ciclo = vertices_fase[1:]

            mejor_delta = None
            mejor_dist = None
            emd_particion = INFTY_POS

            for _ in range(len(deltas_ciclo) - 1):
                emd_local = INFTY_POS
                indice_mejor = -1

                for k, delta in enumerate(deltas_ciclo):
                    emd_union, emd_delta, dist_delta = self._funcion_submodular(delta, omegas_ciclo)
                    emd_iter = emd_union - emd_delta

                    if emd_iter < emd_local:
                        emd_local = emd_iter
                        indice_mejor = k
                        emd_particion = emd_delta
                        mejor_dist = dist_delta

                omegas_ciclo.append(deltas_ciclo[indice_mejor])
                deltas_ciclo.pop(indice_mejor)

            clave = tuple(
                deltas_ciclo[LAST_IDX]
                if isinstance(deltas_ciclo[LAST_IDX], list)
                else deltas_ciclo
            )
            self.memoria_particiones[clave] = (emd_particion, mejor_dist)

            if emd_particion <= 0.01:
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

    def _funcion_submodular(
        self, deltas: Union[tuple, list[tuple]], omegas: list[Union[tuple, list[tuple]]]
    ):
        clave = (tuple(deltas), tuple(map(tuple, omegas)))
        if clave in self.memoria_omega:
            return self.memoria_omega[clave]

        temp = [[], []]
        if isinstance(deltas, tuple):
            temp[deltas[0]].append(deltas[1])
        else:
            for d in deltas:
                temp[d[0]].append(d[1])

        part_delta = self.sia_subsistema.bipartir(
            np.array(temp[EFECTO], dtype=np.int8),
            np.array(temp[ACTUAL], dtype=np.int8),
        )
        dist_delta = part_delta.distribucion_marginal()
        emd_delta = emd_efecto(dist_delta, self.sia_dists_marginales)

        for omega in omegas:
            if isinstance(omega, list):
                for o in omega:
                    temp[o[0]].append(o[1])
            else:
                temp[omega[0]].append(omega[1])

        part_union = self.sia_subsistema.bipartir(
            np.array(temp[EFECTO], dtype=np.int8),
            np.array(temp[ACTUAL], dtype=np.int8),
        )
        dist_union = part_union.distribucion_marginal()
        emd_union = emd_efecto(dist_union, self.sia_dists_marginales)

        self.memoria_omega[clave] = (emd_union, emd_delta, dist_delta)
        return emd_union, emd_delta, dist_delta

    def _nodes_complement(self, nodes: list[tuple[int, int]]):
        return list(self.vertices - set(nodes))
