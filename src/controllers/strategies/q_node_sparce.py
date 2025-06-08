from scipy.sparse import dok_matrix, lil_matrix, csr_matrix
import time
import numpy as np
from typing import Union
from src.middlewares.slogger import SafeLogger
from src.funcs.base import emd_efecto, ABECEDARY
from src.middlewares.profile import profiler_manager, profile
from src.funcs.format import fmt_biparte_q
from src.controllers.manager import Manager
from src.models.base.sia import SIA
from src.models.core.solution import Solution
from src.constants.base import (
    EFECTO,
    ACTUAL,
    INFTY_NEG,
    INFTY_POS,
    LAST_IDX,
    TYPE_TAG,
)


class QNodesFullSparse(SIA):
    def __init__(self, config: Manager):
        super().__init__(config)
        profiler_manager.start_session(f"NET{len(config.estado_inicial)}{config.pagina}")
        self.m: int
        self.n: int
        self.tiempos: tuple[np.ndarray, np.ndarray]
        self.etiquetas = [tuple(s.lower() for s in ABECEDARY), ABECEDARY]
        self.vertices: set[tuple]
        self.memoria_delta = dict()
        self.memoria_omega = dict()
        self.memoria_particiones = dict()
        self.indices_alcance: np.ndarray
        self.indices_mecanismo: np.ndarray
        self.adjacency_matrix = dok_matrix((1000, 1000), dtype=np.float32)
        self.logger = SafeLogger("q_strategy")


    def agregar_conexion(self, i, j, peso=1.0):
        self.adjacency_matrix[i, j] = peso


    def convertir_a_csr(self):
        return self.adjacency_matrix.tocsr()


    @profile(context={TYPE_TAG: "q_analysis"})
    def aplicar_estrategia(self, conditions, alcance, mecanismo):
        self.sia_preparar_subsistema(conditions, alcance, mecanismo)
        futuro = tuple((EFECTO, efecto) for efecto in self.sia_subsistema.indices_ncubos)
        presente = tuple((ACTUAL, actual) for actual in self.sia_subsistema.dims_ncubos)
        self.m = self.sia_subsistema.indices_ncubos.size
        self.n = self.sia_subsistema.dims_ncubos.size
        self.indices_alcance = self.sia_subsistema.indices_ncubos
        self.indices_mecanismo = self.sia_subsistema.dims_ncubos
        self.tiempos = (np.zeros(self.n, dtype=np.int8), np.zeros(self.m, dtype=np.int8))
        vertices = list(presente + futuro)
        self.vertices = set(presente + futuro)
        mip = self.algorithm(vertices)
        fmt_mip = fmt_biparte_q(list(mip), self.nodes_complement(mip))
        return Solution(
            estrategia="Q-Nodes",
            perdida=self.memoria_particiones[mip][0],
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=self.memoria_particiones[mip][1],
            tiempo_total=time.time() - self.sia_tiempo_inicio,
            particion=fmt_mip,
        )


    def funcion_submodular(self, deltas: Union[tuple, list[tuple]], omegas: list[Union[tuple, list[tuple]]]):
        emd_delta = INFTY_NEG
        vector_bin_presente = lil_matrix((1, self.n), dtype=np.int8)
        vector_bin_futuro = lil_matrix((1, self.m), dtype=np.int8)
        if isinstance(deltas, tuple):
            d_tiempo, o_indice = deltas
            if d_tiempo == ACTUAL and o_indice < self.n:
                vector_bin_presente[0, o_indice] = 1
            elif d_tiempo == EFECTO and o_indice < self.m:
                vector_bin_futuro[0, o_indice] = 1
        else:
            for delta in deltas:
                d_tiempo, o_indice = delta
                if d_tiempo == ACTUAL and o_indice < self.n:
                    vector_bin_presente[0, o_indice] = 1
                elif d_tiempo == EFECTO and o_indice < self.m:
                    vector_bin_futuro[0, o_indice] = 1
        vector_bin_presente = vector_bin_presente.tocsr()
        vector_bin_futuro = vector_bin_futuro.tocsr()
        if tuple(deltas) in self.memoria_delta:
            emd_delta, vector_delta_marginal = self.memoria_delta[tuple(deltas)]
        else:
            copia_delta = self.sia_subsistema
            dims_alcance_delta = vector_bin_futuro.nonzero()[1]
            dims_mecanismo_delta = vector_bin_presente.nonzero()[1]
            particion_delta = copia_delta.bipartir(
                np.array(dims_alcance_delta, dtype=np.int8),
                np.array(dims_mecanismo_delta, dtype=np.int8),
            )
            vector_delta_marginal = particion_delta.distribucion_marginal()
            emd_delta = emd_efecto(vector_delta_marginal, self.sia_dists_marginales)
            self.memoria_delta[tuple(deltas)] = emd_delta, vector_delta_marginal
        for omega in omegas:
            if isinstance(omega, list):
                for omg in omega:
                    o_tiempo, o_indice = omg
                    if o_tiempo == ACTUAL and o_indice < self.n:
                        vector_bin_presente[0, o_indice] = 1
                    elif o_tiempo == EFECTO and o_indice < self.m:
                        vector_bin_futuro[0, o_indice] = 1
            else:
                o_tiempo, o_indice = omega
                if o_tiempo == ACTUAL and o_indice < self.n:
                    vector_bin_presente[0, o_indice] = 1
                elif o_tiempo == EFECTO and o_indice < self.m:
                    vector_bin_futuro[0, o_indice] = 1
        copia_union = self.sia_subsistema
        dims_alcance_union = vector_bin_futuro.nonzero()[1]
        dims_mecanismo_union = vector_bin_presente.nonzero()[1]
        particion_union = copia_union.bipartir(
            np.array(dims_alcance_union, dtype=np.int8),
            np.array(dims_mecanismo_union, dtype=np.int8),
        )
        vector_union_marginal = particion_union.distribucion_marginal()
        emd_union = emd_efecto(vector_union_marginal, self.sia_dists_marginales)
        return emd_union, emd_delta, vector_delta_marginal


    def algorithm(self, vertices: list[tuple[int, int]]):
        omegas_origen = [vertices[0]]
        deltas_origen = vertices[1:]
        vertices_fase = vertices
        for i in range(len(vertices_fase) - 2):
            self.logger.critic(f"Fase {i+1} de {len(vertices_fase) - 2}")
            omegas_ciclo = [vertices_fase[0]]
            deltas_ciclo = vertices_fase[1:]
            emd_particion_candidata = INFTY_POS
            for j in range(len(deltas_ciclo) - 1):
                emd_local = INFTY_POS
                indice_mip = -1
                for k in range(len(deltas_ciclo)):
                    emd_union, emd_delta, dist_marginal_delta = self.funcion_submodular(
                        deltas_ciclo[k], omegas_ciclo
                    )
                    emd_iteracion = emd_union - emd_delta
                    if emd_iteracion < emd_local:
                        emd_local = emd_iteracion
                        indice_mip = k
                        emd_particion_candidata = emd_delta
                        dist_particion_candidata = dist_marginal_delta
                omegas_ciclo.append(deltas_ciclo[indice_mip])
                deltas_ciclo.pop(indice_mip)
            clave = tuple(
                deltas_ciclo[LAST_IDX]
                if isinstance(deltas_ciclo[LAST_IDX], list)
                else deltas_ciclo
            )
            self.memoria_particiones[clave] = emd_particion_candidata, dist_particion_candidata


            print(f"Partición candidata: {deltas_ciclo}")
            print(f"EMD candidata: {emd_particion_candidata}")


            TOLERANCIA_PERDIDA = 0.01  # Umbral para considerar pérdidas como cero


            # Verificar si la pérdida es menor o igual al umbral de tolerancia
            if emd_particion_candidata <= TOLERANCIA_PERDIDA:
                emd_particion_candidata = 0
                self.logger.info(
                    f"Partición con pérdida <= {TOLERANCIA_PERDIDA} encontrada. Terminando el proceso."
                )
                # Retornar la partición con la menor pérdida si no se encontró una con pérdida menor o igual al umbral
                mejor_particion = min(
                    self.memoria_particiones, key=lambda k: self.memoria_particiones[k][0]
                )
                perdida_minima, dist_marginal = self.memoria_particiones[mejor_particion]


                # Ajustar la pérdida mínima a cero si es menor o igual al umbral
                if perdida_minima <= TOLERANCIA_PERDIDA:
                    perdida_minima = 0
                    self.memoria_particiones[mejor_particion] = (perdida_minima, dist_marginal)


                return mejor_particion
            #if emd_particion_candidata == 0:
                #return clave
            # En algorithm() — reemplaza esto:


            par_candidato = (
                [omegas_ciclo[LAST_IDX]] if isinstance(omegas_ciclo[LAST_IDX], tuple) else omegas_ciclo[LAST_IDX]
            ) + (
                [deltas_ciclo[LAST_IDX]] if isinstance(deltas_ciclo[LAST_IDX], tuple) else deltas_ciclo[LAST_IDX]
            )


            omegas_ciclo.pop()
            omegas_ciclo.append(par_candidato)
            vertices_fase = omegas_ciclo
        return min(self.memoria_particiones, key=lambda k: self.memoria_particiones[k][0])


    def nodes_complement(self, nodes: list[tuple[int, int]]):
        return list(set(self.vertices) - set(nodes))



