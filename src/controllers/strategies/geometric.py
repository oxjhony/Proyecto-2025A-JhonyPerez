# src/controllers/strategies/geometric.py

import time
import numpy as np
import multiprocessing as mp
from datetime import datetime
from numba import njit
from tqdm import tqdm
import csv

from src.controllers.manager import Manager
from src.models.base.sia import SIA
from src.models.core.solution import Solution
from src.middlewares.profile import profiler_manager, profile
from src.middlewares.slogger import SafeLogger
from src.middlewares.observer import DebugObserver
from src.funcs.system import biparticiones
from src.funcs.format import fmt_biparticion
from src.constants.models import DUMMY_ARR

@njit(cache=True)
def l1_distance(a: np.ndarray, b: np.ndarray) -> float:
    result = 0.0
    for i in range(a.size):
        result += abs(a[i] - b[i])
    return result

@njit(cache=True)
def calcular_tabla_costos(tensor: np.ndarray) -> np.ndarray:
    n = tensor.shape[0]
    tabla = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = 0
            tmp = i ^ j
            while tmp:
                d += tmp & 1
                tmp >>= 1
            gamma = 2 ** (-d)
            tabla[i, j] = gamma * abs(tensor[i] - tensor[j])
    return tabla

def evaluar_biparticion(args):
    subsistema, dist_subsistema, futuros, presentes = args
    bip = subsistema.bipartir(
        np.array(futuros, dtype=np.int8),
        np.array(presentes, dtype=np.int8)
    )

    dist_particion = bip.distribucion_marginal()
    phi = l1_distance(dist_subsistema, dist_particion)
    return (phi, dist_particion, (tuple(futuros), tuple(presentes)))

@profile(context={"strategy": "geometric"})
class Geometric(SIA):
    def __init__(self, config: Manager) -> None:
        super().__init__(config)
        profiler_manager.start_session(f"NET{len(config.estado_inicial)}{config.pagina}")
        self.logger = SafeLogger("geometric")
        self.debug_observer = DebugObserver()

    def aplicar_estrategia(self, condiciones: str, alcance: str, mecanismo: str) -> Solution:
        self.sia_tiempo_inicio = time.time()

        print("[1] Preparando subsistema y distribuciones...")
        self.sia_preparar_subsistema(condiciones, alcance, mecanismo)
        subsistema = self.sia_subsistema
        dist_subsistema = subsistema.distribucion_marginal()

        print("[2] Calculando tabla de costos aproximada...")
        if hasattr(subsistema, 'tensor_principal'):
            tensor = subsistema.tensor_principal
        else:
            tensor = dist_subsistema

        tabla_costos = calcular_tabla_costos(tensor)
        self.debug_observer.on_tensor_product({
    "n_cubes": 1,
    "active_dims": list(range(tensor.shape[0])),
    "cubes": [type('DummyCube', (), {"indices": list(range(tensor.shape[0])), "dims": list(range(tensor.shape[0])), "data": tabla_costos})()]
})

        print("[3] Generando biparticiones candidatas geométricas...")
        futuros = subsistema.indices_ncubos
        presentes = subsistema.dims_ncubos

        candidatas_list = []
        for i, b in enumerate(biparticiones(futuros, presentes)):
            if 2 <= len(b[0]) <= 4 and 2 <= len(b[1]) <= 4 and i % 2 == 0:
                candidatas_list.append(b)
            if len(candidatas_list) >= 5000000:
                break

        print(f"    ▸ Total de biparticiones evaluadas: {len(candidatas_list)}")

        print("[4] Evaluando biparticiones por discrepancia tensorial...")
        tareas = [(subsistema, dist_subsistema, f, p) for f, p in candidatas_list]

        mejor_phi = np.inf
        mejor_dist = DUMMY_ARR
        mejor_bipart = None

        with mp.Pool(processes=mp.cpu_count()) as pool:
            for resultado in tqdm(pool.imap_unordered(evaluar_biparticion, tareas), total=len(tareas), desc="    ▸ Evaluando"):
                phi, dist_part, bipart = resultado
                if phi < mejor_phi:
                    mejor_phi = phi
                    mejor_dist = dist_part
                    mejor_bipart = bipart
                    if mejor_phi == 0.0:
                        print("    ▸ φ = 0 encontrado. Finalizando evaluación anticipadamente.")
                        pool.terminate()
                        break

        fsel, psel = mejor_bipart
        dual_p = set(subsistema.dims_ncubos.tolist()) - set(psel)
        dual_f = set(subsistema.indices_ncubos.tolist()) - set(fsel)
        bipart_str = fmt_biparticion((fsel, psel), (tuple(dual_f), tuple(dual_p)))
        tiempo_total = time.time() - self.sia_tiempo_inicio

        # Descomentar esta sección para guardar resultados en prueba final
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # with open(f"resultados_geometric_{timestamp}.csv", "w", newline="", encoding="utf-8-sig") as f:
        #     writer = csv.writer(f)
        #     writer.writerow(["estrategia", "phi", "tiempo", "particion"])
        #     writer.writerow(["Geometric", mejor_phi, f"{tiempo_total:.2f}", bipart_str])

        return Solution(
            estrategia="Geometric",
            perdida=mejor_phi,
            distribucion_subsistema=dist_subsistema,
            distribucion_particion=mejor_dist,
            particion=bipart_str,
            tiempo_total=tiempo_total,
            hablar=True
        )
