# src/controllers/strategies/geometric_light.py

import time
import numpy as np
import sys
import multiprocessing as mp
from numba import njit
from tqdm import tqdm
import csv
from datetime import datetime

from src.controllers.manager import Manager
from src.middlewares.slogger import SafeLogger
from src.middlewares.observer import DebugObserver
from src.middlewares.profile import profiler_manager, profile
from src.models.base.sia import SIA
from src.models.core.solution import Solution
from src.funcs.system import biparticiones
from src.funcs.format import fmt_biparticion
from src.constants.models import DUMMY_ARR

@njit(cache=True)
def l1_distance(a: np.ndarray, b: np.ndarray) -> float:
    result = 0.0
    for i in range(a.size):
        result += abs(a[i] - b[i])
    return result

def evaluar_biparticion(args):
    subsistema, base_dist, f_sel, p_sel = args
    bip = subsistema.bipartir(
        np.array(f_sel, dtype=np.int8),
        np.array(p_sel, dtype=np.int8)
    )
    dist_part = bip.distribucion_marginal()
    phi = np.sum(np.abs(base_dist - dist_part))
    return (phi, dist_part, (tuple(f_sel), tuple(p_sel)))

@profile(context={"strategy": "geometric_light"})
class Geometric(SIA):
    def __init__(self, config: Manager) -> None:
        super().__init__(config)
        profiler_manager.start_session(f"NET{len(config.estado_inicial)}{config.pagina}")
        self.logger = SafeLogger("geometric_light")
        self.debug_observer = DebugObserver()

    def aplicar_estrategia(self, condiciones: str, alcance: str, mecanismo: str) -> Solution:
        self.sia_tiempo_inicio = time.time()

        print("[1] Preparando subsistema...")
        self.sia_preparar_subsistema(condiciones, alcance, mecanismo)
        subsistema = self.sia_subsistema

        print("[2] Generando biparticiones heurísticas optimizadas...")
        futuros = subsistema.indices_ncubos
        presentes = subsistema.dims_ncubos

        MAX_PARTICIONES = 10000000
        candidatas_list = []
        for i, b in enumerate(biparticiones(futuros, presentes)):
            if 2 <= len(b[0]) <= 4 and 2 <= len(b[1]) <= 4 and i % 2 == 0:
                candidatas_list.append(b)
            if len(candidatas_list) >= MAX_PARTICIONES:
                break

        print(f"    ▸ Total de biparticiones evaluadas: {len(candidatas_list)}")

        print("[3] Evaluando biparticiones en paralelo...")
        base_dist = subsistema.distribucion_marginal()
        tareas = [(subsistema, base_dist, f, p) for f, p in candidatas_list]

        mejor_phi = np.inf
        mejor_dist = DUMMY_ARR
        mejor_bipart = None

        with mp.Pool(processes=mp.cpu_count()) as pool:
            for result in tqdm(pool.imap_unordered(evaluar_biparticion, tareas), total=len(tareas), desc="    ▸ Evaluando"):
                phi, dist_part, bipart = result
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

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"resultados_geometric_{timestamp}.csv", "w", newline="", encoding="utf-8-sig") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["estrategia", "phi", "tiempo", "particion"])
            writer.writerow(["Geometric Light", mejor_phi, f"{tiempo_total:.2f}", bipart_str])

        return Solution(
            estrategia="Geometric Light",
            perdida=mejor_phi,
            distribucion_subsistema=base_dist,
            distribucion_particion=mejor_dist,
            particion=bipart_str,
            tiempo_total=tiempo_total,
            hablar=True
        )
