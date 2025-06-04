from src.controllers.strategies.force import BruteForce
from src.controllers.manager import Manager
from src.controllers.strategies.q_nodes import QNodes


def iniciar():
    """Punto de entrada principal"""
                   # ABCD #
    estado_inicio = "100000000000000"
    condiciones =   "111111111111111"
    alcance =       "111111111111110"
    mecanismo =     "111111111111111"

    config_sistema = Manager(estado_inicial=estado_inicio)

    ### Ejemplo de solución mediante módulo de fuerza bruta ###
    analizador_fb =     QNodes(config_sistema)
    sia_uno = analizador_fb.aplicar_estrategia(condiciones, alcance, mecanismo)

    print(sia_uno)