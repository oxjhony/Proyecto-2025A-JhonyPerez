# main.py

from src.controllers.manager import Manager
from src.controllers.strategies.geometric import Geometric
from src.controllers.strategies.phi import Phi

def iniciar_n6():
    """
    Ejemplo con n = 6 bits (N = 64 estados), para ver en consola todos los prints
    de progreso sin demorar más de ~1 segundo.
    """
    # 6 bits → "bitstrings" de longitud 6
    estado_inicio = "1000011111"  
    condiciones   = "1101111011"  
    alcance       = "1000001111"
    mecanismo     = "1101011111"

    config = Manager(estado_inicial=estado_inicio)
    geom   = Geometric(config)
    print("\n▶︎Ejecutando Geometric en modo 'verbose' con n = 6 (64 estados)...\n")
    solucion = geom.aplicar_estrategia(condiciones, alcance, mecanismo) 
    print("\n▶︎Solución obtenida:")
    print(solucion)




if __name__ == "__main__":
    iniciar_n6()
