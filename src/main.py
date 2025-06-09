# main.py

from src.controllers.manager import Manager
from src.controllers.strategies.geometric import Geometric
from src.controllers.strategies.phi import Phi
from src.models.base.application import aplicacion
from src.controllers.strategies.q_nodesM import QNodesMod
from src.controllers.strategies.q_node_sparce import QNodesFullSparse

def iniciar_n6():
    """
    Ejemplo con n = 6 bits (N = 64 estados), para ver en consola todos los prints
    de progreso sin demorar más de ~1 segundo.
    """
    # 6 bits → "bitstrings" de longitud 6
    estado_inicio = "1000000000"  
    condiciones   = "1111111111"  
    alcance       = "1010111111"
    mecanismo     = "1111111110"

    config = Manager(estado_inicial=estado_inicio)
    geom   = Geometric(config)
    print("\nEjecutando Geometric en modo 'verbose' con n = 6 (64 estados)...\n")
    solucion = geom.aplicar_estrategia(condiciones, alcance, mecanismo) 
    print("\nSolución obtenida:")
    print(solucion)

def generar_red_20A():
    # Configurar valores necesarios en la aplicación (si no están definidos)
    aplicacion.pagina_sample_network = "A"
    aplicacion.semilla_numpy = 42  # Puedes cambiarla si necesitas resultados distintos

    estado_inicial = "0" * 20  # Estado inicial de 20 bits

    # Crear instancia del manejador
    manager = Manager(estado_inicial=estado_inicial)

    # Forzar generación sin interacción y sin preguntar por reemplazo
    filename = manager.generar_red(dimensiones=20, datos_discretos=True)

    print(f"✅ Red generada: {filename}")



if __name__ == "__main__":
    #generar_red_20A()
    iniciar_n6()


