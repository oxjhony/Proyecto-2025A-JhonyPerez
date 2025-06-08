from src.middlewares.profile import profiler_manager
from src.models.base.application import aplicacion
from src.main import iniciar_n6


def main():
    """Inicializar el aplicativo."""
    profiler_manager.enabled = True

    # aplicacion.pagina_sample_network = "B"

    iniciar_n6()


if __name__ == "__main__":
    main()
