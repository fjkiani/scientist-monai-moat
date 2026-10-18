"""Domain routers mounted by api.app.create_app."""
from . import breast, lung, nlp, system, tumor_board

__all__ = ["breast", "lung", "nlp", "system", "tumor_board"]
