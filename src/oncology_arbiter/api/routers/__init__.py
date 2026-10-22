"""Domain routers mounted by api.app.create_app."""
from . import breast, hgsoc, lung, nlp, system, tumor_board

__all__ = ["breast", "hgsoc", "lung", "nlp", "system", "tumor_board"]
