"""Domain routers mounted by api.app.create_app."""
from . import breast, hgsoc, immunotherapy, lung, nlp, ovarian, system, tumor_board

__all__ = [
    "breast",
    "hgsoc",
    "immunotherapy",
    "lung",
    "nlp",
    "ovarian",
    "system",
    "tumor_board",
]
