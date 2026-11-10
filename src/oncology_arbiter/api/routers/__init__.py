"""Domain routers mounted by api.app.create_app."""
from . import breast, crc, hgsoc, immunotherapy, lung, nlp, ovarian, ranker, system, tumor_board

__all__ = [
    "breast",
    "crc",
    "hgsoc",
    "immunotherapy",
    "lung",
    "nlp",
    "ovarian",
    "ranker",
    "system",
    "tumor_board",
]
