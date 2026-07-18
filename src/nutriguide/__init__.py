"""NutriGuide: USDA-grounded RAG nutrition chat assistant."""

import os

# torch and faiss-cpu each bundle their own OpenMP runtime on Windows; loading
# both aborts the process ("OMP: Error #15") without this well-known override.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

__version__ = "0.2.0"
