"""Versiones de esquema que el núcleo ligero puede leer sin el stack research.

``CONFIG_SCHEMA_VERSION`` vivía en ``src.strategy``, y ``src.provenance`` la
importaba desde ahí. Ese import arrastraba cvxpy, scipy y scikit-learn.
El valor sigue siendo 1; ``src.strategy`` lo reexporta.
"""

CONFIG_SCHEMA_VERSION = 1
