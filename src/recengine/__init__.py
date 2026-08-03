"""A product recommender for the Online Retail transaction log.

The package is deliberately split so each piece can be tested on its own:

``recengine.data``
    Loading and cleaning the transaction log, the binary user-item matrix, and
    the temporal train/test split.
``recengine.models``
    The recommenders, all behind one ``fit`` / ``recommend`` interface so the
    evaluation cannot accidentally favour one of them.
``recengine.evaluate``
    Top-N ranking metrics and the evaluation loop.
``recengine.api``
    The Flask application, which loads a trained artifact and does no fitting.
"""

__version__ = "0.2.0"
