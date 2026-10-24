"""VibeSpace data pipeline package.

Every pipeline step lives in its own module as pure, testable functions. The
``tasks`` module wraps those functions for Airflow, and ``run_pipeline`` runs
the same steps from the command line without Airflow.
"""

__version__ = "1.0.0"
