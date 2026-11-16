"""Pytest plugin (``-p vibespace_model.pytest_openmp`` in pytest.ini).

Pytest imports ``-p`` plugins before it starts capturing output, so re-executing here (macOS only, see
:mod:`vibespace_model.runtime`) keeps the normal terminal output of the test run.
"""

from vibespace_model.runtime import ensure_single_openmp

ensure_single_openmp()
