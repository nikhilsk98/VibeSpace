"""DAG integrity tests: import, structure, dependencies, retries and callbacks."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("airflow")

from airflow.exceptions import AirflowFailException  # noqa: E402
from airflow.models import DagBag  # noqa: E402

from scripts.alerts import airflow_failure_callback  # noqa: E402
from scripts.exceptions import DataAcquisitionError, TMDBAuthError  # noqa: E402
from scripts.tasks import PIPELINE_STEPS  # noqa: E402

DAGS_DIR = Path(__file__).resolve().parents[1] / "dags"
EXPECTED_TASKS = {name for name, _ in PIPELINE_STEPS}


@pytest.fixture(scope="module")
def dagbag():
    return DagBag(dag_folder=str(DAGS_DIR), include_examples=False)


def test_dags_import_without_errors(dagbag):
    assert dagbag.import_errors == {}
    assert {"vibespace_data_pipeline", "vibespace_data_pipeline_baseline"} <= set(dagbag.dag_ids)


@pytest.mark.parametrize("dag_id", ["vibespace_data_pipeline", "vibespace_data_pipeline_baseline"])
def test_tasks_retries_callbacks_docs(dagbag, dag_id):
    dag = dagbag.dags[dag_id]
    assert set(dag.task_ids) == EXPECTED_TASKS
    assert dag.catchup is False
    for task in dag.tasks:
        assert task.retries >= 1, task.task_id
        assert task.retry_exponential_backoff, task.task_id
        assert task.on_failure_callback is airflow_failure_callback, task.task_id
        assert task.doc_md, task.task_id
    assert {"tmdb_stage", "sample_size", "use_cache"} <= set(dag.params.keys())


def test_no_cycles(dagbag):
    from airflow.utils.dag_cycle_tester import check_cycle

    for dag_id in ("vibespace_data_pipeline", "vibespace_data_pipeline_baseline"):
        check_cycle(dagbag.dags[dag_id])  # raises on a cycle


def upstream(dag, task_id):
    return set(dag.get_task(task_id).upstream_task_ids)


def test_optimized_dependencies(dagbag):
    dag = dagbag.dags["vibespace_data_pipeline"]
    assert upstream(dag, "download_movielens") == set()
    assert upstream(dag, "fetch_tmdb") == {"download_movielens"}
    assert upstream(dag, "build_dna_labels") == {"download_movielens"}
    assert upstream(dag, "build_tmdb_table") == {"fetch_tmdb"}
    assert upstream(dag, "validate_raw") == {"build_tmdb_table"}
    assert upstream(dag, "join_ids") == {"validate_raw"}
    assert upstream(dag, "clean") == {"join_ids"}
    assert upstream(dag, "encode_plots") == {"clean"}
    assert upstream(dag, "validate_processed") == {"build_dna_labels", "encode_plots"}
    assert upstream(dag, "generate_stats") == {"validate_processed"}
    assert upstream(dag, "bias_slicing") == {"validate_processed"}
    assert upstream(dag, "detect_anomalies") == {"generate_stats"}
    assert upstream(dag, "dvc_version") == {"detect_anomalies", "bias_slicing"}


def test_baseline_is_a_chain(dagbag):
    dag = dagbag.dags["vibespace_data_pipeline_baseline"]
    names = [name for name, _ in PIPELINE_STEPS]
    for before, after in zip(names, names[1:]):
        assert upstream(dag, after) == {before}
    assert dag.params["tmdb_max_workers"] == 1 and dag.params["use_cache"] is False


def test_permanent_errors_skip_retries():
    from dags.data_pipeline_dag import no_retry_on_permanent_errors

    def auth_fail(**_):
        raise TMDBAuthError("bad key")

    def transient(**_):
        raise DataAcquisitionError("network")

    with pytest.raises(AirflowFailException):
        no_retry_on_permanent_errors(auth_fail)()
    with pytest.raises(DataAcquisitionError):
        no_retry_on_permanent_errors(transient)()
    assert no_retry_on_permanent_errors(lambda **_: 5)() == 5
