import asyncio
import json
import operator

import pytest

from src.pipeline.policyIndexer.evaluator import PolicyIndexerEvaluator


def check_case_metric(
    summary, test_case: str, metric_key: str, expected_value, comparator=operator.eq
):
    """
    Helper function to verify that a given test case in the summary satisfies a metric condition.

    Parameters:
      summary (dict): The summary output from the evaluator.
      test_case (str): A substring to identify the specific test case.
      metric_key (str): The key for the metric to check (e.g. "FuzzyEvaluator.indel_similarity").
      expected_value: The expected value for the metric.
      comparator (callable): A function that takes two arguments and returns a boolean.
                             Defaults to operator.eq for equality.
    """
    # Find the case by matching the provided substring
    case = next((c for c in summary["cases"] if test_case in c["case"]), None)
    assert case is not None, f"Case '{test_case}' not found in summary."
    metrics = case.get("results", {}).get("metrics")
    assert metrics is not None, f"Metrics not found for case '{test_case}'."
    actual = metrics.get(metric_key)
    assert (
        actual is not None
    ), f"Metric '{metric_key}' not found for case '{test_case}'."
    assert comparator(
        actual, expected_value
    ), f"Case '{test_case}': expected {metric_key} {comparator.__name__} {expected_value}, got {actual}."


@pytest.fixture(scope="session")
def med_indexer_summary():
    """
    Runs the AgenticRagEvaluator pipeline once and yields its parsed summary output.
    After tests complete, cleans up the temporary directory.
    """
    evaluator = PolicyIndexerEvaluator(
        cases_dir="./evals/cases", temp_dir="./temp_evaluation_rag"
    )
    loop = asyncio.get_event_loop()
    summary_json = loop.run_until_complete(evaluator.run_pipeline())
    summary = (
        json.loads(summary_json) if isinstance(summary_json, str) else summary_json
    )
    yield summary
    evaluator.cleanup_temp_dir()

# Now each test is decorated with the evaluation markers and uses the evaluation_setup fixture.
@pytest.mark.evaluation
@pytest.mark.usefixtures("evaluation_setup")
def test_ocr_extraction_001(med_indexer_summary):
    check_case_metric(
        summary=med_indexer_summary,
        test_case="ocr-extraction-001.v0",
        metric_key="SlidingFuzzyEvaluator.indel_similarity",
        expected_value=75,
        comparator=operator.gt,
    )

@pytest.mark.evaluation
@pytest.mark.usefixtures("evaluation_setup")
def test_ocr_extraction_002(med_indexer_summary):
    check_case_metric(
        summary=med_indexer_summary,
        test_case="ocr-extraction-002.v0",
        metric_key="SlidingFuzzyEvaluator.indel_similarity",
        expected_value=90,
        comparator=operator.gt,
    )

@pytest.mark.evaluation
@pytest.mark.usefixtures("evaluation_setup")
def test_ocr_extraction_003(med_indexer_summary):
    check_case_metric(
        summary=med_indexer_summary,
        test_case="ocr-extraction-003.v0",
        metric_key="SlidingFuzzyEvaluator.indel_similarity",
        expected_value=85,
        comparator=operator.gt,
    )

