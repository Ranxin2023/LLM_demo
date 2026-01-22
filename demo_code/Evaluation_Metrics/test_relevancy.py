import os
import pytest

from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric

def test_relevancy():
    # Ensure key is loaded (DeepEval will look for OPENAI_API_KEY typically)
    assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY is not set in your environment."

    metric = AnswerRelevancyMetric(
        threshold=0.5,
        model="gpt-4o"  # if this errors, see fixes below
    )

    test_case = LLMTestCase(
        input="Can I return these shoes after 30 days?",
        actual_output="Unfortunately, returns are only accepted within 30 days of purchase.",
        retrieval_context=[
            "All customers are eligible for a 30-day full refund at no extra cost.",
            "Returns are only accepted within 30 days of purchase.",
        ],
    )

    assert_test(test_case, [metric])
