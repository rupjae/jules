"""Unit tests for retrieval_agent.need_search heuristic.

These assertions exercise both the keyword-based shortcut and the fallback
length trigger to ensure the function behaves as expected.
"""


from backend.app.agents import retrieval_agent as ra


def test_keyword_trigger():
    """Any recognised keyword should enable retrieval regardless of length."""

    assert ra.need_search("Can you cite the source?")


def test_length_trigger():
    """Prompts exceeding 75 words should trigger retrieval; shorter remain off."""

    assert ra.need_search("word " * 76)
    assert not ra.need_search("short prompt")

