"""Ensure the chat message builder never returns invalid message dicts.

The OpenAI chat completion endpoint requires every element in the *messages*
array to contain a ``role`` field.  A missing key triggers a **400 Bad
Request** at runtime which previously slipped through the test-suite.  This
regression test guarantees that the helper performs the required validation.
"""

from backend.app.graphs.next_gen import build_chat_messages


def test_every_message_has_role():
    system = "You are a helpful assistant."
    history = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]

    msgs = build_chat_messages(system, history)

    assert msgs[0]["role"] == "system"  # system prompt preserved
    assert all("role" in m for m in msgs)

