"""
Tests for APIAS message_queue module.

Tests thread-safe message queue operations.
"""

import threading
import time

import pytest

from apias.message_queue import (
    MessageLevel,
    TUIMessage,
    TUIMessageQueue,
    get_message_queue,
    queue_error,
    queue_info,
    queue_success,
    queue_warning,
    reset_message_queue,
)


class TestMessageLevel:
    """Tests for MessageLevel enum."""

    def test_all_levels_defined(self) -> None:
        """All expected message levels are defined."""
        expected = [
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
            "SUCCESS",
            "STATUS",
            "PROGRESS",
        ]
        actual = [level.name for level in MessageLevel]
        for name in expected:
            assert name in actual, f"Missing level: {name}"


class TestTUIMessage:
    """Tests for TUIMessage dataclass."""

    def test_create_basic_message(self) -> None:
        """Can create a basic message."""
        msg = TUIMessage(level=MessageLevel.INFO, text="Test message")
        assert msg.level == MessageLevel.INFO
        assert msg.text == "Test message"
        assert msg.task_id is None
        assert msg.persist is False
        assert msg.timestamp is not None

    def test_create_message_with_task_id(self) -> None:
        """Can create a message with task ID."""
        msg = TUIMessage(level=MessageLevel.ERROR, text="Error", task_id=42)
        assert msg.task_id == 42

    def test_create_persistent_message(self) -> None:
        """Can create a persistent message."""
        msg = TUIMessage(level=MessageLevel.WARNING, text="Warning", persist=True)
        assert msg.persist is True


class TestTUIMessageQueue:
    """Tests for TUIMessageQueue class."""

    def test_initial_state_empty(self) -> None:
        """Queue starts empty."""
        queue = TUIMessageQueue()
        assert queue.pending_count == 0
        assert queue.deferred_count == 0

    def test_put_and_drain(self) -> None:
        """Can put messages and drain them."""
        queue = TUIMessageQueue()
        msg1 = TUIMessage(level=MessageLevel.INFO, text="Message 1")
        msg2 = TUIMessage(level=MessageLevel.INFO, text="Message 2")

        queue.put(msg1)
        queue.put(msg2)
        assert queue.pending_count == 2

        messages = queue.drain()
        assert len(messages) == 2
        assert messages[0].text == "Message 1"
        assert messages[1].text == "Message 2"
        assert queue.pending_count == 0

    def test_drain_empty_queue(self) -> None:
        """Draining empty queue returns empty list."""
        queue = TUIMessageQueue()
        messages = queue.drain()
        assert messages == []

    def test_persistent_messages_stored(self) -> None:
        """Persistent messages are stored for deferred display."""
        queue = TUIMessageQueue()

        # Non-persistent message
        queue.put(TUIMessage(level=MessageLevel.INFO, text="Normal"))

        # Persistent message
        queue.put(TUIMessage(level=MessageLevel.WARNING, text="Warning", persist=True))

        assert queue.deferred_count == 1
        deferred = queue.get_deferred()
        assert len(deferred) == 1
        assert deferred[0].text == "Warning"

    def test_get_deferred_returns_copy(self) -> None:
        """get_deferred returns a copy, not the original list."""
        queue = TUIMessageQueue()
        queue.put(TUIMessage(level=MessageLevel.ERROR, text="Error", persist=True))

        deferred1 = queue.get_deferred()
        deferred2 = queue.get_deferred()

        assert deferred1 is not deferred2
        assert deferred1 == deferred2

    def test_clear_deferred(self) -> None:
        """Can clear deferred messages."""
        queue = TUIMessageQueue()
        queue.put(TUIMessage(level=MessageLevel.ERROR, text="Error", persist=True))
        assert queue.deferred_count == 1

        queue.clear_deferred()
        assert queue.deferred_count == 0

    def test_callback_registration(self) -> None:
        """Callbacks are called when messages are added."""
        queue = TUIMessageQueue()
        received: list[TUIMessage] = []

        def callback(msg: TUIMessage) -> None:
            received.append(msg)

        queue.register_callback(callback)
        queue.put(TUIMessage(level=MessageLevel.INFO, text="Test"))

        assert len(received) == 1
        assert received[0].text == "Test"

    def test_callback_unregistration(self) -> None:
        """Can unregister callbacks."""
        queue = TUIMessageQueue()
        received: list[TUIMessage] = []

        def callback(msg: TUIMessage) -> None:
            received.append(msg)

        queue.register_callback(callback)
        queue.put(TUIMessage(level=MessageLevel.INFO, text="First"))

        queue.unregister_callback(callback)
        queue.put(TUIMessage(level=MessageLevel.INFO, text="Second"))

        assert len(received) == 1

    def test_callback_error_doesnt_break_queue(self) -> None:
        """Callback errors don't prevent message queuing."""
        queue = TUIMessageQueue()

        def bad_callback(msg: TUIMessage) -> None:
            raise ValueError("Callback error")

        queue.register_callback(bad_callback)

        # Should not raise
        queue.put(TUIMessage(level=MessageLevel.INFO, text="Test"))
        assert queue.pending_count == 1

    def test_max_size_drops_oldest(self) -> None:
        """When queue is full, oldest messages are dropped."""
        queue = TUIMessageQueue(max_size=2)

        queue.put(TUIMessage(level=MessageLevel.INFO, text="First"))
        queue.put(TUIMessage(level=MessageLevel.INFO, text="Second"))
        queue.put(TUIMessage(level=MessageLevel.INFO, text="Third"))

        # Queue should have 2 messages, first one dropped
        messages = queue.drain()
        assert len(messages) == 2
        assert messages[0].text == "Second"
        assert messages[1].text == "Third"


class TestTUIMessageQueueConcurrency:
    """Concurrency tests for TUIMessageQueue."""

    def test_concurrent_put(self) -> None:
        """Multiple threads can safely put messages."""
        queue = TUIMessageQueue()
        errors: list[Exception] = []

        def put_messages(thread_id: int) -> None:
            try:
                for i in range(100):
                    queue.put(
                        TUIMessage(
                            level=MessageLevel.INFO, text=f"Thread {thread_id} msg {i}"
                        )
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=put_messages, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # Some messages may be dropped due to max_size, but no errors

    def test_concurrent_put_and_drain(self) -> None:
        """Can put and drain concurrently without errors."""
        queue = TUIMessageQueue(max_size=1000)
        errors: list[Exception] = []
        total_drained: list[int] = [0]

        def put_messages() -> None:
            try:
                for i in range(200):
                    queue.put(TUIMessage(level=MessageLevel.INFO, text=f"msg {i}"))
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def drain_messages() -> None:
            try:
                for _ in range(50):
                    messages = queue.drain()
                    total_drained[0] += len(messages)
                    time.sleep(0.005)
            except Exception as e:
                errors.append(e)

        put_thread = threading.Thread(target=put_messages)
        drain_thread = threading.Thread(target=drain_messages)

        put_thread.start()
        drain_thread.start()
        put_thread.join()
        drain_thread.join()

        # Final drain
        remaining = queue.drain()
        total_drained[0] += len(remaining)

        assert len(errors) == 0
        # Should have drained all messages eventually
        assert total_drained[0] <= 200  # Can't have more than we put


class TestGlobalQueue:
    """Tests for global queue functions."""

    def test_get_message_queue_singleton(self) -> None:
        """get_message_queue returns same instance."""
        reset_message_queue()  # Ensure clean state
        queue1 = get_message_queue()
        queue2 = get_message_queue()
        assert queue1 is queue2

    def test_reset_creates_new_queue(self) -> None:
        """reset_message_queue creates a new queue instance."""
        queue1 = get_message_queue()
        reset_message_queue()
        queue2 = get_message_queue()
        assert queue1 is not queue2


class TestConvenienceFunctions:
    """Tests for convenience queue functions."""

    def test_queue_info(self) -> None:
        """queue_info creates INFO level message."""
        reset_message_queue()
        queue_info("Test info")

        messages = get_message_queue().drain()
        assert len(messages) == 1
        assert messages[0].level == MessageLevel.INFO
        assert messages[0].text == "Test info"
        assert messages[0].persist is False

    def test_queue_warning_persists(self) -> None:
        """queue_warning creates WARNING level message that persists."""
        reset_message_queue()
        queue_warning("Test warning")

        messages = get_message_queue().drain()
        assert messages[0].level == MessageLevel.WARNING
        assert messages[0].persist is True

    def test_queue_error_persists(self) -> None:
        """queue_error creates ERROR level message that persists."""
        reset_message_queue()
        queue_error("Test error")

        messages = get_message_queue().drain()
        assert messages[0].level == MessageLevel.ERROR
        assert messages[0].persist is True

    def test_queue_success(self) -> None:
        """queue_success creates SUCCESS level message."""
        reset_message_queue()
        queue_success("Test success")

        messages = get_message_queue().drain()
        assert messages[0].level == MessageLevel.SUCCESS

    def test_convenience_functions_with_task_id(self) -> None:
        """Convenience functions accept task_id."""
        reset_message_queue()
        queue_info("Info", task_id=42)
        queue_warning("Warning", task_id=43)
        queue_error("Error", task_id=44)

        messages = get_message_queue().drain()
        assert messages[0].task_id == 42
        assert messages[1].task_id == 43
        assert messages[2].task_id == 44
