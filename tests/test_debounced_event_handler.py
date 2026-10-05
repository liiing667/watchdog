from __future__ import annotations

import threading
import time

import pytest

from watchdog.events import (
    DebouncedEventHandler,
    DirModifiedEvent,
    FileCreatedEvent,
    FileModifiedEvent,
    FileMovedEvent,
    FileSystemEventHandler,
)

DELAY = 0.05


class Collector(FileSystemEventHandler):
    def __init__(self) -> None:
        self.events: list = []
        self._lock = threading.Lock()

    def on_any_event(self, event) -> None:
        with self._lock:
            self.events.append(event)

    def wait(self, count: int, timeout: float = 2.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if len(self.events) >= count:
                    return
            time.sleep(0.005)
        raise AssertionError(f"expected {count} events, got {self.events}")


def make_handler(mode: str = "last", delay: float = DELAY) -> tuple[Collector, DebouncedEventHandler]:
    collector = Collector()
    return collector, DebouncedEventHandler(collector, delay=delay, mode=mode)


def test_coalesces_burst_into_last_event():
    collector, handler = make_handler()
    events = [FileModifiedEvent("/a") for _ in range(5)]
    for event in events:
        handler.dispatch(event)
    collector.wait(1)
    assert collector.events == [events[-1]]
    handler.stop()


def test_mode_first_keeps_first_event():
    collector, handler = make_handler(mode="first")
    events = [FileModifiedEvent("/a") for _ in range(3)]
    for event in events:
        handler.dispatch(event)
    collector.wait(1)
    assert collector.events == [events[0]]
    handler.stop()


def test_quiet_window_restarts_on_new_event():
    collector, handler = make_handler()
    event = FileModifiedEvent("/a")
    handler.dispatch(event)
    time.sleep(DELAY / 2)
    handler.dispatch(event)
    collector.wait(1, timeout=1.0)
    elapsed = time.monotonic()
    # The event must not have been delivered at the end of the first window,
    # so less than a full second window should remain after the second one.
    handler.stop()
    assert time.monotonic() - elapsed < DELAY


def test_different_event_types_are_not_merged():
    collector, handler = make_handler()
    handler.dispatch(FileModifiedEvent("/a"))
    handler.dispatch(FileCreatedEvent("/a"))
    collector.wait(2)
    assert [event.event_type for event in collector.events] == ["modified", "created"]
    handler.stop()


def test_file_and_directory_events_are_merged_separately():
    collector, handler = make_handler()
    file_event = FileModifiedEvent("/a")
    dir_event = DirModifiedEvent("/a")
    handler.dispatch(file_event)
    handler.dispatch(dir_event)
    collector.wait(2)
    assert collector.events == [file_event, dir_event]
    handler.stop()


def test_moved_events_are_keyed_on_both_paths():
    collector, handler = make_handler()
    first = FileMovedEvent("/a", "/b")
    second = FileMovedEvent("/a", "/c")
    third = FileMovedEvent("/a", "/b")
    handler.dispatch(first)
    handler.dispatch(second)
    handler.dispatch(third)
    collector.wait(2)
    assert collector.events == [third, second]
    handler.stop()


def test_delay_zero_passes_events_through_synchronously():
    collector = Collector()
    handler = DebouncedEventHandler(collector, delay=0)
    events = [FileModifiedEvent("/a") for _ in range(5)]
    for event in events:
        handler.dispatch(event)
    assert collector.events == events
    handler.stop()
    assert collector.events == events


def test_stop_flushes_pending_events_synchronously():
    collector, handler = make_handler()
    events = [FileModifiedEvent(f"/{i}") for i in range(3)]
    for event in events:
        handler.dispatch(event)
    handler.stop()
    assert collector.events == events


def test_no_callback_after_stop_and_later_events_dropped():
    collector, handler = make_handler()
    handler.dispatch(FileModifiedEvent("/a"))
    handler.stop()
    count = len(collector.events)
    handler.dispatch(FileModifiedEvent("/later"))
    time.sleep(DELAY * 3)
    assert len(collector.events) == count


def test_start_stop_without_events_is_safe_and_idempotent():
    collector, handler = make_handler()
    handler.start()
    handler.start()
    handler.stop()
    handler.stop()
    assert collector.events == []


def test_invalid_arguments():
    collector = Collector()
    with pytest.raises(ValueError, match="mode"):
        DebouncedEventHandler(collector, mode="middle")
    with pytest.raises(ValueError, match="delay"):
        DebouncedEventHandler(collector, delay=-1)


def test_multiple_feeder_threads_are_safe():
    collector, handler = make_handler()
    feeder_count = 8

    def feed() -> None:
        for _ in range(20):
            handler.dispatch(FileModifiedEvent("/shared"))
            handler.dispatch(FileCreatedEvent("/other"))

    threads = [threading.Thread(target=feed) for _ in range(feeder_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    collector.wait(2)
    time.sleep(DELAY * 3)
    assert len(collector.events) == 2
    types = sorted(event.event_type for event in collector.events)
    assert types == ["created", "modified"]
    handler.stop()
