from __future__ import annotations

import threading
import time

import pytest

from watchdog.events import (
    DebouncedEventHandler,
    DirModifiedEvent,
    FileModifiedEvent,
    FileMovedEvent,
    FileSystemEventHandler,
)

DELAY = 0.1


class RecordingHandler(FileSystemEventHandler):
    def __init__(self) -> None:
        self.events = []
        self.lock = threading.Lock()

    def on_any_event(self, event):
        with self.lock:
            self.events.append(event)

    def recorded(self):
        with self.lock:
            return list(self.events)


def wait_for(condition, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return False


def test_burst_coalesced_into_last_event():
    handler = RecordingHandler()
    debounced = DebouncedEventHandler(handler, delay=DELAY, mode="last")
    events = [FileModifiedEvent("/tmp/file") for _ in range(3)]
    for event in events:
        debounced.dispatch(event)

    assert wait_for(lambda: len(handler.recorded()) == 1)
    time.sleep(2 * DELAY)
    assert handler.recorded() == [events[-1]]


def test_burst_coalesced_into_first_event():
    handler = RecordingHandler()
    debounced = DebouncedEventHandler(handler, delay=DELAY, mode="first")
    events = [FileModifiedEvent("/tmp/file") for _ in range(3)]
    for event in events:
        debounced.dispatch(event)

    assert wait_for(lambda: len(handler.recorded()) == 1)
    time.sleep(2 * DELAY)
    assert handler.recorded() == [events[0]]


def test_invalid_mode_rejected():
    with pytest.raises(ValueError, match="mode"):
        DebouncedEventHandler(RecordingHandler(), mode="middle")


def test_different_paths_not_coalesced():
    handler = RecordingHandler()
    debounced = DebouncedEventHandler(handler, delay=DELAY)
    debounced.dispatch(FileModifiedEvent("/tmp/a"))
    debounced.dispatch(FileModifiedEvent("/tmp/b"))

    assert wait_for(lambda: len(handler.recorded()) == 2)
    assert {e.src_path for e in handler.recorded()} == {"/tmp/a", "/tmp/b"}


def test_file_and_directory_events_coalesced_separately():
    handler = RecordingHandler()
    debounced = DebouncedEventHandler(handler, delay=DELAY)
    debounced.dispatch(FileModifiedEvent("/tmp/x"))
    debounced.dispatch(DirModifiedEvent("/tmp/x"))
    debounced.dispatch(FileModifiedEvent("/tmp/x"))
    debounced.dispatch(DirModifiedEvent("/tmp/x"))

    assert wait_for(lambda: len(handler.recorded()) == 2)
    kinds = {(type(e), e.src_path) for e in handler.recorded()}
    assert kinds == {(FileModifiedEvent, "/tmp/x"), (DirModifiedEvent, "/tmp/x")}


def test_moved_events_keyed_on_both_paths():
    handler = RecordingHandler()
    debounced = DebouncedEventHandler(handler, delay=DELAY)
    debounced.dispatch(FileMovedEvent("/tmp/a", "/tmp/b"))
    debounced.dispatch(FileMovedEvent("/tmp/a", "/tmp/c"))
    debounced.dispatch(FileMovedEvent("/tmp/a", "/tmp/b"))

    assert wait_for(lambda: len(handler.recorded()) == 2)
    pairs = {(e.src_path, e.dest_path) for e in handler.recorded()}
    assert pairs == {("/tmp/a", "/tmp/b"), ("/tmp/a", "/tmp/c")}


def test_zero_delay_passes_through_immediately():
    handler = RecordingHandler()
    debounced = DebouncedEventHandler(handler, delay=0)
    for i in range(5):
        debounced.dispatch(FileModifiedEvent("/tmp/f"))
    assert len(handler.recorded()) == 5


def test_stop_flushes_pending_events_synchronously():
    handler = RecordingHandler()
    debounced = DebouncedEventHandler(handler, delay=60)
    debounced.dispatch(FileModifiedEvent("/tmp/a"))
    debounced.dispatch(FileModifiedEvent("/tmp/a"))
    debounced.dispatch(FileModifiedEvent("/tmp/b"))

    debounced.stop()
    # Queued events are delivered, coalesced, before stop() returns.
    assert len(handler.recorded()) == 2
    debounced.dispatch(FileModifiedEvent("/tmp/c"))
    assert len(handler.recorded()) == 2


def test_no_callbacks_after_stop():
    handler = RecordingHandler()
    debounced = DebouncedEventHandler(handler, delay=DELAY)
    debounced.dispatch(FileModifiedEvent("/tmp/a"))
    debounced.stop()
    count = len(handler.recorded())
    assert count == 1
    time.sleep(3 * DELAY)
    assert len(handler.recorded()) == count


def test_start_reenables_after_stop():
    handler = RecordingHandler()
    debounced = DebouncedEventHandler(handler, delay=DELAY)
    debounced.dispatch(FileModifiedEvent("/tmp/a"))
    debounced.stop()
    assert len(handler.recorded()) == 1

    debounced.start()
    debounced.dispatch(FileModifiedEvent("/tmp/b"))
    assert wait_for(lambda: len(handler.recorded()) == 2)


def test_concurrent_dispatch_from_multiple_threads():
    handler = RecordingHandler()
    debounced = DebouncedEventHandler(handler, delay=DELAY)
    paths = [f"/tmp/file{i}" for i in range(10)]

    def feed(path):
        for _ in range(20):
            debounced.dispatch(FileModifiedEvent(path))

    threads = [threading.Thread(target=feed, args=(p,)) for p in paths]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert wait_for(lambda: len(handler.recorded()) == len(paths))
    time.sleep(2 * DELAY)
    assert sorted(e.src_path for e in handler.recorded()) == paths


def test_wrapped_handler_receives_full_dispatch():
    class CountingHandler(FileSystemEventHandler):
        def __init__(self):
            self.any_count = 0
            self.modified_count = 0

        def on_any_event(self, event):
            self.any_count += 1

        def on_modified(self, event):
            self.modified_count += 1

    handler = CountingHandler()
    debounced = DebouncedEventHandler(handler, delay=DELAY)
    debounced.dispatch(FileModifiedEvent("/tmp/a"))
    debounced.dispatch(FileModifiedEvent("/tmp/a"))

    assert wait_for(lambda: handler.any_count == 1)
    assert handler.modified_count == 1
