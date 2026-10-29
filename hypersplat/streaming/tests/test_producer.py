import time

from hypersplat.streaming import contract as c
from hypersplat.streaming.producer import FakeSender, Sender


def _snap(seq, loss=0.1):
    return c.TrainingSnapshot("job1", seq, 1.0, seq * 10, loss, 1000, 2.0)


def test_fake_sender_roundtrip():
    s = FakeSender()
    assert [s.next_seq("job1") for _ in range(3)] == [0, 1, 2]
    for i in range(3):
        assert s.send(_snap(i))
    s.flush()
    assert [e.seq for e in s.events()] == [0, 1, 2]
    assert s.messages[0][0] == c.TOPIC_TRAINING.name and s.dropped("job1") == 0
    s.close()


def test_full_queue_drops_without_blocking():
    gate = []

    class Slow:
        def send(self, *a, **k):
            while not gate:
                time.sleep(0.01)

        def flush(self, timeout=None):
            pass

        def close(self, timeout=None):
            pass

    s = Sender("x", queue_max=2, producer_factory=lambda _: Slow())
    t0 = time.perf_counter()
    results = [s.send(_snap(i)) for i in range(50)]
    assert time.perf_counter() - t0 < 0.05
    assert results.count(False) >= 45 and s.dropped("job1") >= 45
    gate.append(1)
    s.close(1.0)


def test_broker_down_drops_and_counts():
    def down(_):
        raise ConnectionError("no broker")

    s = Sender("x", producer_factory=down)
    for i in range(5):
        s.send(_snap(i))
    s.flush(2.0)
    assert s.dropped("job1") == 5 and s.sent == 0
    s.close(0.5)


def test_unencodable_event_dropped():
    s = FakeSender()
    s.send(c.TrainingSnapshot("job1", 0, 1.0, 0, float("nan"), 1, 1.0))  # bypassed validate()
    s.flush()
    assert s.dropped("job1") == 1 and not s.messages
    s.close()
