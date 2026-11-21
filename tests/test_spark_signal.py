"""The night-end signal to the Spark (iris/conductor/spark_signal, 2026-11-21). Pure."""
from iris.conductor import spark_signal as ss

BLOCK = {"host": "100.94.42.83", "user": "taylor", "key": "C:/Users/iriso/.ssh/id_ed25519_iris"}


def test_the_command_names_the_event_and_an_incomplete_block_is_off():
    argv = ss.command(BLOCK, "NINA_FLATS_DONE")
    assert argv[0] == "ssh" and "taylor@100.94.42.83" in argv and argv[argv.index("-i") + 1] == BLOCK["key"]
    assert "--signal NINA_FLATS_DONE" in argv[-1] and argv[-1].endswith("&")
    assert ss.command({}, "x") == [] and ss.command({"host": "h", "user": "u"}, "x") == []
    assert ss.command(dict(BLOCK, command="echo {event}"), "SHUTDOWN_DONE")[-1] == "echo SHUTDOWN_DONE"


def test_only_the_end_event_is_signalled_and_the_outcome_is_noted():
    sent, notes = [], []
    sig = ss.SparkSignal(BLOCK, runner=lambda argv: (sent.append(argv), (0, ""))[1],
                         note=lambda ev, data: notes.append((ev, data)), worker=False)
    assert sig.on_transition("NINA_FLATS_DONE", "DONE_MAIN", "DONE_FLATS", {}) is True
    assert sig.on_transition("SLOT_STARTED", "ARMED", "SLOT_IMAGING", {}) is False
    assert len(sent) == 1
    assert notes == [("SPARK_SIGNALLED", {"event": "NINA_FLATS_DONE", "rc": 0, "text": "",
                                          "host": "100.94.42.83"})]


def test_a_failed_ssh_is_a_note_and_never_raises():
    notes = []

    def boom(argv):
        raise OSError("no ssh")
    sig = ss.SparkSignal(BLOCK, runner=boom, note=lambda ev, data: notes.append((ev, data["rc"])),
                         worker=False)
    assert sig.on_transition("NINA_FLATS_DONE", "a", "b", {}) is True
    assert notes == [("SPARK_SIGNAL_FAILED", -1)]
    denied = ss.SparkSignal(BLOCK, runner=lambda argv: (255, "Permission denied"),
                            note=lambda ev, data: notes.append((ev, data["rc"])), worker=False)
    denied.on_transition("NINA_FLATS_DONE", "a", "b", {})
    assert notes[-1] == ("SPARK_SIGNAL_FAILED", 255)
    off = ss.SparkSignal(dict(BLOCK, enabled=False), runner=boom, worker=False)
    assert off.on_transition("NINA_FLATS_DONE", "a", "b", {}) is False
    assert ss.SparkSignal.from_config({}).enabled is False
