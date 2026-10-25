import numpy as np

from src import evaluation as ev


def rows():
    out = []
    for sc, hits in (("closed_recording", [1, 1]), ("closed_speakers", [1, 0]), ("open_speakers", [0, 0])):
        for s in ev.SYSTEMS:
            for v, h in zip("ai", hits):
                out.append({"system": s, "scenario": sc, "speaker": "speaker01", "true": v,
                            "pred": v if h else "u", "correct": h})
    return out


def test_accuracy_filters_and_percent():
    r = rows()
    assert ev.accuracy(r, system="dtw", scenario="closed_recording") == 100
    assert ev.accuracy(r, system="dtw", scenario="closed_speakers") == 50
    assert ev.accuracy(r, system="dtw", scenario="open_speakers", true="a") == 0


def test_summary_average_and_confusion():
    h, b = ev.summary_table(rows())
    assert h[-1] == "Rata-rata" and b[0][-1] == 50
    m = ev.confusion_matrix(rows(), "dtw", "open_speakers")
    assert m.sum() == 2 and m[0, 2] == 1  # a -> u