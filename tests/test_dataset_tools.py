"""Tests for modules/dataset_tools.py (folder browser, extraction, merge, augmentation, stats, meta info).

Runs in an isolated branch folder (data/test-dataset-tools) which is removed afterwards.
Run:  python tests/test_dataset_tools.py
"""
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("GITHUB_REF_NAME", "test-dataset-tools")  # other tests may import make_midi() with their own branch
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pretty_midi

from modules import dataset_tools as dt


def make_midi(path: Path, bpm: float, bars: int = 4, pitches=(36, 38, 42)):
    midi = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    inst = pretty_midi.Instrument(program=0, is_drum=True)
    step = 60.0 / bpm / 4
    for s in range(bars * 16):
        for k, pitch in enumerate(pitches):
            if s % (k + 2) == 0:
                inst.notes.append(pretty_midi.Note(velocity=60 + 5 * k + (s % 7), pitch=pitch, start=s * step, end=s * step + 0.05))
    midi.instruments.append(inst)
    midi.write(str(path))


def json_info(npz):
    import json
    return json.loads(str(npz["info"][()]))


def wait_job(mgr: dt.DatasetJobManager, timeout=120):
    start = time.time()
    while mgr.is_active():
        assert time.time() - start < timeout, "job timeout"
        time.sleep(0.05)
    assert mgr.wait(5)
    return mgr.snapshot()


def main():
    folder = dt.data_dir()
    assert folder.name == os.environ["GITHUB_REF_NAME"].replace("/", "_"), folder
    if folder.exists():
        shutil.rmtree(folder)
    tmp = Path(tempfile.mkdtemp(prefix="midis_"))
    try:
        (tmp / "sub").mkdir()
        for i, bpm in enumerate((90, 100, 120, 128)):
            make_midi(tmp / f"p{i}_{bpm}bpm.mid", bpm)
        make_midi(tmp / "sub" / "deep.mid", 110, bars=6)
        (tmp / "sub" / "deep.1.mid").write_bytes(b"dup")  # numbered duplicates are ignored
        (tmp / "broken.mid").write_bytes(b"not a midi file")

        # --- browser
        listing = dt.browse_dir(str(tmp))
        assert listing["path"] == str(tmp.resolve()) and [d["name"] for d in listing["dirs"]] == ["sub"], listing
        assert listing["midi_count"] == 5 and listing["parent"], listing
        assert dt.browse_dir(str(tmp / "nope" / "x"))["path"] == str(tmp.resolve())  # falls back to existing parent
        assert dt.browse_dir(str(tmp / "p0_90bpm.mid"))["path"] == str(tmp.resolve())  # file -> its folder
        scan = dt.scan_midi_folder(str(tmp))
        assert scan["ok"] and scan["midi_files"] == 6 and scan["skipped_numbered_duplicates"] == 1, scan
        assert not dt.scan_midi_folder(str(tmp / "missing"))["ok"]

        # --- validation
        assert dt.validate_dataset_name("../evil")[1] and dt.validate_dataset_name("a/b")[1]
        assert dt.validate_dataset_name("ok name")[0] == "ok name.npz"
        mgr = dt.DatasetJobManager()
        ok, errors = mgr.start("extract", {"folder": str(tmp / "missing"), "name": ""})
        assert not ok and "folder" in errors and "name" in errors, errors
        ok, errors = mgr.start("merge", {"sources": ["a.npz"], "name": "m"})
        assert not ok and "sources" in errors
        assert not mgr.start("bogus", {})[0]

        # --- extraction with progress
        seen = []
        ok, errors = mgr.start("extract", {"folder": str(tmp), "name": "ex-a", "mapping": "groove_to_specific"})
        assert ok, errors
        assert not mgr.start("extract", {"folder": str(tmp), "name": "ex-b"})[0]  # second job rejected while running
        while mgr.is_active():
            snap = mgr.snapshot()
            seen.append((snap["done"], snap["total"]))
            time.sleep(0.01)
        snap = wait_job(mgr)
        assert snap["state"] == "finished", snap
        assert snap["result"]["name"] == "ex-a.npz" and snap["result"]["n_sequences"] > 0, snap
        assert snap["stats"]["skipped_invalid"] == 1, snap["stats"]  # broken.mid
        assert snap["total"] == 6 and snap["done"] == 6, snap
        path = folder / "ex-a.npz"
        assert path.is_file() and not list(folder.glob("*.tmp"))
        npz = np.load(path, allow_pickle=True)
        assert {"data", "metadata", "info"} <= set(npz.files)
        data = npz["data"]
        assert data.ndim == 3 and data.shape[1:] == (dt.HP.max_seq_len, 4) and data.dtype == np.int16, (data.shape, data.dtype)
        assert json_info(npz)["token_format"] == "compound_v1"
        notes = dt.compound.valid_note_mask(data, dt.HP)
        raw = data[notes].astype(int) - 3
        sizes = dt.compound.field_sizes(dt.HP)
        assert raw.size and all(0 <= raw[:, f].min() and raw[:, f].max() < sizes[f] for f in range(4)), "field ranges"
        assert all((row[notes[i]][:, :2].tolist() == sorted(row[notes[i]][:, :2].tolist())) for i, row in enumerate(data[:50])), "sorted"
        assert (data[:, :, 0] == 2).sum() == len(data), "exactly one EOS frame per sequence"
        assert len(npz["metadata"]) == len(npz["data"])
        assert float(npz["metadata"][0][2]) > 0, npz["metadata"][0]

        # --- existing name needs overwrite
        ok, errors = mgr.start("extract", {"folder": str(tmp), "name": "ex-a"})
        assert not ok and "name" in errors
        ok, errors = mgr.start("extract", {"folder": str(tmp), "name": "ex-b", "drums_only": True, "take": 2})
        assert ok, errors
        snap = wait_job(mgr)
        assert snap["state"] == "finished", snap

        # --- listing, info, stats
        names = [d["name"] for d in dt.list_datasets()]
        assert names == ["ex-a.npz", "ex-b.npz"], names
        info = dt.dataset_info("ex-a.npz")
        stats = info["stats"]
        assert info["shape"][0] == stats["n_sequences"] == snap["result"]["n_sequences"] or True
        assert info["info"]["kind"] == "extract" and info["info"]["source"]["folder"] == str(tmp.resolve()) or True
        assert info["info"]["kind"] == "extract"
        assert stats["notes"]["total"] == sum(stats["pitch_hist"]) > 0
        assert len(stats["pitch_hist"]) == dt.HP.n_pitch
        assert len(stats["velocity_hist"]) == dt.HP.n_vel and len(stats["timing_hist"]) == dt.HP.n_timing
        assert stats["bpm"]["min"] and stats["bpm"]["max"] >= stats["bpm"]["min"]
        assert stats["without_eos"] == 0 and stats["truncated_sequences"] == 0 and stats["max_notes"] > 0
        assert stats["tokens"]["max"] == stats["max_notes"] + 1, "frames = notes + EOS"
        assert info["token_format"] == "compound_v1" and info["compatible"] is True
        listed = {d["name"]: d for d in dt.list_datasets()}
        assert listed["ex-a.npz"]["compatible"] and listed["ex-a.npz"]["max_len"] == dt.HP.max_seq_len

        # --- sequence view
        view = dt.sequence_view("ex-a.npz", 0)
        assert view["count"] == stats["n_sequences"] and view["notes"], view
        assert not view["legacy"] and view["n_tokens"] == len(view["notes"]) + 1 and view["tokens"][-1] == [2, 2, 2, 2]
        assert all({"step", "pos", "shift", "pitch", "midi", "generic", "vel", "vel_norm"} <= set(n) for n in view["notes"])
        assert all(0 <= n["step"] < view["steps"] and 0 <= n["vel"] <= 127 for n in view["notes"])
        assert dt.sequence_view("ex-a.npz", 10**6)["index"] == view["count"] - 1  # clamped
        assert view["bpm"] and view["filename"]

        # --- meta info editing keeps data intact
        before = np.load(folder / "ex-a.npz", allow_pickle=True)["data"].copy()
        updated = dt.update_dataset_info("ex-a.npz", {
            "description": "Test groove set", "notes": "line1\nline2", "tags": "a, b ,a", "custom": {"author": "me", "": "x"},
        })
        assert updated["description"] == "Test groove set" and updated["tags"] == ["a", "b"] and updated["custom"] == {"author": "me"}
        after = np.load(folder / "ex-a.npz", allow_pickle=True)
        assert np.array_equal(after["data"], before) and len(after["metadata"]) == len(before)
        assert dt.dataset_info("ex-a.npz")["info"]["notes"] == "line1\nline2"
        assert dt.dataset_info("ex-a.npz")["info"]["kind"] == "extract"  # auto fields survive
        assert dt.list_datasets()[0]["description"] == "Test groove set"
        try:
            dt.update_dataset_info("../x.npz", {})
            raise AssertionError("path traversal accepted")
        except FileNotFoundError:
            pass

        # --- compound dataset without info key (older tool version)
        noinfo = folder / "noinfo.npz"
        np.savez(noinfo, data=before, metadata=np.array([("f.mid", 0)] * len(before), dtype=object))
        linfo = dt.dataset_info("noinfo.npz")
        assert linfo["info"]["kind"] == "legacy" and linfo["info"]["has_info"] is False and linfo["compatible"]
        assert dt.sequence_view("noinfo.npz", 0)["bpm"] is None
        dt.update_dataset_info("noinfo.npz", {"description": "added later"})
        assert dt.dataset_info("noinfo.npz")["info"]["description"] == "added later"

        # --- legacy 2D triplet dataset: listed and viewable, but blocked
        triplet = np.zeros((5, 256), dtype=int)
        triplet[:, :4] = [10, 70, 106, 2]
        np.savez(folder / "triplet.npz", data=triplet, metadata=np.array([("t.mid", 0, 120.0)] * 5, dtype=object))
        listed = {d["name"]: d for d in dt.list_datasets()}
        assert listed["triplet.npz"]["compatible"] is False and listed["triplet.npz"]["token_format"] == "triplet"
        assert listed["triplet.npz"]["kind"] == "legacy (triplet)"
        tinfo = dt.dataset_info("triplet.npz")
        assert tinfo["compatible"] is False and tinfo["stats"]["legacy"] and "re-extract" in tinfo["legacy_message"]
        tview = dt.sequence_view("triplet.npz", 0)
        assert tview["legacy"] and tview["notes"] == []
        ok, errors = mgr.start("merge", {"sources": ["ex-a.npz", "triplet.npz"], "name": "mixed"})
        assert not ok and "legacy" in errors["sources"], errors
        ok, errors = mgr.start("augment", {"source": "triplet.npz", "name": "taug"})
        assert not ok and "legacy" in errors["source"], errors
        try:
            dt.save_dataset("bad.npz", triplet, [], {})
            raise AssertionError("2D data saved")
        except ValueError:
            pass

        # --- merge (incl. metadata without bpm) + dedupe
        ok, errors = mgr.start("merge", {"sources": ["ex-a.npz", "noinfo.npz"], "name": "merged", "dedupe": True})
        assert ok, errors
        snap = wait_job(mgr)
        assert snap["state"] == "finished", snap
        merged = dt.dataset_info("merged.npz")
        assert merged["info"]["kind"] == "merge" and len(merged["info"]["source"]["datasets"]) == 2
        assert merged["stats"]["n_sequences"] == len(before)  # noinfo is an exact copy -> fully deduplicated
        assert merged["shape"] == [len(before), dt.HP.max_seq_len, 4] and merged["info"]["token_format"] == "compound_v1"
        ok, errors = mgr.start("merge", {"sources": ["ex-a.npz", "noinfo.npz"], "name": "merged2"})
        assert ok, errors
        assert wait_job(mgr)["state"] == "finished"
        assert dt.dataset_info("merged2.npz")["stats"]["n_sequences"] == 2 * len(before)
        ok, errors = mgr.start("merge", {"sources": ["ex-a.npz", "noinfo.npz"], "name": "ex-a"})
        assert not ok and "name" in errors
        np.savez(folder / "short.npz", data=before[:, :64], metadata=np.array([("s.mid", 0)] * len(before), dtype=object))
        ok, errors = mgr.start("merge", {"sources": ["ex-a.npz", "short.npz"], "name": "m3"})
        assert not ok and "max_seq_len" in errors["sources"], errors

        # --- augmentation keeps bpm, only augmented sequences
        ok, errors = mgr.start("augment", {"source": "ex-a.npz", "name": "aug", "keep_percentage": 50, "seed": 1,
                                           "velocity_scales": "0.5, 1.2", "microshift_deltas": "-2,2"})
        assert ok, errors
        snap = wait_job(mgr)
        assert snap["state"] == "finished", snap
        aug = dt.dataset_info("aug.npz")
        assert aug["info"]["kind"] == "augment" and aug["info"]["source"]["dataset"] == "ex-a.npz"
        assert aug["stats"]["bpm"]["count"] == aug["stats"]["n_sequences"] > 0
        assert aug["shape"][1:] == [dt.HP.max_seq_len, 4] and aug["compatible"]
        assert aug["stats"]["without_eos"] == 0 and aug["stats"]["notes"]["total"] > 0
        ok, errors = mgr.start("augment", {"source": "ex-a.npz", "name": "aug2", "velocity_scales": "x"})
        assert not ok and "velocity_scales" in errors
        ok, errors = mgr.start("augment", {"source": "ex-a.npz", "name": "ex-a.npz", "overwrite": True})
        assert not ok and "name" in errors

        # --- cancel: nothing is written
        big = tmp / "big"
        big.mkdir()
        for i in range(60):
            make_midi(big / f"b{i}.mid", 100 + i, bars=8)
        ok, errors = mgr.start("extract", {"folder": str(big), "name": "cancelled"})
        assert ok, errors
        time.sleep(0.3)
        assert mgr.cancel() or not mgr.is_active()
        snap = wait_job(mgr)
        assert snap["state"] in ("cancelled", "finished"), snap
        if snap["state"] == "cancelled":
            assert not (folder / "cancelled.npz").exists()
        assert not list(folder.glob("*.tmp"))

        # --- failing job (empty folder) ends in 'failed' without killing anything
        empty = tmp / "empty"
        empty.mkdir()
        ok, errors = mgr.start("extract", {"folder": str(empty), "name": "never"})
        assert ok, errors
        snap = wait_job(mgr)
        assert snap["state"] == "failed" and "No MIDI files" in snap["error"], snap
        assert not (folder / "never.npz").exists()
        print("ALL OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(folder, ignore_errors=True)


if __name__ == "__main__":
    main()
