"""The cron store's ``sandbox`` field: round-trip, legacy profile, write boundaries.

An ungranted script cron runs under ``cc``: ``~/.aws/credentials``,
``~/.kube/config`` and ``~/.netrc`` are hidden from the child, while
``~/.aws/config`` stays readable so ``credential_process`` auth works. The
WIDE ``standard`` profile leaves all three readable, and the static text scan
in front of it disclaims being a fence in its own docstring.

Two properties carry the whole design and are pinned here:

* **Absence of the key is the pre-upgrade signal.** A script record with no
  ``sandbox`` key at all ran ``standard`` before the upgrade and keeps it, so an
  upgrade changes no behaviour for a job that already existed. Every record
  written since serializes the field explicitly, so ``""`` can never be mistaken
  for "predates the field".
* **Only the owner may widen it.** The dashboard REST PATCH handler is the sole
  write path: it gates the field on ownership and calls
  ``CronService.set_sandbox``. ``update_job`` refuses the key, so the App SDK
  cannot forward it; the MCP ``cron_add`` / ``cron_update`` tools do not carry
  it; and there is no CLI flag. A prompt-injected agent under an auto-approving
  session therefore cannot widen the sandbox its own next script runs in. That
  is the vault-secret grant rule applied to the sandbox.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kiro_crew.cron import (
    _CRON_SANDBOX_MODES,
    CronJob,
    CronSchedule,
    CronService,
    _job_from_record,
    _validate_sandbox_mode,
)


def _record(**over: object) -> dict:
    """A minimally valid on-disk cron record. ``sandbox`` is absent by default."""
    rec: dict = {
        "id": "j1",
        "name": "legacy",
        "message": "args",
        "schedule": {"kind": "every", "every_secs": 60},
    }
    rec.update(over)
    return rec


def _write_store(tmp_path: Path, *records: dict) -> Path:
    path = tmp_path / "crons.json"
    path.write_text(json.dumps({"jobs": list(records)}, indent=2), encoding="utf-8")
    return path


class TestSandboxModeValidator:
    def test_accepts_the_three_spellings(self) -> None:
        for mode in _CRON_SANDBOX_MODES:
            assert _validate_sandbox_mode(mode) == mode

    @pytest.mark.parametrize("bad", ["off", "strict", "CC", "auto", None, 1, True, ["cc"]])
    def test_refuses_everything_else(self, bad: object) -> None:
        # "strict" is refused deliberately: it is the profile a SECRET-GRANTED
        # run takes, chosen by the runner from the grant, never a stored value an
        # operator picks.
        with pytest.raises(ValueError, match="Invalid sandbox"):
            _validate_sandbox_mode(bad)


class TestLegacyProfileOnLoad:
    def test_legacy_script_record_loads_as_standard(self, tmp_path: Path) -> None:
        """No ``sandbox`` key + a script = a job that has always run wide."""
        _write_store(tmp_path, _record(script="x.py:run"))
        svc = CronService(base_dir=tmp_path)
        svc._load()
        assert svc.get_job("j1").sandbox == "standard"

    def test_legacy_script_record_differs_from_the_file(self, tmp_path: Path) -> None:
        """The in-memory job says something the file does not: the record carries
        no ``sandbox`` key at all, while the loaded job reports ``standard``."""
        _write_store(tmp_path, _record(script="x.py:run"))
        svc = CronService(base_dir=tmp_path)
        svc._load()
        stored = json.loads((tmp_path / "crons.json").read_text(encoding="utf-8"))["jobs"][0]
        assert "sandbox" not in stored
        assert svc.get_job("j1").sandbox == "standard"

    def test_the_next_save_makes_the_legacy_value_explicit(self, tmp_path: Path) -> None:
        """One write puts the key in the record, so a later load resolves the
        profile from the stored value rather than from the absence rule."""
        _write_store(tmp_path, _record(script="x.py:run"))
        svc = CronService(base_dir=tmp_path)
        svc._load()
        svc._save()
        stored = json.loads((tmp_path / "crons.json").read_text(encoding="utf-8"))["jobs"][0]
        assert stored["sandbox"] == "standard"

        svc2 = CronService(base_dir=tmp_path)
        svc2._load()
        assert svc2.get_job("j1").sandbox == "standard"

    def test_legacy_command_record_stays_on_the_default(self, tmp_path: Path) -> None:
        """Command jobs already ran cc, so there is no legacy profile to keep."""
        _write_store(tmp_path, _record(id="c1", command="echo hi"))
        svc = CronService(base_dir=tmp_path)
        svc._load()
        assert svc.get_job("c1").sandbox == ""

    def test_legacy_agent_record_stays_on_the_default(self, tmp_path: Path) -> None:
        """An agent job spawns no script child; the field is inert for it."""
        _write_store(tmp_path, _record(id="a1"))
        svc = CronService(base_dir=tmp_path)
        svc._load()
        assert svc.get_job("a1").sandbox == ""

    def test_an_explicit_empty_value_keeps_no_legacy_profile(self, tmp_path: Path) -> None:
        """This is the case the two spellings exist for: a script job written
        AFTER the upgrade that the operator left on the default must stay cc."""
        _write_store(tmp_path, _record(script="x.py:run", sandbox=""))
        svc = CronService(base_dir=tmp_path)
        svc._load()
        assert svc.get_job("j1").sandbox == ""

    def test_an_unrecognised_stored_value_resolves_to_the_safe_default(self) -> None:
        """The store is hand-editable. A junk value must narrow to cc, and must
        not drop the record — the loader's per-entry isolation would erase the
        job from disk on the next write."""
        job = _job_from_record(_record(script="x.py:run", sandbox="off"))
        assert job.sandbox == ""

    def test_a_reload_re_resolves_from_the_current_file(self, tmp_path: Path) -> None:
        """The absence rule describes the CURRENT file, not a high-water mark: a
        record rewritten with an explicit value stops reading as pre-upgrade."""
        _write_store(tmp_path, _record(script="x.py:run"))
        svc = CronService(base_dir=tmp_path)
        svc._load()
        assert svc.get_job("j1").sandbox == "standard"
        _write_store(tmp_path, _record(script="x.py:run", sandbox="cc"))
        svc._load()
        assert svc.get_job("j1").sandbox == "cc"


class TestRoundTrip:
    def test_a_new_job_serializes_the_field_explicitly(self, tmp_path: Path) -> None:
        """Written even when "" — absence of the key is a load-bearing signal."""
        svc = CronService(base_dir=tmp_path)
        svc._load()
        svc.add_job(name="s", message="m", every_secs=60, script="x.py:run")
        stored = json.loads((tmp_path / "crons.json").read_text(encoding="utf-8"))["jobs"][0]
        assert "sandbox" in stored
        assert stored["sandbox"] == ""

    def test_an_explicit_value_round_trips(self, tmp_path: Path) -> None:
        svc = CronService(base_dir=tmp_path)
        svc._load()
        job = svc.add_job(name="s", message="m", every_secs=60, script="x.py:run")
        svc.set_sandbox(job.id, "standard")
        svc2 = CronService(base_dir=tmp_path)
        svc2._load()
        assert svc2.get_job(job.id).sandbox == "standard"

    def test_the_create_path_does_not_accept_the_field_at_all(self, tmp_path: Path) -> None:
        """Create is closed by construction, not by validation. A caller cannot
        name a profile at all, so a new job always starts on the narrow one and
        widening always goes through the owner writer."""
        svc = CronService(base_dir=tmp_path)
        svc._load()
        with pytest.raises(TypeError):
            svc.add_job(  # type: ignore[call-arg]
                name="s", message="m", every_secs=60, script="x.py:run", sandbox="standard"
            )

    def test_the_owner_writer_sets_and_persists_the_value(self, tmp_path: Path) -> None:
        svc = CronService(base_dir=tmp_path)
        svc._load()
        job = svc.add_job(name="s", message="m", every_secs=60, script="x.py:run")
        assert svc.set_sandbox(job.id, "standard") is not None
        svc2 = CronService(base_dir=tmp_path)
        svc2._load()
        assert svc2.get_job(job.id).sandbox == "standard"

    def test_the_owner_writer_refuses_an_invalid_value_and_mutates_nothing(
        self, tmp_path: Path
    ) -> None:
        """The value is checked before the store is touched, so a rejected write
        leaves the record exactly as it was."""
        svc = CronService(base_dir=tmp_path)
        svc._load()
        job = svc.add_job(name="s", message="m", every_secs=60, script="x.py:run")
        with pytest.raises(ValueError, match="Invalid sandbox"):
            svc.set_sandbox(job.id, "off")
        assert svc.get_job(job.id).sandbox == ""

    def test_the_owner_writer_can_narrow_a_widened_job_back(self, tmp_path: Path) -> None:
        svc = CronService(base_dir=tmp_path)
        svc._load()
        job = svc.add_job(name="s", message="m", every_secs=60, script="x.py:run")
        svc.set_sandbox(job.id, "standard")
        svc.set_sandbox(job.id, "cc")
        assert svc.get_job(job.id).sandbox == "cc"

    def test_the_field_defaults_to_the_narrow_profile(self) -> None:
        job = CronJob(
            id="j", name="n", message="m", schedule=CronSchedule(kind="every", every_secs=60)
        )
        assert job.sandbox == ""
