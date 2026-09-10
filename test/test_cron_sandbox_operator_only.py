"""``sandbox`` is an OPERATOR opt-in, not an agent-reachable one.

A script cron runs under the ``cc`` sandbox, which hides the credential stores
from the child. ``"standard"`` widens that, so WHO may write the field is itself
the control: a prompt-injected agent under an auto-approving session must not be
able to widen the sandbox its own next script runs in. That is the rule the
vault-secret grant flow already enforces -- the agent may record a REQUEST, only
the operator mints the grant -- applied here as "the agent cannot ask at all".

Reachable from: the dashboard REST PATCH handler, and only when the caller is the
OWNER. Not reachable from: MCP ``cron_add`` / ``cron_update``, and not from the
CLI -- a shell cannot tell an operator from the agent that shares it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from body_stream_helpers import attach_body

from kiro_crew.cron import CronService
from kiro_crew.dashboard.handlers import api_cron_update
from kiro_crew.validation import MCP_CRON_SCHEMAS, ValidationError, validate_tool_args


def _make_request(body: dict, job_id: str = "abc123") -> MagicMock:
    mock_state = MagicMock()
    mock_job = MagicMock()
    mock_job.id = job_id
    mock_state.crons.update_job_async = AsyncMock(return_value=mock_job)
    mock_state.crons.set_sandbox_async = AsyncMock(return_value=mock_job)

    request = MagicMock()
    request.app = {"state": mock_state}
    request.match_info = {"job_id": job_id}
    attach_body(request, body)
    return request


def _as_owner():
    """Satisfy the owner gate, so a test can exercise the field itself.

    Patches the gate rather than forging a credential: what the gate accepts is
    ``is_owner_dashboard_request``'s business and is tested where it lives.
    """
    return patch(
        "kiro_crew.dashboard.handlers.cron.require_owner_dashboard_request",
        AsyncMock(return_value=None),
    )


def _as_non_owner():
    """Deny the owner gate the way the shared helper does -- a 403 response."""
    return patch(
        "kiro_crew.dashboard.handlers.cron.require_owner_dashboard_request",
        AsyncMock(
            return_value=web.json_response(
                {"error": "owner only", "code": "owner_only"}, status=403
            )
        ),
    )


class TestMcpToolsCannotSetIt:
    @pytest.mark.parametrize("tool", ["cron_add", "cron_update"])
    def test_the_tool_schema_declares_no_sandbox_field(self, tool: str) -> None:
        names = {spec.name for spec in MCP_CRON_SCHEMAS[tool].fields}
        assert "sandbox" not in names, (
            f"{tool} must not accept 'sandbox': an agent that can widen its own "
            "script's sandbox has defeated the cc default. Only the "
            "owner-gated dashboard REST handler may write it."
        )

    @pytest.mark.parametrize(
        "tool,args",
        [
            ("cron_add", {"name": "j", "every": 300, "sandbox": "standard"}),
            ("cron_update", {"job_id": "abc123", "sandbox": "standard"}),
        ],
    )
    def test_passing_it_anyway_is_refused(self, tool: str, args: dict) -> None:
        """The validator rejects unknown keys, so the omission above is enforced
        rather than merely documented -- a smuggled 'sandbox' is an error, not a
        value that lands somewhere by accident."""
        with pytest.raises(ValidationError):
            validate_tool_args(args, MCP_CRON_SCHEMAS[tool])

    @pytest.mark.parametrize("tool", ["cron_add", "cron_update"])
    def test_the_advertised_input_schema_has_no_sandbox_property(self, tool: str) -> None:
        """The model reads the advertised schema, not the validator. Advertising
        a field the validator refuses would make every use of it an error."""
        from kiro_crew.mcp_cron import _list_tools

        defn = next(t for t in _list_tools() if t["name"] == tool)
        assert "sandbox" not in defn["inputSchema"]["properties"]


class TestRestHandlerAcceptsIt:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("value", ["cc", "standard", ""])
    async def test_a_valid_value_reaches_the_store(self, value: str) -> None:
        request = _make_request({"sandbox": value})

        with _as_owner():
            resp = await api_cron_update(request)

        assert resp.status == 200
        request.app["state"].crons.set_sandbox_async.assert_awaited_once_with("abc123", value)
        request.app["state"].crons.update_job_async.assert_not_called()

    @pytest.mark.asyncio
    async def test_null_narrows_to_the_default(self) -> None:
        request = _make_request({"sandbox": None})

        with _as_owner():
            resp = await api_cron_update(request)

        assert resp.status == 200
        request.app["state"].crons.set_sandbox_async.assert_awaited_once_with("abc123", "")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("bad", ["off", "strict", "CC", 1, ["cc"]])
    async def test_an_invalid_value_is_refused_and_nothing_is_written(self, bad: object) -> None:
        """'strict' is refused too: that profile belongs to a secret-granted run
        and is chosen by the runner, never stored as an operator's pick."""
        request = _make_request({"sandbox": bad})

        with _as_owner():
            resp = await api_cron_update(request)

        assert resp.status == 400
        request.app["state"].crons.set_sandbox_async.assert_not_called()
        request.app["state"].crons.update_job_async.assert_not_called()


class TestTheCliCannotSetIt:
    """The CLI is not an operator surface for this field.

    A shell is not evidence of a human: the agent runs its commands in the same
    shell, so a `--sandbox` flag there would be reachable by the very caller the
    field exists to keep out. The write path is the owner-gated REST handler,
    which needs a credential an agent does not hold.
    """

    def test_the_update_path_ignores_a_sandbox_argument(self, tmp_path: Path) -> None:
        """Defence in depth: even handed a `sandbox` attribute, `_cron` must not
        forward it. This is the invariant that survives someone re-adding a flag
        without reading why it went away."""
        svc = CronService(base_dir=tmp_path)
        svc._load()
        job = svc.add_job(name="s", message="m", every_secs=60, script="x.py:run")
        assert job.sandbox == ""

        from kiro_crew.cli_commands import _cron

        args = argparse.Namespace(
            cron_action="update", job_id=job.id, name="renamed", sandbox="standard"
        )
        with patch("kiro_crew.cli_commands.config_dir", return_value=tmp_path):
            _cron(args)

        reloaded = CronService(base_dir=tmp_path)
        reloaded._load()
        assert reloaded.get_job(job.id).name == "renamed"
        assert reloaded.get_job(job.id).sandbox == "", (
            "the CLI must not write `sandbox`: an agent shares the operator's "
            "shell, so a CLI write path is an agent write path"
        )

    def test_an_operator_widened_job_is_left_alone(self, tmp_path: Path) -> None:
        """A CLI update of another field must not reset a job the owner widened."""
        svc = CronService(base_dir=tmp_path)
        svc._load()
        job = svc.add_job(name="s", message="m", every_secs=60, script="x.py:run")
        svc.set_sandbox(job.id, "standard")

        from kiro_crew.cli_commands import _cron

        args = argparse.Namespace(cron_action="update", job_id=job.id, name="renamed")
        with patch("kiro_crew.cli_commands.config_dir", return_value=tmp_path):
            _cron(args)

        reloaded = CronService(base_dir=tmp_path)
        reloaded._load()
        assert reloaded.get_job(job.id).sandbox == "standard"
        assert reloaded.get_job(job.id).name == "renamed"


class TestOnlyTheOwnerMayWriteIt:
    """A dashboard token is not an owner token.

    One is minted for every allowed Slack user (``!dashboard``), who is not the
    owner. Widening a script job hands agent-authored code the host credential
    stores, so this field draws the same boundary the vault grant draws.
    """

    @pytest.mark.asyncio
    async def test_a_non_owner_is_refused_and_nothing_is_written(self) -> None:
        request = _make_request({"sandbox": "standard"})

        with _as_non_owner():
            resp = await api_cron_update(request)

        assert resp.status == 403
        request.app["state"].crons.set_sandbox_async.assert_not_called()

    @pytest.mark.asyncio
    async def test_a_non_owner_is_refused_before_the_value_is_judged(self) -> None:
        """The gate runs first, so a refusal never doubles as a validity oracle:
        a non-owner gets 403 for a bad value too, not the 400 that would tell
        them which values the field takes."""
        request = _make_request({"sandbox": "bogus"})

        with _as_non_owner():
            resp = await api_cron_update(request)

        assert resp.status == 403
        request.app["state"].crons.set_sandbox_async.assert_not_called()

    @pytest.mark.asyncio
    async def test_every_other_field_stays_open_to_a_non_owner(self) -> None:
        """The gate is scoped to `sandbox`. An ordinary PATCH from an allowed
        non-owner still works -- the fix must not close the whole endpoint."""
        request = _make_request({"name": "renamed"})

        with _as_non_owner():
            resp = await api_cron_update(request)

        assert resp.status == 200
        _, kwargs = request.app["state"].crons.update_job_async.call_args
        assert kwargs.get("name") == "renamed"
        assert "sandbox" not in kwargs

    @pytest.mark.asyncio
    async def test_narrowing_is_gated_too(self) -> None:
        """`cc` is the safe direction, and it is gated anyway: the field's
        contract is operator-only in both directions, and one rule is what keeps
        the two from drifting apart."""
        request = _make_request({"sandbox": "cc"})

        with _as_non_owner():
            resp = await api_cron_update(request)

        assert resp.status == 403
        request.app["state"].crons.set_sandbox_async.assert_not_called()


class TestUpdateJobRefusesTheField:
    """``update_job`` is the GENERIC path, so it must refuse `sandbox` outright.

    The App SDK forwards an app's own ``**kwargs`` verbatim into
    ``CronService.update_job``, so a field merely VALIDATED there is a field a
    third-party app can set on a script job it owns -- and ``standard`` hands
    agent-authored code the host credential stores. Refusing is what makes the
    owner-only claim true; validating would accept the value.
    """

    def test_a_generic_update_is_refused(self, tmp_path: Path) -> None:
        svc = CronService(base_dir=tmp_path)
        svc._load()
        job = svc.add_job(name="s", message="m", every_secs=60, script="x.py:run")

        with pytest.raises(ValueError, match="not settable through update_job"):
            svc.update_job(job.id, sandbox="standard")

        reloaded = CronService(base_dir=tmp_path)
        reloaded._load()
        assert reloaded.get_job(job.id).sandbox == ""

    def test_the_refusal_survives_an_sdk_shaped_kwargs_forward(self, tmp_path: Path) -> None:
        """The App SDK's shape is ``update_job(job_id, **app_kwargs)``, so the
        field arrives mixed in with legitimate ones. It must still be refused,
        and nothing in the same call may land."""
        svc = CronService(base_dir=tmp_path)
        svc._load()
        job = svc.add_job(name="s", message="m", every_secs=60, script="x.py:run")

        app_kwargs = {"name": "renamed", "sandbox": "standard"}
        with pytest.raises(ValueError, match="not settable through update_job"):
            svc.update_job(job.id, **app_kwargs)

        reloaded = CronService(base_dir=tmp_path)
        reloaded._load()
        assert reloaded.get_job(job.id).sandbox == ""
        assert reloaded.get_job(job.id).name == "s"

    def test_the_owner_writer_does_set_it(self, tmp_path: Path) -> None:
        """The counterpart: the owner-only path writes the same field fine, so
        the refusal above is about the CALLER, not about the value."""
        svc = CronService(base_dir=tmp_path)
        svc._load()
        job = svc.add_job(name="s", message="m", every_secs=60, script="x.py:run")

        assert svc.set_sandbox(job.id, "standard") is not None

        reloaded = CronService(base_dir=tmp_path)
        reloaded._load()
        assert reloaded.get_job(job.id).sandbox == "standard"

    def test_the_owner_writer_still_rejects_a_junk_value(self, tmp_path: Path) -> None:
        svc = CronService(base_dir=tmp_path)
        svc._load()
        job = svc.add_job(name="s", message="m", every_secs=60, script="x.py:run")

        with pytest.raises(ValueError, match="Invalid sandbox"):
            svc.set_sandbox(job.id, "off")

    def test_the_owner_writer_reports_a_missing_job(self, tmp_path: Path) -> None:
        svc = CronService(base_dir=tmp_path)
        svc._load()
        assert svc.set_sandbox("nope", "standard") is None


class TestImportStampsTheDefault:
    """An import is not an upgrade.

    The loader reads a MISSING key on a script record as "written before the cc
    default" and keeps it on the wide profile -- right for this host's own
    pre-upgrade store, wrong for a record arriving from somewhere else, which
    would get the wide sandbox with nobody having asked for it.
    """

    @staticmethod
    def _store(tmp_path: Path, *records: dict) -> Path:
        path = tmp_path / "crons.json"
        path.write_text(json.dumps({"jobs": list(records)}), encoding="utf-8")
        return path

    @staticmethod
    def _record(**over: object) -> dict:
        rec: dict = {
            "id": "j1",
            "name": "imported",
            "message": "args",
            "schedule": {"kind": "every", "every_secs": 60},
        }
        rec.update(over)
        return rec

    def test_an_imported_script_record_gets_the_explicit_default(self, tmp_path: Path) -> None:
        from kiro_crew.portability import _sanitize_imported_crons

        path = self._store(tmp_path, self._record(script="x.py:run"))
        _sanitize_imported_crons(path)

        stored = json.loads(path.read_text(encoding="utf-8"))["jobs"][0]
        assert stored["sandbox"] == ""

        svc = CronService(base_dir=tmp_path)
        svc._load()
        assert svc.get_job("j1").sandbox == ""

    def test_an_imported_script_record_loses_an_explicit_wide_value(self, tmp_path: Path) -> None:
        """An import carrying ``sandbox="standard"`` is NARROWED, not honoured.

        The file saying so is not authorization: the field is writable only from
        the owner-gated REST handler, and an import arrives with none. Leaving the
        value would let a crafted export hand agent-authored code the host
        credential stores the first time a human enables the paused job.
        """
        from kiro_crew.portability import _sanitize_imported_crons

        path = self._store(tmp_path, self._record(script="x.py:run", sandbox="standard"))
        _sanitize_imported_crons(path)

        stored = json.loads(path.read_text(encoding="utf-8"))["jobs"][0]
        assert stored["sandbox"] == ""

        svc = CronService(base_dir=tmp_path)
        svc._load()
        assert svc.get_job("j1").sandbox == ""

    def test_an_imported_script_record_keeps_an_explicit_cc(self, tmp_path: Path) -> None:
        """The narrow value is already the default, so it round-trips untouched --
        the rule narrows, it does not churn every record."""
        from kiro_crew.portability import _sanitize_imported_crons

        path = self._store(tmp_path, self._record(script="x.py:run", sandbox="cc"))
        _sanitize_imported_crons(path)

        stored = json.loads(path.read_text(encoding="utf-8"))["jobs"][0]
        assert stored["sandbox"] == ""

    def test_a_non_script_import_is_not_stamped(self, tmp_path: Path) -> None:
        """Nothing reads the field for an agent job, so there is nothing to fill."""
        from kiro_crew.portability import _sanitize_imported_crons

        path = self._store(tmp_path, self._record())
        _sanitize_imported_crons(path)

        stored = json.loads(path.read_text(encoding="utf-8"))["jobs"][0]
        assert "sandbox" not in stored
