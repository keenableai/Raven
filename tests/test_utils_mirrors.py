"""When a download may leave its default host for a mirror, and when it may not.

A mirror is only ever a second attempt: the host the file belongs to is asked
first, an answer from it (any answer, an error included) is final, and only a
connection that never completed moves the download on. These cases pin that
line, and the one switch that turns the whole fallback off.
"""

from __future__ import annotations

import errno
import hashlib
import http.server
import json
import os
import shutil
import socket
import ssl
import subprocess
import sys
import threading
import time
import urllib.error
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from raven.utils import mirrors


class TestTheOptOut:
    @pytest.mark.parametrize("value", ["1", "true", "YES", " on "])
    def test_a_truthy_value_turns_every_fallback_off(self, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
        monkeypatch.setenv("RAVEN_NO_MIRROR", value)

        assert not mirrors.enabled()

    @pytest.mark.parametrize("value", ["", "0", "no"])
    def test_anything_else_leaves_them_on(self, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
        """Read the way raven's other opt-outs read, so the installer and the
        `raven resources` it starts agree on what one value means."""
        monkeypatch.setenv("RAVEN_NO_MIRROR", value)

        assert mirrors.enabled()

    def test_unset_leaves_them_on(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("RAVEN_NO_MIRROR", raising=False)

        assert mirrors.enabled()


def _raised_from(error: BaseException, cause: BaseException) -> BaseException:
    try:
        raise error from cause
    except BaseException as raised:  # noqa: BLE001 - built only to be inspected
        return raised


class TestWhatCountsAsUnreachable:
    @pytest.mark.parametrize(
        "error",
        [
            httpx.ConnectError("connection refused"),
            httpx.ConnectTimeout("timed out"),
            httpx.ReadTimeout("timed out"),
            httpx.ReadError("connection reset by peer"),
            httpx.RemoteProtocolError("server disconnected without sending a response"),
            httpx.ProxyError("502 from the proxy"),
            ConnectionResetError(10054, "an existing connection was forcibly closed"),
            TimeoutError("timed out"),
            socket.gaierror(11001, "getaddrinfo failed"),
            ssl.SSLEOFError(8, "EOF occurred in violation of protocol"),
            urllib.error.URLError(ConnectionRefusedError(111, "connection refused")),
            urllib.error.URLError(TimeoutError("timed out")),
            OSError(errno.EHOSTUNREACH, "No route to host"),
            urllib.error.URLError(OSError(errno.ENETUNREACH, "Network is unreachable")),
        ],
    )
    def test_a_connection_that_never_completed_is_unreachable(self, error: BaseException) -> None:
        assert mirrors.unreachable(error)

    @pytest.mark.parametrize(
        ("error", "reached"),
        [
            (httpx.ProxyError("403 Forbidden"), False),
            (httpx.ProxyError("407 Proxy Authentication Required"), False),
            (urllib.error.URLError(OSError("Tunnel connection failed: 403 Forbidden")), False),
            (httpx.ProxyError("503 Service Unavailable"), True),
            (urllib.error.URLError(OSError("Tunnel connection failed: 504 Gateway Timeout")), True),
            (httpx.ProxyError("Proxy Server could not connect: Connection not allowed by ruleset."), True),
        ],
        ids=["refused", "credentials", "refused-urllib", "unavailable", "timeout-urllib", "socks"],
    )
    def test_a_proxys_refusal_of_the_tunnel_answers_for_the_host(self, error: BaseException, reached: bool) -> None:
        """A 5xx says the proxy could not reach the host either; any other
        refusal is final, the mirror being asked through the same proxy. A
        SOCKS proxy gives a reason few clients can read, and counts as not
        reached whatever it is."""
        assert mirrors.unreachable(error) is reached

    def test_the_question_needs_no_httpx(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A Raven node runs with loguru and pillow and nothing else, and starts
        agents through npx all the same."""
        monkeypatch.setitem(sys.modules, "httpx", None)

        assert mirrors.unreachable(urllib.error.URLError(ConnectionRefusedError(111, "connection refused")))
        assert not mirrors.unreachable(ValueError("not a network problem"))

    def test_the_cause_chain_is_walked(self) -> None:
        """huggingface_hub reports a Hub it could not connect to as its own
        error, raised from the transport's."""
        from huggingface_hub.errors import LocalEntryNotFoundError

        error = _raised_from(LocalEntryNotFoundError("cannot find the requested files"), httpx.ConnectTimeout("t"))

        assert mirrors.unreachable(error)

    @pytest.mark.parametrize(
        "error",
        [
            httpx.HTTPStatusError(
                "404 Not Found",
                request=httpx.Request("GET", "https://example.org/f"),
                response=httpx.Response(404, request=httpx.Request("GET", "https://example.org/f")),
            ),
            RuntimeError("401 Unauthorized"),
            httpx.UnsupportedProtocol("Request URL has an unsupported protocol 'ftp://'"),
            urllib.error.HTTPError("https://example.org/f", 404, "Not Found", None, None),
            ValueError("not a network problem"),
            OSError(errno.EMFILE, "Too many open files"),
        ],
    )
    def test_an_answer_is_not_unreachable(self, error: BaseException) -> None:
        """The host spoke. A mirror would only repeat what it said, or serve
        something it has withdrawn."""
        assert not mirrors.unreachable(error)

    def test_a_certificate_that_fails_verification_is_not_unreachable(self) -> None:
        """httpx raises a refused certificate as a ConnectError, but it is a
        question of trust: whatever re-signs this host's traffic re-signs the
        mirror's as well, and a fallback would only hide the reason."""
        refused = ssl.SSLCertVerificationError(1, "certificate verify failed: unable to get local issuer certificate")
        error = _raised_from(httpx.ConnectError("certificate verify failed"), refused)

        assert not mirrors.unreachable(error)

    def test_a_cycle_in_the_chain_ends(self) -> None:
        first, second = RuntimeError("a"), RuntimeError("b")
        first.__cause__, second.__cause__ = second, first

        assert not mirrors.unreachable(first)


class _Network:
    """Which hosts answer, and every probe that was made."""

    def __init__(self) -> None:
        self.down: set[str] = set()
        self.asked: list[str] = []

    def reachable(self, url: str) -> bool:
        self.asked.append(url)
        return url not in self.down


class _Npm:
    """What `npm config get` answers, and the directories it was asked in."""

    def __init__(self, global_prefix: str) -> None:
        self.registry: str | None = mirrors.NPM_DEFAULT_REGISTRY
        self.prefix = global_prefix
        self.asked_in: list[str | None] = []

    def config_in(self, _env: object, cwd: str | None, names: tuple[str, ...]) -> dict[str, str] | None:
        self.asked_in.append(cwd)
        if self.registry is None:
            return None
        holds = {"registry": self.registry, "prefix": self.prefix}
        return {name: holds[name] for name in names}


class TestPackageRunners:
    """uvx fetches a package the first time it is asked for, from the index its
    configuration names. When pypi.org cannot be reached and its mirror can, the
    launch names the mirror on its own command line -- the launch only: nothing
    is written to the user's configuration, and nothing stays in the
    environment of the tool it starts."""

    @pytest.fixture(autouse=True)
    def _no_opt_out(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("RAVEN_NO_MIRROR", raising=False)

    @pytest.fixture
    def network(self, monkeypatch: pytest.MonkeyPatch) -> _Network:
        network = _Network()
        monkeypatch.setattr(mirrors, "_reachable", network.reachable)
        return network

    @pytest.fixture(autouse=True)
    def _no_machine_config(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        """The machine's own /etc is no part of any case here."""
        monkeypatch.setattr(mirrors, "_ETC", str(tmp_path / "etc"))

    @pytest.fixture(autouse=True)
    def npm(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> _Npm:
        npm = _Npm(str(tmp_path / "npm-global"))
        monkeypatch.setattr(mirrors, "_npm_config", npm.config_in)
        return npm

    @staticmethod
    def _home(tmp_path) -> dict[str, str]:
        """An environment whose home and config directories are empty."""
        return {
            "HOME": str(tmp_path),
            "USERPROFILE": str(tmp_path),
            "APPDATA": str(tmp_path / "AppData" / "Roaming"),
            "XDG_CONFIG_HOME": str(tmp_path / ".config"),
            "XDG_CONFIG_DIRS": str(tmp_path / "xdg"),
            "PROGRAMDATA": str(tmp_path / "ProgramData"),
        }

    def test_uvx_resolves_against_the_mirror_named_on_its_command_line(self, network, tmp_path) -> None:
        """On the command line, not in the environment: uv keeps a flag to
        itself, while a UV_DEFAULT_INDEX would stay in the tool's environment
        and reach every `uv run` it makes in a project of the user's, which
        writes the mirror into that project's uv.lock."""
        network.down.add(mirrors.PYPI_DEFAULT_INDEX)

        assert mirrors.package_launch(["uvx", "mcp-server-fetch"], self._home(tmp_path)) == [
            "uvx",
            "--default-index",
            mirrors.PYPI_INDEX,
            "mcp-server-fetch",
        ]

    def test_uvx_is_left_alone_when_pypi_answers(self, network, tmp_path) -> None:
        assert mirrors.package_launch(["uvx", "mcp-server-fetch"], self._home(tmp_path)) == ["uvx", "mcp-server-fetch"]
        assert network.asked == [mirrors.PYPI_DEFAULT_INDEX], "the mirror is not asked about"

    def test_uvx_is_left_alone_when_the_mirror_cannot_be_reached_either(self, network, tmp_path) -> None:
        """uvx keeps a cached tool for the index it was resolved against, and
        re-resolves under any other (measured with uv 0.13): offline, the
        mirror would turn a launch that works into one that cannot."""
        network.down |= {mirrors.PYPI_DEFAULT_INDEX, mirrors.PYPI_INDEX}

        assert mirrors.package_launch(["uvx", "pkg"], self._home(tmp_path)) == ["uvx", "pkg"]

    def test_uvx_is_given_no_python_mirror(self, network, tmp_path) -> None:
        """uvx has no flag for one, and a UV_PYTHON_INSTALL_MIRROR would stay in
        the tool's environment, where it outranks the Python mirror a project
        names in its own uv.toml (measured with uv 0.13)."""
        network.down.add(mirrors.PYTHON_DEFAULT_DOWNLOADS)

        assert mirrors.package_launch(["uvx", "pkg"], self._home(tmp_path)) == ["uvx", "pkg"]
        assert mirrors.PYTHON_DEFAULT_DOWNLOADS not in network.asked

    @pytest.mark.parametrize("program", [r"C:\Users\me\.local\bin\uvx.exe", "/usr/local/bin/uvx", "UVX.EXE"])
    def test_uvx_is_recognised_wherever_it_lives(self, network, tmp_path, program: str) -> None:
        network.down.add(mirrors.PYPI_DEFAULT_INDEX)

        assert mirrors.package_launch([program, "pkg"], self._home(tmp_path)) == [
            program,
            "--default-index",
            mirrors.PYPI_INDEX,
            "pkg",
        ]

    @pytest.mark.parametrize(
        "argv",
        [
            ["uv", "run", "server.py"],
            ["uv", "--directory", "/srv/mcp", "run", "server.py"],
            ["npm", "install"],
            ["pnpm", "dlx", "pkg"],
            ["bunx", "pkg"],
        ],
    )
    def test_a_runner_that_works_on_a_project_or_reads_its_own_files_is_left_alone(
        self, network, tmp_path, argv: list[str]
    ) -> None:
        """`uv run` and `npm install` resolve the project they run in and write
        its lockfile, so a mirror in their way would land in it: one `uv run`
        with a mirror index rewrites every package's address in uv.lock. pnpm
        and bun read registries from files of their own, which this module
        cannot ask them about."""
        network.down |= {mirrors.NPM_DEFAULT_REGISTRY, mirrors.PYPI_DEFAULT_INDEX}

        assert mirrors.package_launch(argv, self._home(tmp_path)) == argv
        assert network.asked == []

    def test_a_program_that_fetches_no_packages_is_never_probed(self, network, tmp_path) -> None:
        """Most agents are a CLI of their own, and asking a registry before
        starting one would only delay it."""
        assert mirrors.package_launch(["codex", "exec", "--json"], self._home(tmp_path)) == ["codex", "exec", "--json"]
        assert network.asked == []

    @pytest.mark.parametrize(
        "args",
        [
            ["--index-url", "https://pypi.example.internal/simple", "pkg"],
            ["--default-index=https://pypi.example.internal/simple", "pkg"],
            ["-i", "https://pypi.example.internal/simple", "pkg"],
            ["--no-index", "--find-links", "/wheels", "pkg"],
        ],
    )
    def test_an_index_on_the_command_line_wins(self, network, tmp_path, args: list[str]) -> None:
        """A second --default-index would also be one uv refuses to start with."""
        network.down.add(mirrors.PYPI_DEFAULT_INDEX)

        assert mirrors.package_launch(["uvx", *args], self._home(tmp_path)) == ["uvx", *args]

    @pytest.mark.parametrize(
        "args",
        [["--config-file", "/srv/uv.toml", "pkg"], ["--config-file=/srv/uv.toml", "pkg"]],
        ids=["config-file", "config-file="],
    )
    def test_a_uv_toml_named_on_the_command_line_wins(self, network, tmp_path, args: list[str]) -> None:
        """That file may name an index, and the flag added here would outrank it."""
        network.down.add(mirrors.PYPI_DEFAULT_INDEX)

        assert mirrors.package_launch(["uvx", *args], self._home(tmp_path)) == ["uvx", *args]
        assert network.asked == []

    @pytest.mark.parametrize("name", ["UV_DEFAULT_INDEX", "UV_INDEX_URL", "UV_INDEX"])
    def test_an_index_already_in_the_environment_wins(self, network, tmp_path, name: str) -> None:
        network.down.add(mirrors.PYPI_DEFAULT_INDEX)
        env = {**self._home(tmp_path), name: "https://pypi.example.internal/simple"}

        assert mirrors.package_launch(["uvx", "pkg"], env) == ["uvx", "pkg"]

    @staticmethod
    def _uv_toml(tmp_path, text: str, *, system: bool = False) -> None:
        """Write a uv.toml where uv reads it on this platform: the user's, or the machine's."""
        if os.name == "nt":
            root = tmp_path / ("ProgramData" if system else "AppData/Roaming")
        else:
            root = tmp_path / ("xdg" if system else ".config")
        config = root / "uv" / "uv.toml"
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text(text, encoding="utf-8")

    @pytest.mark.parametrize("system", [False, True], ids=["users", "machines"])
    @pytest.mark.parametrize(
        "text",
        [
            'index-url = "https://pypi.example.internal/simple"\n',
            '[[index]]\nurl = "https://pypi.example.internal/simple"\ndefault = true\n',
            'index = [{ url = "https://pypi.example.internal/simple", default = true }]\n',
        ],
        ids=["index-url", "default-index", "inline-default-index"],
    )
    def test_an_index_in_uv_toml_wins(self, network, tmp_path, text: str, system: bool) -> None:
        network.down.add(mirrors.PYPI_DEFAULT_INDEX)
        self._uv_toml(tmp_path, text, system=system)

        assert mirrors.package_launch(["uvx", "pkg"], self._home(tmp_path)) == ["uvx", "pkg"]

    @pytest.mark.parametrize("default_in", ["machines", "users"])
    def test_each_uv_toml_is_read_on_its_own(self, network, tmp_path, default_in: str) -> None:
        """One file's list of indexes does not hide the other's default one."""
        network.down.add(mirrors.PYPI_DEFAULT_INDEX)
        extra = '[[index]]\nurl = "https://extra.example.internal/simple"\n'
        default = '[[index]]\nurl = "https://pypi.example.internal/simple"\ndefault = true\n'
        self._uv_toml(tmp_path, default if default_in == "users" else extra)
        self._uv_toml(tmp_path, default if default_in == "machines" else extra, system=True)

        assert mirrors.package_launch(["uvx", "pkg"], self._home(tmp_path)) == ["uvx", "pkg"]

    def test_a_uv_toml_named_by_the_environment_is_the_only_one_read(self, network, tmp_path) -> None:
        """uv reads that file and no other, the user's and the machine's
        included (measured with uv 0.13)."""
        network.down.add(mirrors.PYPI_DEFAULT_INDEX)
        named = tmp_path / "named.toml"
        named.write_text('index-url = "https://pypi.example.internal/simple"\n', encoding="utf-8")
        env = {**self._home(tmp_path), "UV_CONFIG_FILE": str(named)}

        assert mirrors.package_launch(["uvx", "pkg"], env) == ["uvx", "pkg"], "the named file chose"

        named.write_text('python-preference = "managed"\n', encoding="utf-8")
        self._uv_toml(tmp_path, 'index-url = "https://pypi.example.internal/simple"\n')

        assert mirrors.package_launch(["uvx", "pkg"], env) == [
            "uvx",
            "--default-index",
            mirrors.PYPI_INDEX,
            "pkg",
        ], "the user's own file is not read while another is named"

    def test_an_extra_index_leaves_pypi_the_default(self, network, tmp_path) -> None:
        network.down.add(mirrors.PYPI_DEFAULT_INDEX)
        self._uv_toml(tmp_path, '[[index]]\nurl = "https://pypi.example.internal/simple"\n')

        assert mirrors.package_launch(["uvx", "pkg"], self._home(tmp_path)) == [
            "uvx",
            "--default-index",
            mirrors.PYPI_INDEX,
            "pkg",
        ]

    def test_ravens_own_npm_install_is_pointed_at_the_mirror(self, network, npm, tmp_path) -> None:
        """A bridge raven builds in a directory of its own, so the lockfile npm
        writes there is raven's, and the npm it runs starts nothing that stays."""
        staging = str(tmp_path / "staging")
        assert mirrors.npm_registry_env(self._home(tmp_path), staging) == {}

        network.down.add(mirrors.NPM_DEFAULT_REGISTRY)
        assert mirrors.npm_registry_env(self._home(tmp_path), staging) == {"npm_config_registry": mirrors.NPM_REGISTRY}
        assert npm.asked_in == [staging]

    def test_ravens_own_npm_install_is_left_alone_when_npm_answers(self, network, npm, tmp_path) -> None:
        assert mirrors.npm_registry_env(self._home(tmp_path), str(tmp_path)) == {}
        assert network.asked == [mirrors.NPM_DEFAULT_REGISTRY], "the mirror is not asked about"
        assert npm.asked_in == [], "nor is npm, on a network that works"

    def test_ravens_own_npm_install_is_left_alone_when_the_mirror_cannot_be_reached_either(
        self, network, tmp_path
    ) -> None:
        network.down |= {mirrors.NPM_DEFAULT_REGISTRY, mirrors.NPM_REGISTRY}

        assert mirrors.npm_registry_env(self._home(tmp_path), str(tmp_path)) == {}

    @pytest.mark.parametrize("name", ["npm_config_registry", "NPM_CONFIG_REGISTRY"])
    def test_a_registry_already_in_the_environment_wins(self, network, tmp_path, name: str) -> None:
        network.down.add(mirrors.NPM_DEFAULT_REGISTRY)
        env = {**self._home(tmp_path), name: "https://registry.example.internal"}

        assert mirrors.npm_registry_env(env, str(tmp_path)) == {}
        assert network.asked == [], "a choice already made is not second-guessed"

    def test_a_registry_in_the_users_npmrc_wins(self, network, tmp_path) -> None:
        network.down.add(mirrors.NPM_DEFAULT_REGISTRY)
        (tmp_path / ".npmrc").write_text("fund=false\nregistry=https://registry.example.internal/\n", encoding="utf-8")

        assert mirrors.npm_registry_env(self._home(tmp_path), str(tmp_path)) == {}

    def test_a_scoped_registry_is_not_a_choice_of_default(self, network, tmp_path) -> None:
        network.down.add(mirrors.NPM_DEFAULT_REGISTRY)
        (tmp_path / ".npmrc").write_text("@corp:registry=https://registry.example.internal/\n", encoding="utf-8")

        assert mirrors.npm_registry_env(self._home(tmp_path), str(tmp_path)) == {
            "npm_config_registry": mirrors.NPM_REGISTRY
        }

    def test_a_registry_npm_reads_from_anywhere_else_wins(self, network, npm, tmp_path) -> None:
        """A project's .npmrc and npm's global one name registries too, and the
        variable this module would set outranks both (measured with npm 11.12),
        so npm itself is asked what it would use, from where it runs."""
        network.down.add(mirrors.NPM_DEFAULT_REGISTRY)
        npm.registry = "https://registry.example.internal/"
        project = str(tmp_path / "project")

        assert mirrors.npm_registry_env(self._home(tmp_path), project) == {}
        assert npm.asked_in == [project]

    def test_an_npm_that_cannot_say_is_left_alone(self, network, npm, tmp_path) -> None:
        network.down.add(mirrors.NPM_DEFAULT_REGISTRY)
        npm.registry = None

        assert mirrors.npm_registry_env(self._home(tmp_path), str(tmp_path)) == {}

    def test_the_opt_out_leaves_every_runner_alone(self, network, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
        network.down |= {mirrors.NPM_DEFAULT_REGISTRY, mirrors.PYPI_DEFAULT_INDEX}
        monkeypatch.setenv("RAVEN_NO_MIRROR", "1")

        assert mirrors.package_launch(["uvx", "pkg"], self._home(tmp_path)) == ["uvx", "pkg"]
        assert mirrors.package_launch(["npx", "-y", "pkg"], self._home(tmp_path)) == ["npx", "-y", "pkg"]
        assert mirrors.npm_registry_env(self._home(tmp_path)) == {}
        assert network.asked == []

    def test_a_fault_while_deciding_leaves_every_launch_as_configured(
        self, network, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The mirror is a way around an outage, never a gate: whatever goes
        wrong while deciding, the launch goes ahead as it was configured."""

        def broken(*_args: object, **_kwargs: object) -> object:
            raise RuntimeError("a bug of ours, not an unreachable host")

        monkeypatch.setattr(mirrors, "_moved", broken)

        assert mirrors.package_launch(["uvx", "pkg"], self._home(tmp_path)) == ["uvx", "pkg"]
        assert mirrors.npm_registry_env(self._home(tmp_path)) == {}

    async def test_the_async_form_answers_the_same(self, network, tmp_path) -> None:
        network.down.add(mirrors.PYPI_DEFAULT_INDEX)

        assert await mirrors.package_launch_async(["uvx", "pkg"], self._home(tmp_path)) == [
            "uvx",
            "--default-index",
            mirrors.PYPI_INDEX,
            "pkg",
        ]


#: Two packages as a registry would serve them: each one's version and bins,
#: and the files those bins are.
_PACKAGES = {
    "@acme/agent": {
        "version": "1.4.0",
        "bin": {"acme-agent": "dist/cli.js"},
        "files": {"dist/cli.js": "#!/usr/bin/env node\nconsole.log('agent')\n"},
    },
    "acme-tools": {
        "version": "2.0.1",
        "bin": {"acme-lint": "bin/lint.js", "acme-fmt": "bin/fmt.js"},
        "files": {
            "bin/lint.js": "#!/usr/bin/env node\nconsole.log('lint')\n",
            "bin/fmt.js": "#!/usr/bin/env node\nconsole.log('fmt')\n",
        },
    },
}


class _Installs:
    """Stands in for npm in Raven's install directory: every run it is asked
    for, and the packages a run lays out there."""

    def __init__(self) -> None:
        self.runs: list[tuple[Path, str]] = []
        self.fails = False
        self.executable = True
        self.running = 0
        self.overlapped = False

    def __call__(self, prefix: Path, verb: str, _env: object) -> bool:
        self.runs.append((prefix, verb))
        self.running += 1
        self.overlapped |= self.running > 1
        try:
            time.sleep(0.05)
            if self.fails:
                return False
            for name in json.loads((prefix / "package.json").read_text(encoding="utf-8"))["dependencies"]:
                published = _PACKAGES[name]
                package = prefix / "node_modules" / name
                package.mkdir(parents=True, exist_ok=True)
                manifest = {"name": name, "version": published["version"], "bin": published["bin"]}
                (package / "package.json").write_text(json.dumps(manifest), encoding="utf-8")
                for relative, text in published["files"].items():
                    (package / relative).parent.mkdir(parents=True, exist_ok=True)
                    (package / relative).write_text(text, encoding="utf-8")
                    (package / relative).chmod(0o755 if self.executable else 0o644)
            return True
        finally:
            self.running -= 1


class TestNpxLaunches:
    """npx cannot be pointed at a mirror without what it starts keeping the
    mirror (see package_launch). So when registry.npmjs.org cannot be reached
    and its mirror can, npx does not run: the packages it would fetch are
    installed from the mirror into a directory of Raven's own, and the bin it
    would have run starts from there, with the launch's own environment."""

    @pytest.fixture(autouse=True)
    def _setting(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        monkeypatch.delenv("RAVEN_NO_MIRROR", raising=False)
        monkeypatch.setenv("RAVEN_HOME", str(tmp_path / "raven"))

    @pytest.fixture
    def network(self, monkeypatch: pytest.MonkeyPatch) -> _Network:
        network = _Network()
        network.down.add(mirrors.NPM_DEFAULT_REGISTRY)
        monkeypatch.setattr(mirrors, "_reachable", network.reachable)
        return network

    @pytest.fixture(autouse=True)
    def npm(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> _Npm:
        npm = _Npm(str(tmp_path / "npm-global"))
        monkeypatch.setattr(mirrors, "_npm_config", npm.config_in)
        return npm

    @pytest.fixture(autouse=True)
    def installs(self, monkeypatch: pytest.MonkeyPatch) -> _Installs:
        installs = _Installs()
        monkeypatch.setattr(mirrors, "_npm_install", installs)
        return installs

    @pytest.fixture
    def project(self, tmp_path) -> Path:
        project = tmp_path / "work" / "project"
        project.mkdir(parents=True)
        return project

    @staticmethod
    def _env(tmp_path) -> dict[str, str]:
        """An empty home, and a PATH holding a node: Windows starts a bin through one."""
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir(exist_ok=True)
        (bin_dir / "node.exe").write_bytes(b"")
        return {"HOME": str(tmp_path / "home"), "USERPROFILE": str(tmp_path / "home"), "PATH": str(bin_dir)}

    @staticmethod
    def _installed(tmp_path, specs: list[str], name: str, relative: str, args: list[str]) -> list[str]:
        """How a bin of Raven's install of ``specs`` starts, on this platform."""
        key = hashlib.sha256("\n".join(sorted(specs)).encode("utf-8")).hexdigest()[:16]
        target = str((tmp_path / "raven" / "cache" / "npx" / key / "node_modules" / name / relative).resolve())
        if os.name != "nt":
            return [target, *args]
        return [shutil.which("node", path=str(tmp_path / "bin")) or "", target, *args]

    def test_the_package_starts_from_ravens_own_install_of_it(self, network, installs, project, tmp_path) -> None:
        env = self._env(tmp_path)
        before = dict(env)

        argv = mirrors.package_launch(["npx", "-y", "@acme/agent", "--acp"], env, str(project))

        assert argv == self._installed(tmp_path, ["@acme/agent"], "@acme/agent", "dist/cli.js", ["--acp"])
        assert [verb for _prefix, verb in installs.runs] == ["install"]
        assert env == before, "nothing of the mirror reaches the launch's environment"

    @pytest.mark.parametrize("program", ["npx", r"C:\Program Files\nodejs\npx.cmd", "/usr/local/bin/npx"])
    def test_npx_is_recognised_wherever_it_lives(self, network, project, tmp_path, program: str) -> None:
        argv = mirrors.package_launch([program, "-y", "@acme/agent"], self._env(tmp_path), str(project))

        assert argv == self._installed(tmp_path, ["@acme/agent"], "@acme/agent", "dist/cli.js", [])

    def test_npx_is_left_alone_when_the_registry_answers(self, network, npm, installs, project, tmp_path) -> None:
        network.down.clear()
        argv = ["npx", "-y", "@acme/agent"]

        assert mirrors.package_launch(argv, self._env(tmp_path), str(project)) == argv
        assert network.asked == [mirrors.NPM_DEFAULT_REGISTRY], "the mirror is not asked about"
        assert npm.asked_in == [] and installs.runs == [], "nor is npm, on a network that works"

    def test_npx_is_left_alone_when_the_mirror_cannot_be_reached_either(
        self, network, installs, project, tmp_path
    ) -> None:
        """npx starts a package it fetched before out of its own cache when
        the registry is down; a mirror that is down too would only stop it."""
        network.down.add(mirrors.NPM_REGISTRY)
        argv = ["npx", "-y", "@acme/agent"]

        assert mirrors.package_launch(argv, self._env(tmp_path), str(project)) == argv
        assert installs.runs == []

    @pytest.mark.parametrize(
        ("words", "specs", "name", "relative", "args"),
        [
            (["@acme/agent"], ["@acme/agent"], "@acme/agent", "dist/cli.js", []),
            (["--yes", "--", "@acme/agent", "--flag"], ["@acme/agent"], "@acme/agent", "dist/cli.js", ["--flag"]),
            (["@acme/agent@1.4.0", "-y"], ["@acme/agent@1.4.0"], "@acme/agent", "dist/cli.js", ["-y"]),
            (["-p", "acme-tools", "acme-fmt", "src"], ["acme-tools"], "acme-tools", "bin/fmt.js", ["src"]),
            (
                ["--package=@acme/agent", "-p=acme-tools", "acme-lint"],
                ["@acme/agent", "acme-tools"],
                "acme-tools",
                "bin/lint.js",
                [],
            ),
        ],
        ids=["bare", "yes-and-dashes", "pinned", "package-and-bin", "two-packages"],
    )
    def test_each_shape_npx_takes_starts_the_bin_npx_would_run(
        self, network, project, tmp_path, words, specs, name, relative, args
    ) -> None:
        """Without -p the package's own bin runs, with -p the one named, and
        whatever follows the first word is that bin's own."""
        argv = mirrors.package_launch(["npx", *words], self._env(tmp_path), str(project))

        assert argv == self._installed(tmp_path, specs, name, relative, args)

    @pytest.mark.parametrize(
        "words",
        [
            ["--registry", "https://registry.example.internal/", "@acme/agent"],
            ["--no", "@acme/agent"],
            ["-c", "acme-agent --acp"],
            ["--cache", "/srv/npm-cache", "@acme/agent"],
            ["-q", "@acme/agent"],
            ["github:acme/agent"],
            ["./agent"],
            ["https://example.org/agent.tgz"],
            ["agent@npm:@acme/agent@1"],
            ["-p", "acme-tools", "-p", "acme-tools@2", "acme-fmt"],
            ["-p", "acme-tools", "./bin/acme-fmt"],
            ["-y"],
            [],
        ],
        ids=[
            "registry",
            "no",
            "call",
            "cache",
            "quiet",
            "git",
            "path",
            "tarball",
            "alias",
            "one-package-twice",
            "a-path-to-run",
            "no-package",
            "bare-npx",
        ],
    )
    def test_any_other_shape_starts_as_configured(self, network, installs, project, tmp_path, words) -> None:
        """An option is a choice of how npx works, which a launch without npx
        would drop, and a spec that is not a registry's is not fetched from one."""
        argv = ["npx", *words]

        assert mirrors.package_launch(argv, self._env(tmp_path), str(project)) == argv
        assert installs.runs == []

    def test_an_exact_version_is_installed_once(self, network, installs, project, tmp_path) -> None:
        argv = ["npx", "-y", "@acme/agent@1.4.0"]
        first = mirrors.package_launch(argv, self._env(tmp_path), str(project))
        again = mirrors.package_launch(argv, self._env(tmp_path), str(project))

        assert first == again != argv
        assert [verb for _prefix, verb in installs.runs] == ["install"]

    def test_anything_else_is_brought_up_to_date_at_each_launch(self, network, installs, project, tmp_path) -> None:
        """As npx asks the registry each time what a tag or a range means now."""
        for _ in range(2):
            mirrors.package_launch(["npx", "-y", "@acme/agent@^1"], self._env(tmp_path), str(project))

        assert [verb for _prefix, verb in installs.runs] == ["install", "update"]

    def test_a_failed_update_starts_the_install_made_before(self, network, installs, project, tmp_path) -> None:
        argv = ["npx", "-y", "@acme/agent"]
        first = mirrors.package_launch(argv, self._env(tmp_path), str(project))
        installs.fails = True

        assert mirrors.package_launch(argv, self._env(tmp_path), str(project)) == first != argv

    def test_a_failed_install_starts_npx_as_configured(self, network, installs, project, tmp_path) -> None:
        installs.fails = True
        argv = ["npx", "-y", "@acme/agent"]

        assert mirrors.package_launch(argv, self._env(tmp_path), str(project)) == argv

    def test_a_bin_npx_could_not_pick_starts_npx_as_configured(self, network, project, tmp_path) -> None:
        """acme-tools has two bins and neither is named after it: npx would ask
        which, and so would this."""
        argv = ["npx", "-y", "acme-tools"]

        assert mirrors.package_launch(argv, self._env(tmp_path), str(project)) == argv

    def test_launches_of_the_same_packages_take_turns(self, network, installs, project, tmp_path) -> None:
        argv = ["npx", "-y", "@acme/agent"]
        started: list[list[str]] = []
        workers = [
            threading.Thread(
                target=lambda: started.append(mirrors.package_launch(argv, self._env(tmp_path), str(project)))
            )
            for _ in range(4)
        ]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()

        assert len(started) == 4 and all(argv_ == started[0] != argv for argv_ in started)
        assert not installs.overlapped, "two npm runs in one directory at once"

    @pytest.mark.parametrize("where", ["the-project", "a-folder-above"], ids=["in-the-project", "above-the-project"])
    @pytest.mark.parametrize("has", ["bin", "package", "own-bin"])
    def test_what_the_project_has_stays_npxs_to_start(
        self, network, npm, installs, project, tmp_path, where: str, has: str
    ) -> None:
        """npx runs a bin or a package the project already has without asking a
        registry; that is not a launch that needs a mirror."""
        folder = project if where == "the-project" else project.parent
        if has == "bin":
            (folder / "node_modules" / ".bin").mkdir(parents=True)
            (folder / "node_modules" / ".bin" / "acme-agent").write_text("", encoding="utf-8")
        elif has == "package":
            (folder / "node_modules" / "@acme" / "agent").mkdir(parents=True)
            (folder / "node_modules" / "@acme" / "agent" / "package.json").write_text("{}", encoding="utf-8")
        else:
            manifest = {"name": "acme-workspace", "bin": {"acme-agent": "cli.js"}}
            (folder / "package.json").write_text(json.dumps(manifest), encoding="utf-8")
        spec = "acme-agent" if has != "package" else "@acme/agent"
        argv = ["npx", "-y", spec]

        assert mirrors.package_launch(argv, self._env(tmp_path), str(project)) == argv
        assert network.asked == [] and npm.asked_in == [] and installs.runs == []

    def test_a_bin_npm_installed_globally_stays_npxs_to_start(self, network, npm, installs, project, tmp_path) -> None:
        bin_dir = Path(npm.prefix) if os.name == "nt" else Path(npm.prefix) / "bin"
        bin_dir.mkdir(parents=True)
        (bin_dir / "acme-agent").write_text("", encoding="utf-8")
        argv = ["npx", "-y", "acme-agent"]

        assert mirrors.package_launch(argv, self._env(tmp_path), str(project)) == argv
        assert installs.runs == []

    @pytest.mark.parametrize(
        "npmrc", ["@acme:registry=https://registry.example.internal/\n", "registry=https://registry.npmjs.org/\n"]
    )
    @pytest.mark.parametrize("where", ["the-launch-folder", "the-project-root-above-it"])
    def test_a_registry_the_projects_npmrc_names_keeps_npx(
        self, network, installs, project, tmp_path, npmrc: str, where: str
    ) -> None:
        """npx reads the project's .npmrc; Raven's install directory does not,
        and would fetch from the mirror what the project gets elsewhere."""
        if where == "the-launch-folder":
            (project / ".npmrc").write_text(npmrc, encoding="utf-8")
        else:
            (project.parent / "package.json").write_text("{}", encoding="utf-8")
            (project.parent / ".npmrc").write_text(npmrc, encoding="utf-8")
        argv = ["npx", "-y", "@acme/agent"]

        assert mirrors.package_launch(argv, self._env(tmp_path), str(project)) == argv
        assert installs.runs == []

    def test_an_npmrc_npm_reads_for_no_project_is_passed_over(self, network, project, tmp_path) -> None:
        """Above the project's root, in a folder that is no project's, npm reads
        no .npmrc either."""
        (project / "package.json").write_text("{}", encoding="utf-8")
        (project.parent / ".npmrc").write_text("@acme:registry=https://registry.example.internal/\n")
        argv = ["npx", "-y", "@acme/agent"]

        assert mirrors.package_launch(argv, self._env(tmp_path), str(project)) != argv

    def test_the_users_own_npmrc_is_no_projects(self, network, installs, tmp_path) -> None:
        """Raven's install reads it as npx would, so a registry it names is kept
        either way -- even with the home folder holding a node_modules, which
        makes npm take it for a project's root, and npm skips it there too."""
        env = self._env(tmp_path)
        home = Path(env["HOME"])
        (home / "node_modules").mkdir(parents=True)
        (home / ".npmrc").write_text("@acme:registry=https://registry.example.internal/\n", encoding="utf-8")
        project = home / "project"
        project.mkdir()
        argv = ["npx", "-y", "@acme/agent"]

        assert mirrors.package_launch(argv, env, str(project)) != argv

    def test_a_bin_that_points_out_of_its_package_starts_npx_as_configured(
        self, network, project, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        outside = {"bin": {"acme-agent": "../../escape.js"}, "files": {"../../escape.js": "console.log('out')\n"}}
        monkeypatch.setitem(_PACKAGES, "@acme/agent", {**_PACKAGES["@acme/agent"], **outside})
        argv = ["npx", "-y", "@acme/agent"]

        assert mirrors.package_launch(argv, self._env(tmp_path), str(project)) == argv

    @pytest.mark.skipif(os.name == "nt", reason="Windows starts a bin through the program its first line names")
    def test_a_bin_npm_did_not_mark_executable_starts_npx_as_configured(
        self, network, installs, project, tmp_path
    ) -> None:
        installs.executable = False
        argv = ["npx", "-y", "@acme/agent"]

        assert mirrors.package_launch(argv, self._env(tmp_path), str(project)) == argv

    @pytest.mark.parametrize("chosen", ["environment", "users-npmrc", "npm-elsewhere", "npm-cannot-say"])
    def test_a_registry_already_chosen_keeps_npx(self, network, npm, installs, project, tmp_path, chosen: str) -> None:
        env = self._env(tmp_path)
        if chosen == "environment":
            env["npm_config_registry"] = "https://registry.example.internal/"
        elif chosen == "users-npmrc":
            Path(env["HOME"]).mkdir()
            (Path(env["HOME"]) / ".npmrc").write_text("registry=https://registry.example.internal/\n")
        elif chosen == "npm-elsewhere":
            npm.registry = "https://registry.example.internal/"
        else:
            npm.registry = None
        argv = ["npx", "-y", "@acme/agent"]

        assert mirrors.package_launch(argv, env, str(project)) == argv
        assert installs.runs == []

    def test_npm_is_asked_where_the_launch_runs(self, network, npm, project, tmp_path) -> None:
        mirrors.package_launch(["npx", "-y", "@acme/agent"], self._env(tmp_path), str(project))

        assert npm.asked_in == [str(project)]

    def test_a_launch_that_runs_in_a_sandbox_is_left_alone(self, network, installs, project, tmp_path) -> None:
        """The install would be made on this machine, and the sandbox starts
        what it has in there."""
        argv = ["npx", "-y", "@acme/agent"]

        assert mirrors.package_launch(argv, self._env(tmp_path), str(project), local=False) == argv
        assert installs.runs == []

    async def test_the_async_form_answers_the_same(self, network, project, tmp_path) -> None:
        argv = await mirrors.package_launch_async(["npx", "-y", "@acme/agent"], self._env(tmp_path), str(project))

        assert argv == self._installed(tmp_path, ["@acme/agent"], "@acme/agent", "dist/cli.js", [])


class TestTheBinNpxWouldRun:
    @pytest.mark.parametrize(
        ("bins", "picked"),
        [
            ({"one": "cli.js"}, "one"),
            ({"one": "cli.js", "alias": "cli.js"}, "one"),
            ({"agent": "a.js", "other": "b.js"}, "agent"),
            ({"x": "a.js", "y": "b.js"}, None),
            ({}, None),
        ],
        ids=["only-one", "aliases", "named-after-it", "cannot-tell", "none"],
    )
    def test_without_p_the_package_names_it(self, bins: dict[str, str], picked: str | None) -> None:
        assert mirrors._default_bin("@acme/agent", bins) == picked

    def test_a_bin_given_as_a_path_is_named_after_the_package(self) -> None:
        assert mirrors._bins({"bin": "cli.js"}, "@acme/agent") == {"agent": "cli.js"}

    @pytest.mark.parametrize(
        ("line", "options"),
        [
            ("#!/usr/bin/env node", []),
            ("#!/usr/bin/env -S node --no-warnings", ["--no-warnings"]),
            ("#!/usr/local/bin/node --harmony\r", ["--harmony"]),
            ("#!node", []),
            ("#!/usr/bin/env NODE_ENV=production node", None),
            ("#!/bin/sh", None),
            ("#!/usr/bin/env python3", None),
            ("console.log('no first line to read')", None),
        ],
        ids=["env", "env-S", "absolute", "bare", "variables", "sh", "python", "none"],
    )
    def test_windows_starts_a_bin_through_the_program_its_first_line_names(
        self, line: str, options: list[str] | None
    ) -> None:
        """As npm's own shims do there (cmd-shim): node and the options it is
        given, and nothing a .cmd would set up first, such as variables."""
        assert mirrors._node_options(line) == options


def _ended(pid: int) -> bool:
    """Whether ``pid`` is gone, or a zombie no one has reaped yet: either way it runs nothing."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    try:
        return Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()[0] == "Z"
    except OSError:
        return False


@pytest.mark.skipif(os.name == "nt", reason="a POSIX process group; Windows ends the tree with taskkill")
def test_an_npm_past_its_deadline_is_ended_with_what_it_started(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """npm runs install scripts of its own; one that hangs must not outlive the
    install it belongs to, still writing into the directory the next launch
    installs into."""
    monkeypatch.setattr(mirrors, "_NPX_INSTALL_TIMEOUT", 0.5)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    started = tmp_path / "started.pid"
    (bin_dir / "npm").write_text(f"#!/bin/sh\nsleep 30 &\necho $! > '{started}'\nwait\n", encoding="utf-8")
    (bin_dir / "npm").chmod(0o755)

    began = time.monotonic()
    assert not mirrors._npm_install(tmp_path, "install", {"PATH": f"{bin_dir}:/usr/bin:/bin"})
    assert time.monotonic() - began < 10, "the deadline waited for what npm started, which holds its output open"
    child = int(started.read_text(encoding="utf-8"))
    deadline = time.monotonic() + 5
    while not _ended(child) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert _ended(child), "the script npm started is still running"


class TestTheInstallDirectoryLock:
    def test_a_second_holder_waits_for_the_first(self, tmp_path) -> None:
        lock = tmp_path / "npx" / ".lock"
        order: list[str] = []

        def wait() -> None:
            with mirrors._exclusive(lock, 5):
                order.append("second")

        with mirrors._exclusive(lock, 5):
            waiter = threading.Thread(target=wait)
            waiter.start()
            time.sleep(0.3)
            order.append("first done")
        waiter.join(5)

        assert order == ["first done", "second"]

    def test_a_holder_that_never_lets_go_is_given_up_on(self, tmp_path) -> None:
        lock = tmp_path / "npx" / ".lock"
        with mirrors._exclusive(lock, 5):
            failed: list[BaseException] = []

            def wait() -> None:
                try:
                    with mirrors._exclusive(lock, 0.3):
                        pass
                except TimeoutError as exc:
                    failed.append(exc)

            waiter = threading.Thread(target=wait)
            waiter.start()
            waiter.join(5)

        assert failed


class _Registry(http.server.BaseHTTPRequestHandler):
    """An npm registry serving one package whose bin reports what it was given:
    its arguments, and the registry npm left in its environment."""

    published: dict[str, bytes] = {}

    def do_GET(self) -> None:  # noqa: N802 - the name http.server dispatches on
        body = self.published.get(self.path.split("?", 1)[0])
        if body is None:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header(
            "Content-Type", "application/json" if not self.path.endswith(".tgz") else "application/octet-stream"
        )
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: object) -> None:
        return None


@pytest.fixture
def npm_registry() -> Iterator[str]:
    import base64
    import io
    import tarfile

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Registry)
    origin = f"http://127.0.0.1:{server.server_address[1]}"
    manifest = {"name": "raven-npx-probe", "version": "1.0.0", "bin": {"raven-npx-probe": "cli.js"}}
    cli = (
        "#!/usr/bin/env node\n"
        "console.log(JSON.stringify({args: process.argv.slice(2), registry: process.env.npm_config_registry || null}))\n"
    )
    packed = io.BytesIO()
    with tarfile.open(fileobj=packed, mode="w:gz") as tar:
        for name, text in (("package/package.json", json.dumps(manifest)), ("package/cli.js", cli)):
            data = text.encode("utf-8")
            entry = tarfile.TarInfo(name)
            entry.size, entry.mode, entry.mtime = len(data), 0o755, int(time.time())
            tar.addfile(entry, io.BytesIO(data))
    tarball = packed.getvalue()
    dist = {
        "tarball": f"{origin}/raven-npx-probe/-/raven-npx-probe-1.0.0.tgz",
        "integrity": "sha512-" + base64.b64encode(hashlib.sha512(tarball).digest()).decode("ascii"),
        "shasum": hashlib.sha1(tarball).hexdigest(),
    }
    packument = {
        "name": "raven-npx-probe",
        "dist-tags": {"latest": "1.0.0"},
        "versions": {"1.0.0": {**manifest, "dist": dist}},
    }
    _Registry.published = {
        "/raven-npx-probe": json.dumps(packument).encode("utf-8"),
        "/raven-npx-probe/-/raven-npx-probe-1.0.0.tgz": tarball,
    }
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield origin
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("npm") is None or shutil.which("node") is None, reason="needs node and npm")
def test_an_npx_launch_starts_from_the_mirror_with_nothing_of_it_left_behind(
    npm_registry: str, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real npm, against a registry on 127.0.0.1 standing in for the mirror:
    the bin starts, takes its arguments, and finds no registry in its
    environment; the lockfile Raven's install wrote names the mirror."""
    monkeypatch.setattr(mirrors, "NPM_REGISTRY", npm_registry)
    monkeypatch.setattr(mirrors, "_reachable", lambda url: url != mirrors.NPM_DEFAULT_REGISTRY)
    monkeypatch.delenv("RAVEN_NO_MIRROR", raising=False)
    monkeypatch.setenv("RAVEN_HOME", str(tmp_path / "raven"))
    home = tmp_path / "home"
    home.mkdir()
    proxies = {"http_proxy", "https_proxy", "all_proxy", "no_proxy", "npm_config_registry"}
    env = {key: value for key, value in os.environ.items() if key.lower() not in proxies}
    env |= {
        "HOME": str(home),
        "USERPROFILE": str(home),
        "APPDATA": str(home / "AppData" / "Roaming"),
        "LOCALAPPDATA": str(home / "AppData" / "Local"),
        "npm_config_cache": str(tmp_path / "npm-cache"),
        "NO_PROXY": "127.0.0.1,localhost",
    }
    project = tmp_path / "project"
    project.mkdir()

    argv = mirrors.package_launch(["npx", "-y", "raven-npx-probe", "one", "two words"], env, str(project))

    assert argv[0] != "npx", "the launch was left as configured"
    started = subprocess.run(argv, env=env, cwd=project, capture_output=True, text=True, timeout=60, check=False)
    assert started.returncode == 0, started.stderr
    assert json.loads(started.stdout) == {"args": ["one", "two words"], "registry": None}
    key = hashlib.sha256(b"raven-npx-probe").hexdigest()[:16]
    lock = json.loads((tmp_path / "raven" / "cache" / "npx" / key / "package-lock.json").read_text(encoding="utf-8"))
    assert lock["packages"]["node_modules/raven-npx-probe"]["resolved"].startswith(npm_registry)


class TestAskingNpm:
    """npm's own answer to which registry it would use, from a real process:
    the environment, the directory, and the failures that mean "cannot say"."""

    @staticmethod
    def _fake_npm(tmp_path, *, exit_code: int = 0, hang: bool = False):
        """An npm that answers `config get registry` with the file NPM_SAYS names,
        read from where it runs; one given an ``exit_code`` prints npm's default
        and then fails with it, and one that hangs prints npm's default only
        after a wait no launch should sit through.

        Shell builtins only, beside an absolute path or two: the PATH it runs
        on holds nothing but this npm.
        """
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        if os.name == "nt":
            wait = '"%SYSTEMROOT%\\System32\\ping.exe" -n 2 127.0.0.1 >nul\r\n' if hang else ""
            (bin_dir / "npm.cmd").write_text(
                "@echo off\r\n"
                'if not "%1 %2 %3"=="config get registry" exit /b 9\r\n'
                f"{wait}"
                f"if {exit_code} neq 0 echo https://registry.npmjs.org/\r\n"
                f"if {exit_code} neq 0 exit /b {exit_code}\r\n"
                'if defined NPM_SAYS type "%NPM_SAYS%"\r\n'
                "if not defined NPM_SAYS echo https://registry.npmjs.org/\r\n",
                encoding="ascii",
            )
        else:
            script = bin_dir / "npm"
            script.write_text(
                "#!/bin/sh\n"
                '[ "$1 $2 $3" = "config get registry" ] || exit 9\n'
                + ("/bin/sleep 1\n" if hang else "")
                + f"[ {exit_code} -eq 0 ] || {{ echo https://registry.npmjs.org/; exit {exit_code}; }}\n"
                '[ -n "${NPM_SAYS:-}" ] || { echo https://registry.npmjs.org/; exit 0; }\n'
                'while IFS= read -r line || [ -n "$line" ]; do printf \'%s\\n\' "$line"; done < "$NPM_SAYS"\n',
                encoding="ascii",
            )
            script.chmod(0o755)
        return bin_dir

    @staticmethod
    def _env(bin_dir) -> dict[str, str]:
        env = {"PATH": str(bin_dir), "NPM_SAYS": "registry.txt"}
        # cmd.exe, which runs a .cmd, needs these to start at all.
        for name in ("SYSTEMROOT", "COMSPEC", "PATHEXT"):
            if name in os.environ:
                env[name] = os.environ[name]
        return env

    def test_npm_is_asked_where_the_launch_runs_with_the_environment_it_gets(self, tmp_path) -> None:
        bin_dir = self._fake_npm(tmp_path)
        project = tmp_path / "project"
        project.mkdir()
        (project / "registry.txt").write_text("https://registry.example.internal/\n", encoding="ascii")

        assert mirrors._npm_registry(self._env(bin_dir), str(project)) == "https://registry.example.internal/"

    def test_a_directory_not_made_yet_is_asked_about_from_the_nearest_one_that_is(self, tmp_path) -> None:
        """A node makes a task's directory only after it has decided how to
        start it; nothing there can name a registry before it exists."""
        bin_dir = self._fake_npm(tmp_path)
        project = tmp_path / "project"
        project.mkdir()
        (project / "registry.txt").write_text("https://registry.example.internal/\n", encoding="ascii")

        assert mirrors._npm_registry(self._env(bin_dir), str(project / "task" / "work")) == (
            "https://registry.example.internal/"
        )

    def test_an_npm_that_fails_cannot_say(self, tmp_path) -> None:
        """Whatever it printed before it failed."""
        bin_dir = self._fake_npm(tmp_path, exit_code=3)

        assert mirrors._npm_registry(self._env(bin_dir), str(tmp_path)) is None

    def test_an_npm_that_hangs_cannot_say(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Asked with a deadline: an npm that does not answer must not hold the
        launch it was asked about."""
        monkeypatch.setattr(mirrors, "_NPM_TIMEOUT", 0.2)
        bin_dir = self._fake_npm(tmp_path, hang=True)
        env = {key: value for key, value in self._env(bin_dir).items() if key != "NPM_SAYS"}

        assert mirrors._npm_registry(env, str(tmp_path)) is None

    def test_no_npm_cannot_say(self, tmp_path) -> None:
        (tmp_path / "bin").mkdir()

        assert mirrors._npm_registry(self._env(tmp_path / "bin"), str(tmp_path)) is None


class _AnswerEverything(http.server.BaseHTTPRequestHandler):
    """A host that is there, and says no to everything."""

    def do_HEAD(self) -> None:  # noqa: N802 - the name http.server dispatches on
        self.send_response(404)
        self.end_headers()

    def log_message(self, *_args: object) -> None:
        return None


@pytest.fixture
def answering_host():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _AnswerEverything)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/"
    finally:
        server.shutdown()
        server.server_close()


def _closed_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class TestTheProbe:
    """Whether a default registry answers, asked once and remembered for a
    while: a gateway starts many agents, and each would otherwise pay for the
    same question."""

    @pytest.fixture(autouse=True)
    def _fresh(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(mirrors, "_verdicts", {})
        # Loopback is what these hosts are; a proxy from the runner's
        # environment must not stand in front of them.
        monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
        monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")

    def test_any_answer_counts_as_reachable(self, answering_host: str) -> None:
        """A 404 at the root still proves the host is there."""
        assert mirrors._reachable(answering_host)

    def test_a_refused_connection_is_not(self) -> None:
        assert not mirrors._reachable(f"http://127.0.0.1:{_closed_port()}/")

    def test_the_probe_needs_no_httpx(self, monkeypatch: pytest.MonkeyPatch, answering_host: str) -> None:
        """A Raven node has loguru and pillow, and starts npx agents too."""
        monkeypatch.setitem(sys.modules, "httpx", None)

        assert mirrors._reachable(answering_host)
        assert not mirrors._reachable(f"http://127.0.0.1:{_closed_port()}/")

    def test_a_failure_that_is_not_the_network_does_not_move_anything(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import urllib.request

        def broken(*_a: object, **_kw: object):
            raise ValueError("a bug of ours, not an unreachable host")

        monkeypatch.setattr(urllib.request, "urlopen", broken)

        assert mirrors._reachable("https://registry.npmjs.org/")

    def test_a_verdict_is_remembered_for_a_while(self, monkeypatch: pytest.MonkeyPatch) -> None:
        asked: list[str] = []
        now = [1000.0]
        monkeypatch.setattr(mirrors, "_answers", lambda url: asked.append(url) or True)
        monkeypatch.setattr(mirrors.time, "monotonic", lambda: now[0])

        mirrors._reachable("https://registry.npmjs.org/")
        mirrors._reachable("https://registry.npmjs.org/")
        assert len(asked) == 1

        now[0] += mirrors._VERDICT_TTL + 1
        mirrors._reachable("https://registry.npmjs.org/")
        assert len(asked) == 2, "a network that came back is noticed"

    def test_the_probe_does_not_wait_for_a_dropped_connection(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """It runs before an agent starts; a minute-long connect would be a
        minute-long launch."""
        import urllib.request

        seen: dict[str, object] = {}

        class _Answer:
            def __enter__(self) -> "_Answer":
                return self

            def __exit__(self, *_exc: object) -> None:
                return None

        def answer(request: object, timeout: float | None = None):
            seen["timeout"] = timeout
            return _Answer()

        monkeypatch.setattr(urllib.request, "urlopen", answer)

        mirrors._reachable("https://registry.npmjs.org/")

        assert isinstance(seen["timeout"], (int, float)) and seen["timeout"] <= 5
