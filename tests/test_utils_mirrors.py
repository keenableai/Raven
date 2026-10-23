"""When a download may leave its default host for a mirror, and when it may not.

A mirror is only ever a second attempt: the host the file belongs to is asked
first, an answer from it (any answer, an error included) is final, and only a
connection that never completed moves the download on. These cases pin that
line, and the one switch that turns the whole fallback off.
"""

from __future__ import annotations

import http.server
import os
import socket
import ssl
import sys
import threading
import urllib.error

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
        ],
    )
    def test_a_connection_that_never_completed_is_unreachable(self, error: BaseException) -> None:
        assert mirrors.unreachable(error)

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
    """What `npm config get registry` answers, and the directories it was asked in."""

    def __init__(self) -> None:
        self.registry: str | None = mirrors.NPM_DEFAULT_REGISTRY
        self.asked_in: list[str | None] = []

    def registry_in(self, _env: object, cwd: str | None) -> str | None:
        self.asked_in.append(cwd)
        return self.registry


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
    def npm(self, monkeypatch: pytest.MonkeyPatch) -> _Npm:
        npm = _Npm()
        monkeypatch.setattr(mirrors, "_npm_registry", npm.registry_in)
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

    @pytest.mark.parametrize("program", ["npx", r"C:\Program Files\nodejs\npx.cmd", "/usr/local/bin/npx"])
    def test_npx_is_left_alone(self, network, npm, tmp_path, program: str) -> None:
        """npm hands its registry to everything npx starts, however it was given
        (measured with npm 11.12), and there it outranks the registry a project
        names in its own .npmrc: an agent's `npm install` in such a project
        would resolve that project's private packages from a public mirror."""
        network.down.add(mirrors.NPM_DEFAULT_REGISTRY)
        argv = [program, "-y", "@agentclientprotocol/codex-acp@1.13.1"]

        assert mirrors.package_launch(argv, self._home(tmp_path)) == argv
        assert network.asked == [] and npm.asked_in == [], "nothing is asked for a launch left alone"

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
