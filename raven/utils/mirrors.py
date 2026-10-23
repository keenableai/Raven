"""Second hosts for downloads whose first host cannot be reached at all.

Measured from a mainland-China home line with no proxy: huggingface.co does not
answer, because its name resolves to an address that drops every packet, and
raw.githubusercontent.com times out more often than it serves. Each has a mirror
inside the country that serves the same bytes, and so do the registries and
download hosts the installers and the package runners rely on.

A mirror is only ever a second attempt. The default host is asked first, and an
answer from it, an error included, is final. Only a failure to reach it -- a
name that does not resolve, a connection refused, reset or left unanswered --
moves a download on (:func:`unreachable`). A host the operator chose, such as
``HF_ENDPOINT``, ``UV_DEFAULT_INDEX`` or an npm registry, is never replaced.
``RAVEN_NO_MIRROR=1`` turns all of it off.

install.sh, install.ps1 and the upgrade helper cannot import this module, so
each one carries its own copy of the hosts it needs. A test holds the copies
equal.
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit

from loguru import logger

OPT_OUT_ENV = "RAVEN_NO_MIRROR"
_TRUTHY = {"1", "true", "yes", "on"}

#: The default hosts asked before anything is redirected: npm's registry for
#: raven's own npm installs, PyPI for uvx, and the Python builds the installers
#: and the upgrade helper fetch through uv. Any HTTP answer from one proves it
#: can be reached.
NPM_DEFAULT_REGISTRY = "https://registry.npmjs.org/"
PYPI_DEFAULT_INDEX = "https://pypi.org/simple/"
PYTHON_DEFAULT_DOWNLOADS = "https://releases.astral.sh/"

#: huggingface.co's mirror, for the deepdoc models.
HF_ENDPOINT = "https://hf-mirror.com"
#: Files in GitHub repositories, through jsDelivr, byte for byte.
JSDELIVR_GITHUB = "https://cdn.jsdelivr.net/gh"
#: PyPI's mirror.
PYPI_INDEX = "https://pypi.tuna.tsinghua.edu.cn/simple"
#: The npm registry's mirror.
NPM_REGISTRY = "https://registry.npmmirror.com"
#: The Python builds uv downloads when no interpreter on the machine will do.
PYTHON_DOWNLOADS = "https://registry.npmmirror.com/-/binary/python-build-standalone"


def enabled() -> bool:
    """Whether a download may fall back to a mirror at all.

    Read the way raven's other opt-outs are read, so the installer and the
    `raven resources` it starts agree on what a value means.
    """
    return os.environ.get(OPT_OUT_ENV, "").strip().lower() not in _TRUTHY


def unreachable(exc: BaseException) -> bool:
    """Whether ``exc`` says a host could not be reached, rather than that it answered.

    A connection that was refused, reset or never completed counts, as does a
    name that did not resolve. An HTTP error does not, because the host spoke,
    and a mirror would only repeat the answer or serve something the host has
    withdrawn. A certificate that fails verification does not either: whatever
    re-signs this host's traffic re-signs the mirror's too, so falling back
    would only hide the reason. The cause chain is walked because libraries
    re-raise a failed connection as an error of their own (huggingface_hub's
    LocalEntryNotFoundError, httpx's ConnectError over the socket's, urllib's
    URLError with the socket's error as its reason).

    Standard library only, with httpx's errors added once a caller has loaded
    httpx: no other process can have raised one, and a Raven node runs without
    httpx installed.
    """
    import socket
    import ssl

    from raven.security.tls import is_untrusted_certificate

    if is_untrusted_certificate(exc):
        return False
    network: tuple[type[BaseException], ...] = (ConnectionError, TimeoutError, socket.gaierror, ssl.SSLError)
    httpx = sys.modules.get("httpx")
    if httpx is not None:
        network += (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError, httpx.ProxyError)
    seen: set[int] = set()
    pending: list[BaseException] = [exc]
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, network):
            return True
        reason = getattr(current, "reason", None)
        pending += [e for e in (current.__cause__, current.__context__, reason) if isinstance(e, BaseException)]
    return False


#: How long a probe's verdict stands. A gateway starts many agents, and each
#: would otherwise pay for the same question; a network that comes back is
#: noticed within this long.
_VERDICT_TTL = 600.0
#: How long a probe waits. It runs before a package runner starts, so a probe
#: that waited out a dropped connection would hold the launch just as long.
_PROBE_TIMEOUT = 5.0
_verdicts: dict[str, tuple[float, bool]] = {}
_verdicts_lock = threading.Lock()


def _answers(url: str) -> bool:
    """Whether anything answers at ``url``. Any HTTP status counts.

    urllib rather than httpx, for the node's sake (see :func:`unreachable`).
    Only a connection that never completed counts as no answer; any other
    failure is taken as one, so a fault of ours cannot send a runner to a mirror.
    """
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=_PROBE_TIMEOUT):
            return True
    except urllib.error.HTTPError:
        return True
    except Exception as exc:  # noqa: BLE001 - only the network may say no; see above
        if not unreachable(exc):
            return True
        logger.debug("{} could not be reached: {}", url, exc)
        return False


def _reachable(url: str) -> bool:
    """:func:`_answers`, remembered for :data:`_VERDICT_TTL`."""
    with _verdicts_lock:
        known = _verdicts.get(url)
    if known is not None and time.monotonic() - known[0] < _VERDICT_TTL:
        return known[1]
    answered = _answers(url)
    with _verdicts_lock:
        _verdicts[url] = (time.monotonic(), answered)
    return answered


def _reachable_all(urls: Sequence[str]) -> dict[str, bool]:
    """:func:`_reachable` for each of ``urls``, asked together, so hosts that
    drop every packet cost one wait between them rather than one each."""
    if len(urls) < 2:
        return {url: _reachable(url) for url in urls}
    with ThreadPoolExecutor(max_workers=len(urls)) as pool:
        return dict(zip(urls, pool.map(_reachable, urls), strict=True))


def _moved(mirror_of: Mapping[str, str]) -> set[str]:
    """Which of the default hosts in ``mirror_of`` to leave for its mirror.

    One that cannot be reached, and only when its mirror can: a runner offline
    still starts a package it fetched before, out of a cache it keeps by the
    registry's address, and pointed at a mirror it cannot reach either the
    same launch would fail.
    """
    down = [url for url, answered in _reachable_all(list(mirror_of)).items() if not answered]
    if not down:
        return set()
    up = _reachable_all([mirror_of[url] for url in down])
    return {url for url in down if up[mirror_of[url]]}


def _host(url: str) -> str:
    return urlsplit(url).hostname or url


def _program(word: str) -> str:
    """``uvx`` for ``uvx``, ``/usr/bin/uvx`` and ``C:\\...\\uvx.exe`` alike."""
    name = re.split(r"[\\/]", word)[-1].lower()
    stem, dot, suffix = name.rpartition(".")
    return stem if dot and suffix in {"cmd", "exe", "bat", "ps1"} else name


def _home(env: Mapping[str, str]) -> Path:
    home = env.get("USERPROFILE") if os.name == "nt" else env.get("HOME")
    return Path(home) if home else Path.home()


def _set(env: Mapping[str, str], *names: str) -> bool:
    """Whether any of ``names`` holds a value in ``env``, case-insensitively.

    npm reads its variables in either case, and Windows does not tell them apart.
    """
    wanted = {name.lower() for name in names}
    return any(key.lower() in wanted and value.strip() for key, value in env.items())


def _npm_registry_chosen(env: Mapping[str, str]) -> bool:
    """Whether the environment or the user's npmrc names a default registry,
    which settles it without asking anything. A scoped ``@corp:registry`` is
    not a default."""
    if _set(env, "npm_config_registry"):
        return True
    npmrc = Path(env["NPM_CONFIG_USERCONFIG"]) if env.get("NPM_CONFIG_USERCONFIG") else _home(env) / ".npmrc"
    try:
        text = npmrc.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return re.search(r"^\s*registry\s*=\s*\S", text, re.MULTILINE) is not None


#: How long npm gets to say which registry it reads; starting npm is most of it.
_NPM_TIMEOUT = 15.0


def _npm_registry(env: Mapping[str, str], cwd: str | None) -> str | None:
    """The registry npm would read, run in ``cwd`` with ``env``, or None when it cannot say.

    npm itself is asked because the variable this module sets outranks every
    file npm reads -- the project's .npmrc, the global one, npm's own -- so
    anything less would replace a registry the user named in one of them. A
    directory not made yet is asked about from the nearest one that is:
    nothing below it can name a registry before it exists.
    """
    npm = shutil.which("npm", path=env.get("PATH") or os.environ.get("PATH", os.defpath))
    if npm is None:
        return None
    where = Path(cwd) if cwd else None
    while where is not None and not where.is_dir():
        where = where.parent if where.parent != where else None
    try:
        done = subprocess.run(
            [npm, "config", "get", "registry"],
            cwd=where,
            env=dict(env),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=_NPM_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    return done.stdout.decode("utf-8", "replace").strip() or None


def npm_registry_env(env: Mapping[str, str], cwd: str | None = None) -> dict[str, str]:
    """The variable that points npm, run in ``cwd`` with ``env``, at its
    mirror; empty unless the registry npm would use is its default, which
    cannot be reached while the mirror can.

    For an npm raven runs in a directory of its own, whose lockfile is raven's,
    and which starts nothing that outlives it; :func:`package_launch` says why
    an npx launch is not given one. May block for a probe and for npm itself,
    and never raises: a fault while deciding leaves npm as it is.
    """
    if not enabled():
        return {}
    try:
        return _npm_registry_env(env, cwd)
    except Exception as exc:  # noqa: BLE001 - the mirror is a way around an outage, never a gate
        logger.opt(exception=exc).warning("could not decide on an npm mirror; npm starts as configured")
        return {}


def _npm_registry_env(env: Mapping[str, str], cwd: str | None) -> dict[str, str]:
    if _npm_registry_chosen(env) or not _moved({NPM_DEFAULT_REGISTRY: NPM_REGISTRY}):
        return {}
    configured = _npm_registry(env, cwd)
    if configured is None or configured.rstrip("/") != NPM_DEFAULT_REGISTRY.rstrip("/"):
        return {}
    logger.info(
        "{} could not be reached; npm fetches from {} instead ({}=1 turns this off)",
        _host(NPM_DEFAULT_REGISTRY),
        _host(NPM_REGISTRY),
        OPT_OUT_ENV,
    )
    return {"npm_config_registry": NPM_REGISTRY}


#: Where uv looks for the machine's uv.toml after the XDG directories.
_ETC = "/etc"


def _uv_configs(env: Mapping[str, str]) -> list[Path]:
    """The uv.toml files uv reads for a user who named none: theirs, then the machine's."""
    if env.get("UV_CONFIG_FILE"):
        return [Path(env["UV_CONFIG_FILE"])]
    if os.name == "nt":
        roots = [env.get("APPDATA"), env.get("PROGRAMDATA")]
    else:
        system = (env.get("XDG_CONFIG_DIRS") or "/etc/xdg").split(os.pathsep)
        roots = [env.get("XDG_CONFIG_HOME") or str(_home(env) / ".config"), *system, _ETC]
    return [Path(root) / "uv" / "uv.toml" for root in roots if root]


def _uv_settings(env: Mapping[str, str]) -> list[dict]:
    """Each of :func:`_uv_configs` that parses, on its own: one file's list of
    indexes does not hide another's default."""
    import tomllib

    settings: list[dict] = []
    for path in _uv_configs(env):
        try:
            settings.append(tomllib.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return settings


def _uv_index_chosen(env: Mapping[str, str]) -> bool:
    if _set(env, "UV_DEFAULT_INDEX", "UV_INDEX_URL", "UV_INDEX"):
        return True
    for config in _uv_settings(env):
        indexes = config.get("index") if isinstance(config.get("index"), list) else []
        if config.get("index-url") or any(isinstance(i, dict) and i.get("default") for i in indexes):
            return True
    return False


def _names(args: Sequence[str], flags: Sequence[str]) -> bool:
    """Whether any of ``flags`` is among ``args``, as a word of its own or with ``=value``."""
    return any(arg in flags or arg.startswith(tuple(f"{flag}=" for flag in flags)) for arg in args)


def _uv_index_named(args: Sequence[str]) -> bool:
    """Whether ``args`` choose uvx's index already; a second --default-index is one uv refuses."""
    return _names(args, ("--default-index", "--index-url", "--no-index")) or any(
        arg.startswith("-i") and not arg.startswith("--") for arg in args
    )


def package_launch(argv: Sequence[str], env: Mapping[str, str]) -> list[str]:
    """``argv``, with PyPI's mirror named on it where a ``uvx`` launch cannot
    reach pypi.org and the mirror answers.

    Only uvx, and only on its command line: uv keeps a flag to itself, so the
    choice ends with the fetch, while a variable would stay in the tool's
    environment and outrank the index a project names for itself. npx gets
    nothing, because npm hands its registry to everything it starts, however
    the registry was given -- a variable, a flag, an npmrc (measured with npm
    11.12) -- and there it would replace the registry a project names in its
    own .npmrc. ``uv run`` and ``npm install`` resolve the project they run
    in, and pnpm and bun read registries from files of their own; those start
    as configured too. ``env`` is the environment the launch gets, where the
    user's own choice of index would be found. A choice already made -- in the
    environment, a configuration file, or on the command line -- stands, and
    ``RAVEN_NO_MIRROR`` leaves every launch alone. May block for a probe; see
    :func:`package_launch_async`. Never raises: a fault while deciding leaves
    the launch as it was configured.
    """
    if not argv or not enabled():
        return list(argv)
    try:
        return _package_launch(list(argv), env)
    except Exception as exc:  # noqa: BLE001 - the mirror is a way around an outage, never a gate
        logger.opt(exception=exc).warning("could not decide on a mirror for {!r}; it starts as configured", argv[0])
        return list(argv)


def _package_launch(argv: list[str], env: Mapping[str, str]) -> list[str]:
    if _program(argv[0]) != "uvx":
        return argv
    # A uv.toml named on the command line may choose the index, and the flag
    # added here would outrank it.
    if _names(argv[1:], ("--config-file",)) or _uv_index_named(argv[1:]) or _uv_index_chosen(env):
        return argv
    if not _moved({PYPI_DEFAULT_INDEX: PYPI_INDEX}):
        return argv
    logger.info(
        "{} could not be reached; uvx fetches from {} instead ({}=1 turns this off)",
        _host(PYPI_DEFAULT_INDEX),
        _host(PYPI_INDEX),
        OPT_OUT_ENV,
    )
    return [argv[0], "--default-index", PYPI_INDEX, *argv[1:]]


async def package_launch_async(argv: Sequence[str], env: Mapping[str, str]) -> list[str]:
    """:func:`package_launch` off the event loop, since its probe blocks."""
    return await asyncio.to_thread(package_launch, argv, env)


__all__ = [
    "HF_ENDPOINT",
    "JSDELIVR_GITHUB",
    "NPM_DEFAULT_REGISTRY",
    "NPM_REGISTRY",
    "OPT_OUT_ENV",
    "PYPI_DEFAULT_INDEX",
    "PYPI_INDEX",
    "PYTHON_DEFAULT_DOWNLOADS",
    "PYTHON_DOWNLOADS",
    "enabled",
    "npm_registry_env",
    "package_launch",
    "package_launch_async",
    "unreachable",
]
