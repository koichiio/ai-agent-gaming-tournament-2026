"""Installed-copy gate: immutable dependency provenance and byte-for-byte sources."""

import hashlib
import importlib
import json
import sys
from importlib.metadata import distribution
from pathlib import Path

import pytest

COPY = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "name,module",
    [
        ("final-providers", "final_providers"),
        ("werewolf-runtime", "werewolf_runtime"),
    ],
)
def test_installed_dependency_is_copy_local_and_noneditable(name, module):
    assert Path(sys.prefix).resolve() == COPY / ".venv"
    package = importlib.import_module(module)
    assert Path(package.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    source = json.loads(distribution(name).read_text("direct_url.json"))
    assert source["url"] == (COPY / "vendor" / name).as_uri()
    assert source["dir_info"].get("editable", False) is False
    if module == "final_providers":
        factory = importlib.import_module(module + ".factory")
        assert (
            Path(factory.__file__).resolve().parent
            == Path(package.__file__).resolve().parent
        )


def test_frozen_connector_and_installed_runtime_match_copy_sources():
    provenance = json.loads((COPY / "PROVENANCE.json").read_text())
    hashes = provenance["connector_snapshot"]["files"]
    assert (
        hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
        == (provenance["connector_snapshot"]["sha256"])
    )
    for relative, digest in provenance["connector_snapshot"]["files"].items():
        source = COPY / "vendor/final-providers" / relative
        assert hashlib.sha256(source.read_bytes()).hexdigest() == digest, relative
    for name, module in (
        ("final-providers", "final_providers"),
        ("werewolf-runtime", "werewolf_runtime"),
    ):
        package = importlib.import_module(module)
        installed = Path(package.__file__).resolve().parent
        source = COPY / "vendor" / name / "src" / module
        for original in source.rglob("*.py"):
            assert (
                installed / original.relative_to(source)
            ).read_bytes() == original.read_bytes(), str(original)


def test_candidate_identity_and_editable_install_belong_to_this_copy():
    from agent_koichi.config import WORKSPACE

    assert COPY.name == "koichi-sol-v4"
    assert WORKSPACE == COPY
    source = json.loads(distribution("agent-koichi").read_text("direct_url.json"))
    assert source["url"] == COPY.as_uri()
    assert source["dir_info"]["editable"] is True
    assert json.loads((COPY / "PROVENANCE.json").read_text())["copy"] == COPY.name
