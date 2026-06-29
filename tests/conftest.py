from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def dataset_dir(repo_root: Path) -> Path:
    return repo_root / "data" / "dataset"


@pytest.fixture(scope="session")
def rawdata_root(repo_root: Path) -> Path:
    return repo_root / "data" / "rawdata"


@pytest.fixture(scope="session")
def fixtures_dir(repo_root: Path) -> Path:
    return repo_root / "tests" / "fixtures"
