from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _neutralize_icsr8_colab_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """全テストで ICSR8_REPO_SOURCE / ICSR8_WORKDIR 環境変数を中和する。

    # 2026-07-29 実機事故対応（2 段階で学習）:
    # ① env が実運用値のまま残ると、bootstrap()/stage_working_copy() を
    #    workdir 省略で呼ぶテストが本物の作業コピーを退避・再ステージする
    #    → まず delenv で中和した。
    # ② ところが実 Colab では delenv だけだと既定値 /content/icsr8_work に
    #    落ち、/content が書き込み可能なため結局本物を破壊した（ローカル
    #    macOS では / が書き込み不可で顕在化しない）。よって WORKDIR は
    #    「削除」ではなくテスト専用 tmp への**強制設定**にする。テストが
    #    明示的に setenv/delenv すればそちらが勝つ（autouse が先に走る）。
    """
    monkeypatch.delenv("ICSR8_REPO_SOURCE", raising=False)
    monkeypatch.setenv("ICSR8_WORKDIR", str(tmp_path / "icsr8_env_workdir"))


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
