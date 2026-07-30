"""凍結図 PDF 5 本の sha256 を実測し `scripts/frozen_pdf_hashes.json` を re-pin する。

matplotlib の既定 savefig は wall-clock CreationDate を埋め込むため、内容が
数値的に同一でも `regenerate_main_body.py` / `regenerate_appendix_a.py` を
再実行するたびに main body 図 4 本（cdf_lolo・cdf_protocol_a_* 2 本・
segment_heatmap）の PDF は byte が変わる（`cdf_lolo_tier4.pdf` のみ
metadata={"CreationDate": None} で決定的）。そのため「正当な再生成をした後は
manifest を re-pin する」という運用が CLAUDE.md §1 / README §凍結契約で
決まっているが、その re-pin 作業自体を行う小さな helper がこのスクリプト。

使用例:
    uv run python scripts/regenerate_main_body.py    # 正当な再生成
    uv run python scripts/repin_pdf_hashes.py         # manifest を re-pin
    # → 生成された PDF 5 本を Preview.app 等で目視レビューしてから commit する
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FROZEN_PDF_HASHES_JSON = ROOT / "scripts" / "frozen_pdf_hashes.json"

try:
    from icsr8.harness_tier4 import FROZEN_OUTPUT_PATHS
except ImportError:  # editable install 未実施でも動くよう src を通す
    sys.path.insert(0, str(ROOT / "src"))
    from icsr8.harness_tier4 import FROZEN_OUTPUT_PATHS


def main() -> int:
    """`regenerate_main_body()` / `regenerate_appendix_a()` で PDF を正当に
    再生成した直後にのみ実行する。CreationDate 差異で変わった hash を
    manifest へ上書きするだけで、内容が意図どおりかどうかまでは検証しない
    ——実行後は必ず生成 PDF を目視 (Preview.app 等) でレビューし、意図した
    内容であることを確認してから commit すること。
    """
    old_entries = {
        e["path"]: e["sha256"]
        for e in json.loads(FROZEN_PDF_HASHES_JSON.read_text(encoding="utf-8"))
    }
    frozen_pdfs = sorted(
        str(p.relative_to(ROOT)) for p in FROZEN_OUTPUT_PATHS if p.suffix == ".pdf"
    )

    new_entries = []
    changed = 0
    for rel in frozen_pdfs:
        p = ROOT / rel
        new_hash = hashlib.sha256(p.read_bytes()).hexdigest()
        old_hash = old_entries.get(rel, "(new)")
        if new_hash != old_hash:
            changed += 1
            print(f"{rel}: {old_hash[:12]}… -> {new_hash[:12]}…")
        new_entries.append({"path": rel, "sha256": new_hash})

    FROZEN_PDF_HASHES_JSON.write_text(
        json.dumps(new_entries, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[repin_pdf_hashes] {changed} 件更新（manifest {len(new_entries)} 件）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
