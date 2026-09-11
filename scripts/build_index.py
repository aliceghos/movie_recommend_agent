#!/usr/bin/env python
"""
离线构建 RAG 索引。

首次运行会下载嵌入模型权重（约 450MB），放在这里跑完，
避免用户第一次打开页面时对着空白界面等下载。

    python scripts/build_index.py            # 索引陈旧时才重建
    python scripts/build_index.py --force    # 无条件重建
"""

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv

load_dotenv(_REPO_ROOT / ".env")

from movie_agent import rag  # noqa: E402 - 必须在 sys.path 修补之后


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="即使索引是最新的也重建")
    args = parser.parse_args()

    if not args.force and not rag.index_is_stale():
        print("index is up to date — nothing to do (use --force to rebuild)")
        return 0

    print("building index...")
    rag.build_index(verbose=True)
    print(f"index written to {rag._INDEX_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
