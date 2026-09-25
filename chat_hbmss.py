"""HBMSS モデル対話シェルの起動口。実体は src/llm_lab/chat_hbmss.py（eval_hbmss.py と同じ流儀）。

    .venv/bin/python chat_hbmss.py --adapter outputs/<run>/checkpoint-340
    .venv/bin/python chat_hbmss.py --adapter outputs/<run>/checkpoint-340 --load-in-4bit
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from llm_lab.chat_hbmss import main  # noqa: E402

if __name__ == "__main__":
    main()
