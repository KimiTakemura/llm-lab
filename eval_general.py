"""汎用能力の回帰テストの起動口。実体は src/llm_lab/eval_general.py。

    .venv/bin/python eval_general.py --help
    .venv/bin/python eval_general.py --backend hf --model-name Qwen/Qwen3-14B --limit 200
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from llm_lab.eval_general import main  # noqa: E402

if __name__ == "__main__":
    main()
