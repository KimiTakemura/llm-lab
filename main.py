"""学習ランナーの起動口。

実体は src/llm_lab/train.py。パッケージが venv に未インストールなので、
src/ を import パスに足してから呼び出す。

    .venv/bin/python main.py --help
    .venv/bin/python main.py --lr 5e-4 --lora-r 64 --target-modules attn+mlp
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from llm_lab.train import main  # noqa: E402

if __name__ == "__main__":
    main()
