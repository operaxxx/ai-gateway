"""极简 .env 加载器：项目不引第三方依赖，用 10 行代码实现 dotenv 的核心语义。

规则：
- 只在文件存在时读取
- 每行一条 KEY=VALUE，支持 # 注释
- 用 setdefault：**已存在的环境变量优先**，.env 只是兜底（显式 export 永远赢）
"""

import os
from pathlib import Path


def load_env(path: str = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))
