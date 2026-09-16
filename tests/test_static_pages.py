"""静态管理页面测试：首页与静态资源可达，API 路由未被静态挂载遮蔽。

运行：
  uv run python -m pytest tests/test_static_pages.py -v
"""

from fastapi.testclient import TestClient

import server


def test_index_serves_html():
    c = TestClient(server.app)
    resp = c.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "AI Gateway" in resp.text


def test_static_assets_served():
    c = TestClient(server.app)
    assert c.get("/static/app.js").status_code == 200
    assert c.get("/static/style.css").status_code == 200


def test_api_routes_not_shadowed():
    c = TestClient(server.app)
    assert c.get("/health").status_code == 200
    assert c.get("/v1/models").status_code == 200
    assert c.get("/v1/prompts").status_code == 200
