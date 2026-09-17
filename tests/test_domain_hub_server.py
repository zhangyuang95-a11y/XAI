from __future__ import annotations

import http.client
import json
import threading

from domains.pong.web.server import PongApplication
from tests.test_warehouse_r42_manual_preview import context_for_protocol
from ui import domain_hub_server as hub
from ui import warehouse_alignment_r42_server as warehouse


def _request(port: int, method: str, path: str, *, body: bytes | None = None,
             headers: dict[str, str] | None = None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
    connection.request(method, path, body=body, headers=headers or {})
    response = connection.getresponse()
    result = response.status, dict(response.getheaders()), response.read()
    connection.close()
    return result


def test_domain_hub_mounts_both_studies_without_rewriting_warehouse_release(tmp_path):
    store = warehouse.OnlineAlignmentStudyStore(
        context_for_protocol(tmp_path), database=tmp_path / "hub.sqlite3",
        storage_mode="ephemeral", allow_internal_manual_groups=True,
    )
    server = hub.DomainHubHTTPServer(
        ("127.0.0.1", 0), store=store, pong=PongApplication(),
        public_origin="http://127.0.0.1",
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        status, _headers, body = _request(port, "GET", "/")
        assert status == 200
        assert b"/warehouse/" in body and b"/pong/" in body

        status, _headers, body = _request(port, "GET", "/warehouse/")
        assert status == 200
        assert b'id="warehouseCanvas"' in body
        assert b'"/warehouse/assets/' in body
        status, _headers, body = _request(port, "GET", "/warehouse/assets/app.js")
        assert status == 200
        assert b'"/warehouse/api/view"' in body

        status, _headers, body = _request(port, "GET", "/pong/")
        assert status == 200
        assert "合作接球".encode() in body
        payload = json.dumps({"group": "A", "participant_id": "hub_test"}).encode()
        status, headers, body = _request(
            port, "POST", "/pong/api/start", body=payload,
            headers={"Content-Type": "application/json", "Content-Length": str(len(payload))},
        )
        assert status == 200
        pong_cookie = headers["Set-Cookie"].split(";", 1)[0]
        started = json.loads(body)
        assert started["started"] is True and started["domain_id"] == "pong"
        status, _headers, body = _request(port, "GET", "/pong/api/view", headers={"Cookie": pong_cookie})
        assert status == 200 and json.loads(body)["started"] is True

        status, _headers, body = _request(port, "GET", "/health")
        assert status == 200
        assert json.loads(body)["domains"] == {"warehouse": "/warehouse/", "pong": "/pong/"}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        store.close()
