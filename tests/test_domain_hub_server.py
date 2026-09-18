from __future__ import annotations

import http.client
import json
from threading import RLock, Thread

from domains.pong.web.server import PongApplication
from ui import domain_hub_server as hub


class FakeWarehouseState:
    def __init__(self) -> None:
        self.lock = RLock()
        self.command_requests: list[str] = []
        self.reference_requests = 0

    def view(self):
        return {"domain_id": "warehouse", "stage": "task1"}

    def reference_trajectory(self):
        self.reference_requests += 1
        return {"reference": True}

    def command(self, payload):
        self.command_requests.append(str(payload.get("command")))
        return {"accepted": True}


class FakeWarehouseSessions:
    def __init__(self) -> None:
        self.state = FakeWarehouseState()
        self.page_ids: list[str | None] = []

    def state_for(self, page_id):
        self.page_ids.append(page_id)
        return self.state


def request(port: int, method: str, path: str, *, body: bytes | None = None, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
    connection.request(method, path, body=body, headers=headers or {})
    response = connection.getresponse()
    result = response.status, dict(response.getheaders()), response.read()
    connection.close()
    return result


def test_domain_hub_mounts_existing_warehouse_and_pong():
    sessions = FakeWarehouseSessions()
    server = hub.DomainHubHTTPServer(
        ("127.0.0.1", 0),
        warehouse_sessions=sessions,
        pong=PongApplication(),
        public_origin="http://127.0.0.1",
    )
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        status, _headers, body = request(port, "GET", "/")
        assert status == 200
        assert b"/warehouse/" in body and b"/pong/" in body

        status, _headers, body = request(port, "GET", "/warehouse/")
        assert status == 200
        assert b"/warehouse/assets/" in body
        status, _headers, body = request(port, "GET", "/warehouse/assets/app.js")
        assert status == 200
        assert b'"/warehouse/api/view"' in body
        status, _headers, body = request(port, "GET", "/warehouse/api/view", headers={"X-Warehouse-Page": "page-1"})
        assert status == 200 and json.loads(body)["domain_id"] == "warehouse"
        command = json.dumps({"command": "start"}).encode()
        status, _headers, body = request(
            port,
            "POST",
            "/warehouse/api/study/command",
            body=command,
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(command)),
                "X-Warehouse-Page": "page-1",
            },
        )
        assert status == 200 and json.loads(body)["accepted"] is True
        assert sessions.state.command_requests == ["start"]

        status, _headers, body = request(port, "GET", "/pong/")
        assert status == 200
        assert "合作接球".encode() in body
        assert b'href="styles.css"' in body and b'src="app.js"' in body
        status, _headers, body = request(port, "GET", "/pong/styles.css")
        assert status == 200 and b".court" in body
        status, _headers, body = request(port, "GET", "/pong/app.js")
        assert status == 200 and b"nn_model.json" in body
        for asset in ("coordinated.js", "explain.js", "nn_model.json", "controller_config.json"):
            status, _headers, body = request(port, "GET", f"/pong/{asset}")
            assert status == 200 and body
        assert json.loads(request(port, "GET", "/pong/controller_config.json")[2])["controller_mode"] == "coordinated"
        status, headers, _body = request(port, "GET", "/pong")
        assert status == 308 and headers["Location"] == "/pong/"

        status, _headers, body = request(port, "GET", "/pong/")
        assert status == 200 and "合作接球".encode() in body
        payload = json.dumps({"group": "A", "participant_id": "hub_test"}).encode()
        status, headers, body = request(port, "POST", "/pong/api/start", body=payload, headers={"Content-Type": "application/json"})
        assert status == 200
        cookie = headers["Set-Cookie"].split(";", 1)[0]
        assert json.loads(body)["domain_id"] == "pong"
        status, _headers, body = request(port, "GET", "/pong/api/view", headers={"Cookie": cookie})
        assert status == 200 and json.loads(body)["started"] is True

        status, _headers, body = request(port, "GET", "/health")
        assert status == 200
        assert json.loads(body)["domains"] == {"warehouse": "/warehouse/", "pong": "/pong/"}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
