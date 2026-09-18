#!/usr/bin/env python3
"""Serve Pong with an explicit exported NN bundle for local browser play."""
from __future__ import annotations
import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT=Path(__file__).resolve().parents[1]; WEB=ROOT/"domains"/"pong"/"web"
class Handler(SimpleHTTPRequestHandler):
    bundle: Path
    def translate_path(self,path: str) -> str:
        item=urlparse(path).path
        if item in {"/pong","/pong/","/pong/index.html"}: return str(WEB/"index.html")
        if item in {"/pong/app.js","/pong/styles.css","/pong/coordinated.js","/pong/explain.js"}: return str(WEB/item.rsplit("/",1)[-1])
        if item=="/pong/nn_model.json": return str(self.bundle/"nn_model.json")
        if item=="/pong/program.json": return str(self.bundle/"program.json")
        if item=="/pong/controller_config.json": return str(self.bundle/"controller_config.json")
        return str(WEB/"index.html")
def main()->int:
    parser=argparse.ArgumentParser();parser.add_argument("--bundle",required=True);parser.add_argument("--port",type=int,default=18765);args=parser.parse_args(); bundle=Path(args.bundle)
    if not (bundle/"nn_model.json").exists(): raise FileNotFoundError("bundle requires nn_model.json from export_pong_nn.py")
    Handler.bundle=bundle;server=ThreadingHTTPServer(("127.0.0.1",args.port),Handler);print(f"Pong NN: http://127.0.0.1:{args.port}/pong/")
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()
    return 0
if __name__=="__main__":raise SystemExit(main())
