"""stdio wrapper around mini_server: newline-delimited JSON on stdin/stdout."""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from mini_server import handle  # noqa: E402

print("starting up (this line is not JSON and must be ignored by the client)", flush=True)

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    message = json.loads(line)
    if message.get("method") == "tools/call" and message["params"]["name"] == "crash":
        os._exit(3)
    for reply in handle(message):
        print(json.dumps(reply), flush=True)
