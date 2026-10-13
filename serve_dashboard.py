"""
Local Dashboard Server with Cross-Origin Sharing (CORS) enabled.
Serves the dashboard directory on http://localhost:8000
"""

import http.server
import socketserver
from pathlib import Path

PORT = 8000
DIRECTORY = Path(__file__).resolve().parent / "dashboard"


class CORSRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Simple HTTP request handler with CORS headers enabled."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DIRECTORY), **kwargs)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "X-Requested-With, Content-Type")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()


def run_server(port: int = PORT):
    """Start local dashboard server with cross-sharing / CORS enabled."""
    handler = CORSRequestHandler
    with socketserver.TCPServer(("0.0.0.0", port), handler) as httpd:
        print(f"Dashboard serving at: http://localhost:{port}")
        print(f"Local network access:  http://0.0.0.0:{port}")
        print("Press Ctrl+C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")


if __name__ == "__main__":
    run_server()
