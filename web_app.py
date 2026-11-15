"""
web_app.py  -  Zero-Dependency Web Server for Smart Hostel System.

Runs completely on Python's BUILT-IN standard library (http.server).
Does NOT require Streamlit, Flask, or any external web framework.

Usage:
    python web_app.py
"""

import http.server
import socketserver
import webbrowser
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

# If dashboard.html does not exist, build it first
DASHBOARD_HTML = ROOT / "dashboard.html"
if not DASHBOARD_HTML.exists():
    print("[INFO] dashboard.html not found. Generating from outputs...")
    from build_html_dashboard import build_dashboard
    build_dashboard()

PORT = 8000

class CustomHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # Redirect root / to /dashboard.html
        if self.path in ["/", "/index.html"]:
            self.path = "/dashboard.html"
        return super().do_GET()

def start_server():
    # Allow port reuse so restarting doesn't hit 'Address already in use'
    socketserver.TCPServer.allow_reuse_address = True
    
    with socketserver.TCPServer(("", PORT), CustomHandler) as httpd:
        url = f"http://localhost:{PORT}/dashboard.html"
        print("=" * 65)
        print("  SMART HOSTEL SYSTEM - LOCAL WEB SERVER")
        print("=" * 65)
        print(f"  [OK] Server running at: {url}")
        print("  Opening browser automatically...")
        print("  Press Ctrl+C to stop the server.")
        print("=" * 65 + "\n")
        
        # Open in browser
        webbrowser.open(url)
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[INFO] Server stopped.")

if __name__ == "__main__":
    start_server()
