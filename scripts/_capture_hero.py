
"""Capture a short hero GIF of the Resolution Console (bounded, headless).

Serves docs/ locally, records ~24s of the console after the boot sequence,
assembles docs/demo/hero.gif. Scratch tool; re-run any time.
"""
import functools, http.server, io, socketserver, threading, time
from playwright.sync_api import sync_playwright
from PIL import Image

PORT, W, H = 8732, 1280, 800
OUT = "docs/demo/hero.gif"
DURATION_S, STEP_S, SCALE = 24.0, 0.28, 0.75

Handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory="docs")
socketserver.TCPServer.allow_reuse_address = True
srv = socketserver.TCPServer(("127.0.0.1", PORT), Handler)
threading.Thread(target=srv.serve_forever, daemon=True).start()

frames = []
with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": W, "height": H})
    page.goto(f"http://127.0.0.1:{PORT}/", wait_until="load")
    time.sleep(4.0)  # boot sequence fades
    t_end = time.time() + DURATION_S
    while time.time() < t_end:
        frames.append(Image.open(io.BytesIO(page.screenshot(type="png"))).convert("RGB"))
        time.sleep(STEP_S)
    browser.close()
srv.shutdown()

frames = [f.resize((int(W * SCALE), int(H * SCALE))) for f in frames]
frames[0].save(
    OUT, save_all=True, append_images=frames[1:], duration=int(STEP_S * 1000),
    loop=0, optimize=True,
)
print(f"wrote {OUT}: {len(frames)} frames")
