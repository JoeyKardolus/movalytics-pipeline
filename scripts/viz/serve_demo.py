"""Simple HTTP server with Range request support for video seeking."""

import argparse
import os
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler


class RangeHTTPRequestHandler(SimpleHTTPRequestHandler):
    """HTTP handler that supports Range requests (required for video seeking)."""

    def send_head(self):
        path = self.translate_path(self.path)
        if not os.path.isfile(path):
            return super().send_head()

        range_header = self.headers.get("Range")
        if range_header is None:
            return super().send_head()

        # Parse Range: bytes=start-end
        try:
            range_spec = range_header.replace("bytes=", "").strip()
            parts = range_spec.split("-")
            file_size = os.path.getsize(path)
            start = int(parts[0]) if parts[0] else 0
            end = int(parts[1]) if parts[1] else file_size - 1
            end = min(end, file_size - 1)
            length = end - start + 1
        except (ValueError, IndexError):
            return super().send_head()

        ctype = self.guess_type(path)
        try:
            f = open(path, "rb")
        except OSError:
            self.send_error(404)
            return None

        f.seek(start)
        self.send_response(206)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()

        # Return only the requested range
        remaining = length
        buf_size = 64 * 1024
        import io
        buf = io.BytesIO()
        while remaining > 0:
            chunk = f.read(min(buf_size, remaining))
            if not chunk:
                break
            buf.write(chunk)
            remaining -= len(chunk)
        f.close()
        buf.seek(0)
        return buf


def main():
    parser = argparse.ArgumentParser(description="Serve demo page with video seeking support")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--directory", type=str, default="data/demo")
    args = parser.parse_args()

    handler = partial(RangeHTTPRequestHandler, directory=args.directory)
    server = HTTPServer(("", args.port), handler)
    print(f"Serving {args.directory} at http://localhost:{args.port}/demo.html")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
