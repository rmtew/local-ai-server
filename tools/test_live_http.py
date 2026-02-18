#!/usr/bin/env python3
"""Test live HTTP streaming ASR by sending a WAV file in chunks."""

import argparse
import json
import struct
import sys
import threading
import time
import urllib.request
import wave


def main():
    parser = argparse.ArgumentParser(description="Test live streaming ASR")
    parser.add_argument("wav", help="WAV file to stream")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--chunk-ms", type=int, default=500,
                        help="Chunk size in ms (default 500)")
    parser.add_argument("--delay", type=float, default=0,
                        help="Delay between chunks in seconds (0 = no delay)")
    parser.add_argument("--language", default="en")
    args = parser.parse_args()

    base = f"http://localhost:{args.port}"

    # Read WAV file
    with wave.open(args.wav, "rb") as wf:
        assert wf.getnchannels() == 1, "Must be mono"
        assert wf.getsampwidth() == 2, "Must be 16-bit"
        rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    # Resample to 16kHz if needed
    if rate != 16000:
        print(f"WARNING: WAV is {rate}Hz, expected 16000Hz. Sending raw anyway.")

    n_samples = len(frames) // 2
    chunk_samples = int(rate * args.chunk_ms / 1000)
    chunk_bytes = chunk_samples * 2
    print(f"Audio: {n_samples} samples ({n_samples/rate:.1f}s), "
          f"chunk: {chunk_samples} samples ({args.chunk_ms}ms)")

    # Start SSE session
    body = json.dumps({"language": args.language}).encode()
    req = urllib.request.Request(
        f"{base}/v1/audio/transcriptions/live/start",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    sse_resp = urllib.request.urlopen(req)

    # Read SSE events in background
    tokens = []
    done_event = [None]

    def sse_reader():
        buf = b""
        while True:
            chunk = sse_resp.read(1)
            if not chunk:
                break
            buf += chunk
            while b"\n\n" in buf:
                event, buf = buf.split(b"\n\n", 1)
                for line in event.split(b"\n"):
                    if line.startswith(b"data: "):
                        data = json.loads(line[6:])
                        if "token" in data:
                            tokens.append(data["token"])
                            sys.stdout.write(data["token"])
                            sys.stdout.flush()
                        elif data.get("done"):
                            done_event[0] = data
                            return

    reader = threading.Thread(target=sse_reader, daemon=True)
    reader.start()

    # Send audio chunks
    offset = 0
    chunk_num = 0
    while offset < len(frames):
        end = min(offset + chunk_bytes, len(frames))
        data = frames[offset:end]
        chunk_num += 1
        samples_in_chunk = len(data) // 2

        req = urllib.request.Request(
            f"{base}/v1/audio/transcriptions/live/audio",
            data=data,
            headers={"Content-Type": "application/octet-stream"},
            method="POST"
        )
        urllib.request.urlopen(req)

        if args.delay > 0:
            time.sleep(args.delay)
        offset = end

    print(f"\n--- Sent {chunk_num} chunks, {offset//2} samples ---")

    # Stop
    req = urllib.request.Request(
        f"{base}/v1/audio/transcriptions/live/stop",
        data=b"",
        method="POST"
    )
    urllib.request.urlopen(req)

    # Wait for done event
    reader.join(timeout=60)

    print("\n--- Results ---")
    if done_event[0]:
        d = done_event[0]
        print(f"Done text: {d.get('text', '')}")
        print(f"Duration: {d.get('duration', 0):.1f}s")
        print(f"Perf: total={d.get('perf_total_ms', 0)}ms "
              f"enc={d.get('perf_encode_ms', 0)}ms "
              f"dec={d.get('perf_decode_ms', 0)}ms")
    else:
        print("No done event received!")

    print(f"\nTokens received ({len(tokens)}): {''.join(tokens)}")


if __name__ == "__main__":
    main()
