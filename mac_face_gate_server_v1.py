#!/usr/bin/env python3
"""
Mac face gate server for Pi line-follow integration.

Flow:
- Pi sends: 4-byte big-endian length + jpeg bytes
- Mac replies per frame: 'wait' or 'resume' or 'quit'\n
User interaction:
- Press 'f' in window: send 'resume' and end current gate session (Pi continues line-follow)
- Press 'q' in window: send 'quit' and exit server
"""

from __future__ import annotations

import argparse
from collections import Counter
import socket
import struct
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from mac_face_test_v1 import (
    classify_faces,
    draw_results,
    load_known_faces_with_cache,
    resolve_cache_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mac face gate server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=5002, help="Bind port")
    parser.add_argument(
        "--known-faces-dir",
        type=Path,
        default=Path("known_faces"),
        help="Known faces root dir, e.g. known_faces/Olean/*.jpg",
    )
    parser.add_argument("--default-name", default="Olean", help="Fallback name")
    parser.add_argument("--cache-file", type=Path, default=None, help="Encoding cache path")
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable cache and force rebuilding known-face encodings",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.40,
        help="Match threshold. Smaller = stricter",
    )
    parser.add_argument(
        "--min-distance-gap",
        type=float,
        default=0.08,
        help="Require minimum distance gap between best and 2nd-best identity",
    )
    parser.add_argument(
        "--detection-model",
        choices=["hog", "cnn"],
        default="hog",
        help="Face detection model. hog is faster on CPU",
    )
    parser.add_argument(
        "--process-scale",
        type=float,
        default=0.5,
        help="Resize ratio for recognition processing",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=2,
        help="Run recognition every N frames",
    )
    parser.add_argument(
        "--show-no-face",
        action="store_true",
        help="Overlay 'no face' when no face is detected",
    )
    parser.add_argument(
        "--print-interval",
        type=float,
        default=1.0,
        help="Seconds between debug prints",
    )
    parser.add_argument(
        "--max-frame-bytes",
        type=int,
        default=5_000_000,
        help="Reject incoming frame payload larger than this",
    )
    parser.add_argument(
        "--swap-rb",
        action="store_true",
        default=True,
        help="Swap R/B channels on received frames before recognition/display (default: on)",
    )
    parser.add_argument(
        "--no-swap-rb",
        action="store_false",
        dest="swap_rb",
        help="Disable R/B swap if upstream already provides correct BGR",
    )
    return parser.parse_args()


def recv_exact(sock: socket.socket, n: int) -> Optional[bytes]:
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return bytes(data)


def send_cmd(conn: socket.socket, cmd: str) -> bool:
    try:
        conn.sendall((cmd + "\n").encode("utf-8"))
        return True
    except Exception:
        return False


def main() -> int:
    args = parse_args()

    if args.process_scale <= 0 or args.process_scale > 1:
        print("[ERROR] --process-scale must be in (0, 1].")
        return 2
    if args.frame_skip <= 0:
        print("[ERROR] --frame-skip must be >= 1.")
        return 2
    if args.min_distance_gap < 0:
        print("[ERROR] --min-distance-gap must be >= 0.")
        return 2

    known_dir = args.known_faces_dir
    cache_path = resolve_cache_path(known_dir, args.cache_file)
    known_encodings, known_names, loaded_images, loaded_from_cache = load_known_faces_with_cache(
        known_faces_dir=known_dir,
        default_name=args.default_name,
        cache_path=cache_path,
        use_cache=not args.no_cache,
    )

    print("\n=== Face DB Loaded ===")
    print(f"Known dir          : {known_dir.resolve()}")
    print(f"Cache file         : {cache_path.resolve()}")
    print(f"Loaded from cache  : {loaded_from_cache}")
    print(f"Images scanned     : {loaded_images}")
    print(f"Valid encodings    : {len(known_encodings)}")
    print(f"Identity labels    : {sorted(set(known_names))}")
    per_identity_counts = Counter(known_names)
    per_identity_text = ", ".join(
        f"{name}:{count}" for name, count in sorted(per_identity_counts.items())
    )
    print(f"Encodings by name  : {per_identity_text}")
    print("====================\n")

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(1)

    print(f"[mac-gate] Listening on {args.host}:{args.port} ...")
    print("[mac-gate] Key bindings: 'f' -> resume Pi, 'q' -> quit server")

    should_stop_server = False

    try:
        while not should_stop_server:
            conn, addr = server.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print(f"[mac-gate] Pi connected from {addr}")

            frame_count = 0
            last_face_locations: List[Tuple[int, int, int, int]] = []
            last_labels: List[str] = []
            last_distances: List[Optional[float]] = []
            last_print_ts = 0.0
            resume_requested = False

            try:
                while True:
                    header = recv_exact(conn, 4)
                    if header is None:
                        print("[mac-gate] Pi disconnected.")
                        break

                    msg_len = struct.unpack(">I", header)[0]
                    if msg_len <= 0 or msg_len > args.max_frame_bytes:
                        print(f"[WARN] Invalid payload length: {msg_len}, closing connection.")
                        break

                    jpeg_data = recv_exact(conn, msg_len)
                    if jpeg_data is None:
                        print("[mac-gate] Connection closed while receiving frame.")
                        break

                    npbuf = np.frombuffer(jpeg_data, dtype=np.uint8)
                    frame_bgr = cv2.imdecode(npbuf, cv2.IMREAD_COLOR)
                    if frame_bgr is None:
                        if not send_cmd(conn, "wait"):
                            break
                        continue

                    if args.swap_rb:
                        frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                    frame_count += 1
                    if frame_count % args.frame_skip == 0:
                        (
                            last_face_locations,
                            last_labels,
                            last_distances,
                        ) = classify_faces(
                            frame_bgr=frame_bgr,
                            known_encodings=known_encodings,
                            known_names=known_names,
                            tolerance=args.tolerance,
                            min_distance_gap=args.min_distance_gap,
                            detection_model=args.detection_model,
                            process_scale=args.process_scale,
                        )

                    canvas = draw_results(
                        frame_bgr=frame_bgr,
                        face_locations=last_face_locations,
                        labels=last_labels,
                        process_scale=args.process_scale,
                    )

                    if args.show_no_face and not last_face_locations:
                        cv2.putText(
                            canvas,
                            "no face",
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (80, 180, 255),
                            2,
                        )

                    cv2.putText(
                        canvas,
                        "press f to resume Pi",
                        (20, canvas.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (30, 255, 255),
                        2,
                    )

                    cv2.imshow("Mac Face Gate Server (f=resume, q=quit)", canvas)

                    now = time.time()
                    if now - last_print_ts >= args.print_interval:
                        if not last_face_locations:
                            print("[DEBUG] no face")
                        else:
                            parts = []
                            for i, label in enumerate(last_labels):
                                d = last_distances[i]
                                if d is None:
                                    parts.append(f"{label}(dist=n/a)")
                                else:
                                    parts.append(f"{label}(dist={d:.3f})")
                            print(
                                f"[DEBUG] faces={len(last_face_locations)} -> {', '.join(parts)}"
                            )
                        last_print_ts = now

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("f"):
                        resume_requested = True
                    elif key == ord("q"):
                        send_cmd(conn, "quit")
                        print("[mac-gate] Quit requested by user.")
                        should_stop_server = True
                        break

                    if resume_requested:
                        if send_cmd(conn, "resume"):
                            print("[mac-gate] Sent resume to Pi. Session ended.")
                        break
                    else:
                        if not send_cmd(conn, "wait"):
                            break

            finally:
                try:
                    conn.close()
                except Exception:
                    pass

    finally:
        try:
            server.close()
        except Exception:
            pass
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
