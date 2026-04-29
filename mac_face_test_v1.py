#!/usr/bin/env python3
"""
Mac Local Face Recognition Test v1

Goal:
- Run only on Mac
- Use built-in camera
- Load known faces from local directory
- Show configured name for matched face, otherwise unknown
- Press q to quit
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import face_recognition
import numpy as np
from PIL import Image, ImageOps

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CACHE_FORMAT_VERSION = 1
DEFAULT_CACHE_FILENAME = ".face_cache.pkl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mac local face recognition test")
    parser.add_argument(
        "--known-faces-dir",
        type=Path,
        default=Path("known_faces"),
        help="Known faces root dir, e.g. known_faces/Olean/*.jpg",
    )
    parser.add_argument(
        "--default-name",
        type=str,
        default="Olean",
        help="Fallback display name if directory name is empty",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Preferred camera index. Program will auto-fallback if this index is unavailable",
    )
    parser.add_argument(
        "--camera-scan-max",
        type=int,
        default=5,
        help="Maximum camera index to probe for fallback (0..N)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.45,
        help="Match threshold. Smaller = stricter. Typical range: 0.35~0.60",
    )
    parser.add_argument(
        "--min-distance-gap",
        type=float,
        default=0.06,
        help="Require a minimum gap between best and 2nd-best identity distance. Larger = stricter",
    )
    parser.add_argument(
        "--detection-model",
        choices=["hog", "cnn"],
        default="hog",
        help="Face detection model. hog is faster on CPU; cnn is slower but often more accurate",
    )
    parser.add_argument(
        "--process-scale",
        type=float,
        default=0.5,
        help="Resize ratio for processing each frame. Smaller = faster, but may miss tiny faces",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=2,
        help="Process every N frames (draw uses latest result). 1 means process every frame",
    )
    parser.add_argument(
        "--camera-width",
        type=int,
        default=1280,
        help="Requested camera width",
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=720,
        help="Requested camera height",
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
        help="Seconds between terminal debug prints",
    )
    parser.add_argument(
        "--cache-file",
        type=Path,
        default=None,
        help="Path to face-encoding cache file (default: known_faces/.face_cache.pkl)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable cache and force rebuilding known-face encodings",
    )
    return parser.parse_args()


def iter_image_files(folder: Path) -> List[Path]:
    files: List[Path] = []
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            files.append(p)
    return files


def load_image_rgb_with_exif(image_path: Path) -> np.ndarray:
    # iPhone photos often rely on EXIF orientation; normalize before detection.
    with Image.open(image_path) as pil_image:
        normalized = ImageOps.exif_transpose(pil_image).convert("RGB")
        return np.array(normalized)


def encode_known_face_image(
    image_rgb: np.ndarray,
) -> Tuple[List[np.ndarray], int, Optional[str], Optional[int]]:
    # Fallback attempts help recover faces that are small or slightly hard to detect.
    detection_attempts = [("hog", 0), ("hog", 1), ("hog", 2)]

    for model, upsample in detection_attempts:
        face_locations = face_recognition.face_locations(
            image_rgb,
            number_of_times_to_upsample=upsample,
            model=model,
        )
        if not face_locations:
            continue

        encodings = face_recognition.face_encodings(
            image_rgb,
            known_face_locations=face_locations,
            model="small",
        )
        if encodings:
            return encodings, len(face_locations), model, upsample

    return [], 0, None, None


def resolve_cache_path(known_faces_dir: Path, cache_file: Optional[Path]) -> Path:
    if cache_file is not None:
        return cache_file
    return known_faces_dir / DEFAULT_CACHE_FILENAME


def compute_dataset_signature(known_faces_dir: Path) -> Tuple[str, int]:
    records: List[Dict[str, Any]] = []
    image_count = 0

    person_dirs = [p for p in sorted(known_faces_dir.iterdir()) if p.is_dir()]
    for person_dir in person_dirs:
        for image_path in iter_image_files(person_dir):
            stat = image_path.stat()
            records.append(
                {
                    "relative_path": image_path.relative_to(known_faces_dir).as_posix(),
                    "size": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                }
            )
            image_count += 1

    payload = {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "records": records,
    }
    payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    signature = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
    return signature, image_count


def load_known_faces(
    known_faces_dir: Path,
    default_name: str,
) -> Tuple[List[np.ndarray], List[str], int]:
    if not known_faces_dir.exists() or not known_faces_dir.is_dir():
        raise FileNotFoundError(
            f"Known faces directory not found: {known_faces_dir.resolve()}"
        )

    known_encodings: List[np.ndarray] = []
    known_names: List[str] = []
    loaded_images = 0

    person_dirs = [p for p in sorted(known_faces_dir.iterdir()) if p.is_dir()]
    if not person_dirs:
        raise RuntimeError(
            f"No identity folder found under: {known_faces_dir.resolve()}"
        )

    for person_dir in person_dirs:
        name = person_dir.name.strip() or default_name
        image_files = iter_image_files(person_dir)

        if not image_files:
            print(f"[WARN] No images under {person_dir}")
            continue

        for image_path in image_files:
            loaded_images += 1
            try:
                image_rgb = load_image_rgb_with_exif(image_path)
                encodings, detected_faces, used_model, used_upsample = encode_known_face_image(
                    image_rgb
                )
            except Exception as exc:
                print(f"[WARN] Failed to parse {image_path.name}: {exc}")
                continue

            if len(encodings) == 0:
                print(f"[WARN] No face found in {image_path.name}, skipped")
                continue
            if detected_faces > 1:
                print(
                    f"[WARN] Multiple faces in {image_path.name}, using first face only"
                )
            elif used_model is not None and used_upsample is not None and used_upsample > 0:
                print(
                    f"[INFO] Recovered face in {image_path.name} "
                    f"(model={used_model}, upsample={used_upsample})"
                )

            known_encodings.append(encodings[0])
            known_names.append(name)

    if not known_encodings:
        raise RuntimeError(
            "No valid known face encodings loaded. Please add clear face photos first."
        )

    return known_encodings, known_names, loaded_images


def load_known_faces_with_cache(
    known_faces_dir: Path,
    default_name: str,
    cache_path: Path,
    use_cache: bool,
) -> Tuple[List[np.ndarray], List[str], int, bool]:
    dataset_signature, dataset_image_count = compute_dataset_signature(known_faces_dir)

    if use_cache and cache_path.exists():
        try:
            with cache_path.open("rb") as fp:
                cached = pickle.load(fp)

            valid_cache = (
                isinstance(cached, dict)
                and cached.get("cache_format_version") == CACHE_FORMAT_VERSION
                and cached.get("dataset_signature") == dataset_signature
                and isinstance(cached.get("known_names"), list)
                and isinstance(cached.get("known_encodings"), list)
                and len(cached.get("known_names")) == len(cached.get("known_encodings"))
                and len(cached.get("known_encodings")) > 0
            )
            if valid_cache:
                known_encodings = cached["known_encodings"]
                known_names = cached["known_names"]
                loaded_images = int(cached.get("loaded_images", dataset_image_count))
                print(f"[INFO] Loaded known-face cache: {cache_path.resolve()}")
                return known_encodings, known_names, loaded_images, True

            print("[INFO] Cache mismatch or invalid. Rebuilding known-face encodings...")
        except Exception as exc:
            print(f"[WARN] Failed to load cache file {cache_path}: {exc}")
            print("[INFO] Rebuilding known-face encodings from images...")

    known_encodings, known_names, loaded_images = load_known_faces(
        known_faces_dir=known_faces_dir,
        default_name=default_name,
    )

    if use_cache:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_payload = {
                "cache_format_version": CACHE_FORMAT_VERSION,
                "dataset_signature": dataset_signature,
                "loaded_images": loaded_images,
                "known_names": known_names,
                "known_encodings": known_encodings,
            }
            with cache_path.open("wb") as fp:
                pickle.dump(cache_payload, fp)
            print(f"[INFO] Saved known-face cache: {cache_path.resolve()}")
        except Exception as exc:
            print(f"[WARN] Failed to save cache file {cache_path}: {exc}")

    return known_encodings, known_names, loaded_images, False


def open_camera(camera_index: int, width: int, height: int) -> cv2.VideoCapture:
    if sys.platform == "darwin":
        cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
    else:
        cap = cv2.VideoCapture(camera_index)

    if width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    return cap


def open_camera_with_fallback(
    preferred_index: int,
    width: int,
    height: int,
    scan_max: int,
) -> Tuple[Optional[cv2.VideoCapture], Optional[int], List[int]]:
    if scan_max < 0:
        scan_max = 0

    candidates: List[int] = [preferred_index]
    for idx in range(scan_max + 1):
        if idx != preferred_index:
            candidates.append(idx)

    attempted: List[int] = []
    for idx in candidates:
        if idx < 0:
            continue
        attempted.append(idx)
        cap = open_camera(idx, width, height)
        if not cap.isOpened():
            cap.release()
            continue

        ok, frame = cap.read()
        if ok and frame is not None and frame.size > 0:
            return cap, idx, attempted

        cap.release()

    return None, None, attempted


def classify_faces(
    frame_bgr: np.ndarray,
    known_encodings: Sequence[np.ndarray],
    known_names: Sequence[str],
    tolerance: float,
    min_distance_gap: float,
    detection_model: str,
    process_scale: float,
) -> Tuple[List[Tuple[int, int, int, int]], List[str], List[Optional[float]]]:
    # Shrink frame for speed, then convert BGR -> RGB for face_recognition.
    small_frame = cv2.resize(
        frame_bgr, (0, 0), fx=process_scale, fy=process_scale, interpolation=cv2.INTER_AREA
    )
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame, model=detection_model)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    labels: List[str] = []
    best_distances: List[Optional[float]] = []

    for face_encoding in face_encodings:
        distances = face_recognition.face_distance(known_encodings, face_encoding)
        if len(distances) == 0:
            labels.append("unknown")
            best_distances.append(None)
            continue

        # Pick best match per identity first, then compare identities.
        per_identity_best: Dict[str, float] = {}
        for idx, d in enumerate(distances):
            name = known_names[idx]
            dist = float(d)
            prev = per_identity_best.get(name)
            if prev is None or dist < prev:
                per_identity_best[name] = dist

        ranked = sorted(per_identity_best.items(), key=lambda x: x[1])
        best_name, best_distance = ranked[0]
        second_best_distance = ranked[1][1] if len(ranked) > 1 else None

        gap_ok = (
            True
            if second_best_distance is None
            else (second_best_distance - best_distance) >= min_distance_gap
        )

        if best_distance <= tolerance and gap_ok:
            labels.append(best_name)
        else:
            labels.append("unknown")

        best_distances.append(best_distance)

    return face_locations, labels, best_distances


def draw_results(
    frame_bgr: np.ndarray,
    face_locations: Sequence[Tuple[int, int, int, int]],
    labels: Sequence[str],
    process_scale: float,
) -> np.ndarray:
    output = frame_bgr.copy()
    inv_scale = 1.0 / process_scale

    for (top, right, bottom, left), label in zip(face_locations, labels):
        top = int(top * inv_scale)
        right = int(right * inv_scale)
        bottom = int(bottom * inv_scale)
        left = int(left * inv_scale)

        color = (0, 180, 0) if label != "unknown" else (0, 0, 220)

        cv2.rectangle(output, (left, top), (right, bottom), color, 2)
        cv2.rectangle(output, (left, bottom - 26), (right, bottom), color, cv2.FILLED)
        cv2.putText(
            output,
            label,
            (left + 6, bottom - 6),
            cv2.FONT_HERSHEY_DUPLEX,
            0.6,
            (255, 255, 255),
            1,
        )

    return output


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

    print("\n=== Local Face DB Loaded ===")
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
    print("============================\n")

    cap, actual_camera_index, attempted_indices = open_camera_with_fallback(
        preferred_index=args.camera_index,
        width=args.camera_width,
        height=args.camera_height,
        scan_max=args.camera_scan_max,
    )
    if cap is None:
        print(
            "[ERROR] Failed to open camera. "
            f"Attempted indices: {attempted_indices}. "
            "Check camera permission and index."
        )
        return 1

    if actual_camera_index != args.camera_index:
        print(
            "[INFO] Preferred camera index "
            f"{args.camera_index} unavailable. Using fallback index {actual_camera_index}."
        )
    else:
        print(f"[INFO] Using camera index {actual_camera_index}.")

    print("Camera started. Press 'q' in the video window to exit.")

    last_face_locations: List[Tuple[int, int, int, int]] = []
    last_labels: List[str] = []
    last_distances: List[Optional[float]] = []
    frame_count = 0
    last_print_ts = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] Failed to read frame from camera.")
                continue

            frame_count += 1
            should_process = (frame_count % args.frame_skip) == 0

            if should_process:
                (
                    last_face_locations,
                    last_labels,
                    last_distances,
                ) = classify_faces(
                    frame_bgr=frame,
                    known_encodings=known_encodings,
                    known_names=known_names,
                    tolerance=args.tolerance,
                    min_distance_gap=args.min_distance_gap,
                    detection_model=args.detection_model,
                    process_scale=args.process_scale,
                )

            canvas = draw_results(
                frame_bgr=frame,
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

            cv2.imshow("Mac Local Face Test (press q to quit)", canvas)

            now = time.time()
            if (now - last_print_ts) >= args.print_interval:
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
                    print(f"[DEBUG] faces={len(last_face_locations)} -> {', '.join(parts)}")
                last_print_ts = now

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Exit requested by user.")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted by user.")
        raise SystemExit(130)
