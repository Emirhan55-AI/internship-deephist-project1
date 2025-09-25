# -*- coding: utf-8 -*-
"""EyesOnYou ana uygulaması"""

import argparse
import time
from collections import deque
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union

import cv2
import numpy as np
import torch

try:
    import supervision as sv
except ImportError as exc:  # pragma: no cover - modül eksikliği bilgi amaçlı
    raise ImportError(
        "supervision kütüphanesi bulunamadı. `pip install supervision[video]` komutuyla kurulum yap."
    ) from exc

try:
    from boxmot import StrongSort
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "BoxMOT (StrongSort) kurulmamış. `pip install boxmot` komutunu çalıştırıp README notlarını takip et."
    ) from exc

try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Ultralytics paketi yüklenmemiş. README'deki kurulum adımlarını uygula."
    ) from exc

from config import RuntimeConfig, default_runtime_config


class IdentityCounter:
    """Kimlikleri kısa süreli bellekle takip ederek sayaç değerinin kararlı kalmasını sağlar."""

    def __init__(self, ttl_seconds: float, print_interval: float) -> None:
        self.ttl_seconds = ttl_seconds
        self.print_interval = print_interval
        self.last_seen: dict[int, float] = {}
        self._last_report_ts: float = 0.0

    def update(self, track_ids: Iterable[int], timestamp: float) -> int:
        for track_id in track_ids:
            self.last_seen[int(track_id)] = timestamp

        expired = [tid for tid, seen_at in self.last_seen.items() if timestamp - seen_at > self.ttl_seconds]
        for tid in expired:
            del self.last_seen[tid]

        active_count = len(self.last_seen)
        if timestamp - self._last_report_ts >= self.print_interval:
            print(f"[EyesOnYou] Aktif öğrenci sayısı: {active_count}")
            self._last_report_ts = timestamp
        return active_count


class FPSMeter:
    """Hareketli pencere üzerinden saniyedeki kare sayısını hesaplar."""

    def __init__(self, window_size: int = 30) -> None:
        self.timestamps: deque[float] = deque(maxlen=window_size)
        self._fps: float = 0.0

    def update(self, timestamp: float) -> float:
        self.timestamps.append(timestamp)
        if len(self.timestamps) >= 2:
            delta = self.timestamps[-1] - self.timestamps[0]
            if delta > 0:
                self._fps = (len(self.timestamps) - 1) / delta
        return self._fps

    @property
    def fps(self) -> float:
        return self._fps


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EyesOnYou tespit ve takip prototipi")
    parser.add_argument("--kaynak", type=str, default="0", help="Video kaynağı (kamera indexi, dosya yolu veya RTSP URI)")
    parser.add_argument("--cihaz", type=str, default=None, help="YOLO modelinin çalışacağı cihaz (cuda, cpu, cuda:0 vb.)")
    parser.add_argument("--conf", type=float, default=None, help="Tespit için minimum güven değeri")
    parser.add_argument("--iou", type=float, default=None, help="NMS için IoU eşiği")
    parser.add_argument("--ttl", type=float, default=None, help="Kimlik hafızası için saniye cinsinden yaşam süresi")
    return parser.parse_args()


def apply_cli_overrides(config: RuntimeConfig, args: argparse.Namespace) -> None:
    if args.cihaz:
        config.detector.device = args.cihaz
    if args.conf is not None:
        config.detector.conf_threshold = args.conf
        config.tracker.min_confidence = args.conf
    if args.iou is not None:
        config.detector.iou_threshold = args.iou
    if args.ttl is not None:
        config.counter.ttl_seconds = args.ttl


def build_detector(config: RuntimeConfig) -> YOLO:
    model = YOLO(config.detector.model)
    try:
        model.to(config.detector.device)
    except Exception as exc:  # pragma: no cover - cihaz seçimi ortam bağlı
        print(f"[Uyarı] Model {config.detector.device} cihazına taşınamadı ({exc}). CPU kullanılacak.")
        model.to("cpu")
    return model


def build_tracker(config: RuntimeConfig) -> StrongSort:
    weights_dir = Path(__file__).resolve().parent.parent / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    weights_path = config.tracker.reid_model_path or weights_dir / "osnet_x0_25_msmt17.pt"

    return StrongSort(
        reid_weights=weights_path,
        device=config.detector.device,
        half=config.tracker.half,
        det_thresh=config.detector.conf_threshold,
        max_age=config.tracker.max_age,
        max_obs=max(config.tracker.max_observations, config.tracker.max_age + 5),
        min_hits=config.tracker.min_hits,
        iou_threshold=config.tracker.max_iou_distance,
        min_conf=config.tracker.min_confidence,
        max_cos_dist=config.tracker.max_distance,
        max_iou_dist=config.tracker.max_iou_distance,
        n_init=config.tracker.n_init,
        nn_budget=config.tracker.nn_budget,
        mc_lambda=config.tracker.mc_lambda,
        ema_alpha=config.tracker.ema_alpha,
    )


def detections_to_tracker_input(detections: sv.Detections) -> np.ndarray:
    if detections.xyxy.shape[0] == 0:
        return np.zeros((0, 6), dtype=np.float32)

    xyxy = detections.xyxy.astype(np.float32)
    confidence = detections.confidence.reshape(-1, 1).astype(np.float32)
    class_id = detections.class_id.reshape(-1, 1).astype(np.float32)
    return np.hstack((xyxy, confidence, class_id))


def tracker_output_to_detections(outputs: Optional[np.ndarray]) -> sv.Detections:
    if outputs is None or len(outputs) == 0:
        return sv.Detections.empty()

    xyxy = outputs[:, 0:4]
    track_ids = outputs[:, 4].astype(int)
    confidences = outputs[:, 5]
    class_ids = outputs[:, 6].astype(int)
    return sv.Detections(xyxy=xyxy, confidence=confidences, class_id=class_ids, tracker_id=track_ids)


def create_annotator() -> sv.BoxAnnotator:
    palette = sv.ColorPalette.DEFAULT
    return sv.BoxAnnotator(color=palette)


def draw_annotations(
    frame: np.ndarray,
    detections: sv.Detections,
    active_count: int,
    fps: float,
    show_fps: bool,
    box_annotator: sv.BoxAnnotator,
) -> np.ndarray:
    annotated = frame.copy()
    if detections.xyxy.shape[0] > 0:
        annotated = box_annotator.annotate(scene=annotated, detections=detections)
        tracker_ids = (
            detections.tracker_id
            if detections.tracker_id is not None
            else [None] * len(detections.xyxy)
        )
        for bbox, tracker_id, confidence in zip(detections.xyxy, tracker_ids, detections.confidence):
            x1, y1, x2, y2 = bbox.astype(int)
            label_parts = []
            if tracker_id is not None:
                label_parts.append(f"ID {tracker_id}")
            label_parts.append(f"Güven {confidence:.2f}")
            label = " | ".join(label_parts)
            position = (x1, max(y1 - 8, 12))
            cv2.putText(
                annotated,
                label,
                position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

    overlay_lines = [f"Aktif Öğrenci: {active_count}"]
    if show_fps and fps > 0:
        overlay_lines.append(f"FPS: {fps:.1f}")

    y_offset = 28
    for line in overlay_lines:
        cv2.putText(
            annotated,
            line,
            (16, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        y_offset += 30
    return annotated


def resolve_source(source: str) -> Union[str, int]:
    return int(source) if source.isdigit() else source


def process_frame(
    frame: np.ndarray,
    model: YOLO,
    tracker: StrongSort,
    config: RuntimeConfig,
    counter: IdentityCounter,
    fps_meter: FPSMeter,
    box_annotator: sv.BoxAnnotator,
) -> Tuple[np.ndarray, int, float]:
    with torch.inference_mode():
        results = model.predict(
            source=frame,
            conf=config.detector.conf_threshold,
            iou=config.detector.iou_threshold,
            verbose=False,
        )
    result = results[0]

    detections = sv.Detections.from_ultralytics(result)
    mask = detections.class_id == config.detector.person_class_id
    detections = detections[mask]

    tracker_input = detections_to_tracker_input(detections)
    tracker_outputs = tracker.update(tracker_input, frame)
    tracked_detections = tracker_output_to_detections(tracker_outputs)

    timestamp = time.time()
    track_ids = tracked_detections.tracker_id if tracked_detections.tracker_id is not None else []
    active_count = counter.update(track_ids, timestamp)
    fps = fps_meter.update(timestamp)

    annotated = draw_annotations(
        frame=frame,
        detections=tracked_detections,
        active_count=active_count,
        fps=fps,
        show_fps=config.visualization.show_fps,
        box_annotator=box_annotator,
    )
    return annotated, active_count, fps


def run() -> None:
    args = parse_arguments()
    config = default_runtime_config()
    apply_cli_overrides(config, args)

    model = build_detector(config)
    tracker = build_tracker(config)
    counter = IdentityCounter(ttl_seconds=config.counter.ttl_seconds, print_interval=config.counter.print_interval)
    fps_meter = FPSMeter()
    box_annotator = create_annotator()

    source = resolve_source(args.kaynak)
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Video kaynağı açılamadı: {args.kaynak}")

    window_name = config.visualization.window_name
    window_flags = cv2.WINDOW_NORMAL if config.visualization.allow_resize else cv2.WINDOW_AUTOSIZE
    cv2.namedWindow(window_name, window_flags)
    if config.visualization.allow_resize:
        width = config.visualization.initial_width or int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = config.visualization.initial_height or int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width and height:
            cv2.resizeWindow(window_name, int(width), int(height))

    try:
        while True:
            success, frame = capture.read()
            if not success:
                print("[Uyarı] Kare okunamadı, döngü sonlandırılıyor.")
                break

            annotated, _, _ = process_frame(
                frame=frame,
                model=model,
                tracker=tracker,
                config=config,
                counter=counter,
                fps_meter=fps_meter,
                box_annotator=box_annotator,
            )

            cv2.imshow(window_name, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                print("[Bilgi] Kullanıcı çıkışı algılandı.")
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


def main() -> None:
    run()


if __name__ == "__main__":
    main()
