# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class DetectorConfig:
    """YOLO dedektörü için temel çalışma parametreleri."""

    model: str = "yolov8s.pt"
    device: str = "cpu"
    conf_threshold: float = 0.35
    iou_threshold: float = 0.5
    person_class_id: int = 0


@dataclass
class TrackerConfig:
    """StrongSort izleyicisinin davranışını belirleyen ayarlar."""

    reid_model_path: Optional[Path] = None
    max_age: int = 30
    max_distance: float = 0.2
    max_iou_distance: float = 0.7
    max_observations: int = 50
    min_hits: int = 3
    min_confidence: float = 0.1
    mc_lambda: float = 0.98
    ema_alpha: float = 0.9
    n_init: int = 3
    nn_budget: int = 100
    half: bool = False


@dataclass
class CounterConfig:
    """Kimlik sayacının hafıza süresi ve raporlama aralığını yönetir."""

    ttl_seconds: float = 2.0
    print_interval: float = 1.0


@dataclass
class VisualizationConfig:
    """Yerel pencere ayarları ve isteğe bağlı başlangıç boyutu."""

    window_name: str = "EyesOnYou"
    show_fps: bool = True
    allow_resize: bool = True
    initial_width: Optional[int] = None
    initial_height: Optional[int] = None


@dataclass
class RuntimeConfig:
    """Tüm alt konfigürasyonları bir araya getirir."""

    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    counter: CounterConfig = field(default_factory=CounterConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)


def default_runtime_config() -> RuntimeConfig:
    """Varsayılan çalışma konfigürasyonunu döndürür."""

    return RuntimeConfig()
