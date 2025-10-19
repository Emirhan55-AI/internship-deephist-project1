# Configuration Guide - Eyes on You

This comprehensive guide explains every parameter in the `config.yaml` file. Use this document to understand what each setting does and how to configure it for optimal performance.

---

## Table of Contents

1. [Video Settings](#video-settings)
2. [Display Settings](#display-settings)
3. [YOLO Model Settings](#yolo-model-settings)
4. [BoT-SORT Tracking Settings](#bot-sort-tracking-settings)
5. [Student Counting Settings](#student-counting-settings)
6. [Visualization Settings](#visualization-settings)
7. [Performance Settings](#performance-settings)
8. [Statistics Settings](#statistics-settings)

---

## Video Settings

### `input_source`

**Type:** String (file path)
**Default:** `"data/input/classroom_1.mp4"`

Path to the input video file that will be processed.

**Examples:**

```yaml
input_source: "data/input/classroom_1.mp4"  # MP4 file
input_source: "data/input/classroom_2.mp4"  # Another video
input_source: 0                              # Webcam (camera index)
```

**Tips:**

- Use relative paths from the project root
- Supported formats: MP4, AVI, MOV, MKV
- For webcam: use `0` for default camera, `1` for secondary camera

---

### `frame_rate`

**Type:** Integer or null
**Default:** `null`

Video playback speed in frames per second (FPS). This is **critical** for the tracking algorithm.

**Why it matters:**

- The BoT-SORT tracking algorithm uses FPS to predict where students will be in the next frame
- It calculates motion speed and direction
- Determines how long to wait for temporarily lost tracks

**Options:**

- `null` - Auto-detect from video file (recommended)
- `30` - Standard video (NTSC)
- `25` - PAL video (European standard)
- `60` - High-speed video
- `24` - Cinema standard

**Example calculation:**

```yaml
track_buffer: 120
frame_rate: 30
# Wait time = 120 frames ÷ 30 fps = 4 seconds
```

**Best practice:** Leave as `null` unless you know the exact FPS or need to override it.

---

### `output.enabled`

**Type:** Boolean
**Default:** `true`

Enable or disable saving the processed video with tracking results.

**When to disable:**

- Testing only (don't need output file)
- Real-time processing with webcam
- Saving disk space

---

### `output.path`

**Type:** String (file path)
**Default:** `"data/output/tracked_output.mp4"`

Path where the processed video will be saved.

**Tips:**

- Directory will be created automatically if it doesn't exist
- Use `.mp4` extension for best compatibility
- Overwrites existing files

---

### `output.codec`

**Type:** String
**Default:** `"mp4v"`

Video codec for output file. Codec determines compression and quality.

**Common codecs:**

- `"mp4v"` - MPEG-4 (good compatibility)
- `"XVID"` - Xvid (smaller file size)
- `"MJPG"` - Motion JPEG (good quality, larger files)
- `"X264"` - H.264 (best quality, may not work on all systems)

**Note:** The system will try multiple codecs if the specified one fails.

---

### `output.fps`

**Type:** Float or null
**Default:** `null`

Output video frame rate. If `null`, matches the input video FPS.

**When to change:**

- Slow down video: `15` (half speed)
- Speed up video: `60` (double speed)
- Match display: `30` (standard playback)

---

## Display Settings

### `display.enabled`

**Type:** Boolean
**Default:** `true`

Show or hide the real-time display window during processing.

**When to disable:**

- Server/headless environments
- Faster processing (no rendering overhead)
- Batch processing multiple videos

---

### `display.window_name`

**Type:** String
**Default:** `"Eyes on You - Student Tracking"`

Title of the display window.

**Customization:**

```yaml
window_name: "My Custom Title"
```

---

### `display.fps_limit`

**Type:** Integer
**Default:** `30`

Maximum frames per second for display window. Higher values = smoother but more CPU usage.

**Recommendations:**

- `30` - Smooth playback, balanced
- `60` - Very smooth, more CPU
- `15` - Lower CPU, choppy playback

---

### `display.resize.enabled`

**Type:** Boolean
**Default:** `true`

Enable window resizing.

---

### `display.resize.width`

**Type:** Integer
**Default:** `1280`

Display window width in pixels.

**Common resolutions:**

- `1280` × `720` - HD (720p)
- `1920` × `1080` - Full HD (1080p)
- `640` × `480` - VGA

---

### `display.resize.height`

**Type:** Integer
**Default:** `720`

Display window height in pixels.

---

### `display.resize.maintain_aspect_ratio`

**Type:** Boolean
**Default:** `true`

Keep the original video aspect ratio when resizing. If `false`, video may be stretched.

**Recommendation:** Always keep `true` unless you have a specific reason.

---

## YOLO Model Settings

### `model.path`

**Type:** String (file path)
**Default:** `"models/yolo11s.pt"`

Path to the YOLO model weights file.

**Available models:**

- `yolo11s.pt` - Small, fast (recommended)

---

### `model.device`

**Type:** String
**Default:** `"cpu"`

Processing device for detection.

**Options:**

- `"cpu"` - CPU processing (slower, works everywhere)
- `"cuda"` - GPU processing (much faster, requires NVIDIA GPU and CUDA)

**Check if CUDA is available:**

```python
import torch
print(torch.cuda.is_available())
```

---

### `model.confidence_threshold`

**Type:** Float (0.0 - 1.0)
**Default:** `0.40`

Minimum confidence score for a detection to be considered valid.

**How it works:**

- `0.40` = 40% confidence required
- Lower value = more detections (including false positives)
- Higher value = fewer detections (more accurate, may miss some)

**Tuning tips:**

- Too many false detections? Increase to `0.50` or `0.60`
- Missing students? Decrease to `0.30` or `0.35`

---

### `model.iou_threshold`

**Type:** Float (0.0 - 1.0)
**Default:** `0.50`

Intersection over Union (IoU) threshold for Non-Maximum Suppression (NMS).

**What it does:**

- Removes overlapping bounding boxes
- Higher value = more aggressive removal of duplicates
- Lower value = keeps more boxes (may have duplicates)

**Recommendation:** Keep between `0.40` and `0.60`.

---

### `model.classes`

**Type:** List of integers
**Default:** `[0]`

COCO class IDs to detect. `[0]` = person class only.

**COCO classes:**

- `0` - person
- `2` - bicycle
- `3` - car
- `5` - bus
- (see full list: [COCO Classes](https://cocodataset.org/#explore))

**Example:** Detect people and cars:

```yaml
classes: [0, 3]
```

---

## BoT-SORT Tracking Settings

BoT-SORT (Boosting Tracking by SORT) is a robust multi-object tracking algorithm that combines motion and appearance features.

### `tracking.track_high_thresh`

**Type:** Float (0.0 - 1.0)
**Default:** `0.467`

High confidence threshold for track confirmation. A detection must exceed this to be confirmed as a new track.

**How it works:**

- Higher value = stricter confirmation (fewer false tracks)
- Lower value = easier confirmation (may create false tracks)

---

### `tracking.track_low_thresh`

**Type:** Float (0.0 - 1.0)
**Default:** `0.159`

Low confidence threshold for track deletion. A track below this is removed.

**How it works:**

- Higher value = tracks deleted sooner (fewer false tracks)
- Lower value = tracks kept longer (may keep false tracks)

---

### `tracking.new_track_thresh`

**Type:** Float (0.0 - 1.0)
**Default:** `0.716`

Threshold for creating new tracks from unmatched detections.

**Tuning:**

- Higher value = fewer new tracks (stricter)
- Lower value = more new tracks (may fragment)

---

### `tracking.track_buffer`

**Type:** Integer
**Default:** `120`

Number of frames to keep a lost track before deletion.

**Real-world example:**

```yaml
track_buffer: 120
frame_rate: 30
# Wait time = 120 ÷ 30 = 4 seconds
```

**When to adjust:**

- Students frequently occluded? Increase to `150` or `180`
- Too many false tracks? Decrease to `90` or `60`

---

### `tracking.match_thresh`

**Type:** Float (0.0 - 1.0)
**Default:** `0.90`

Threshold for matching detections to existing tracks.

**How it works:**

- Higher value = stricter matching (fewer mismatches)
- Lower value = easier matching (may cause ID switches)

---

### `tracking.proximity_thresh`

**Type:** Float (0.0 - 1.0)
**Default:** `0.50`

Proximity threshold for track association based on spatial distance.

**Tuning:**

- Dense crowds? Decrease to `0.40`
- Sparse scenes? Increase to `0.60`

---

### `tracking.appearance_thresh`

**Type:** Float (0.0 - 1.0)
**Default:** `0.70`

Appearance similarity threshold using ReID features.

**What it does:**

- Compares visual features to match tracks
- Higher value = stricter appearance matching
- Lower value = easier matching (may cause ID switches)

---

### `tracking.cmc_method`

**Type:** String
**Default:** `"ecc"`

Camera Motion Compensation method.

**Options:**

- `"ecc"` - Enhanced Correlation Coefficient (recommended)
- `"sparse"` - Sparse optical flow

**When to change:**

- Static camera? Either works
- Moving camera? Use `"ecc"`

---

### `tracking.reid_weights_path`

**Type:** String (file path)
**Default:** `"models/osnet_x0_25_msmt17.pt"`

Path to ReID (Re-identification) model weights for appearance matching.

**What it does:**

- Extracts visual features for appearance matching
- Helps maintain consistent IDs across occlusions

**Download:** Pre-trained models available in the models folder.

---

## Student Counting Settings

### `counter.confidence_threshold`

**Type:** Float (0.0 - 1.0)
**Default:** `0.5`

Minimum confidence to count a detection as a student.

**Note:** This is separate from `model.confidence_threshold`. This threshold is specifically for counting.

---

### `counter.max_confirmed_students`

**Type:** Integer
**Default:** `1000`

Maximum number of students to track simultaneously.

**When to change:**

- Large classroom? Increase to `2000`
- Small classroom? Decrease to `100`

---

## Visualization Settings

### `visualization.show_confidence`

**Type:** Boolean
**Default:** `true`

Display detection confidence scores on bounding boxes.

**Example:** `Person 0.87` (87% confidence)

---

### `visualization.show_track_id`

**Type:** Boolean
**Default:** `true`

Display unique track IDs on bounding boxes.

**Example:** `ID: 5`

---

### `visualization.show_trajectory`

**Type:** Boolean
**Default:** `true`

Show movement trails (trajectories) behind tracked students.

**Visual effect:** Colored lines showing recent movement path.

---

### `visualization.box_thickness`

**Type:** Integer
**Default:** `2`

Bounding box line thickness in pixels.

**Options:**

- `1` - Thin
- `2` - Medium (recommended)
- `3` - Thick

---

### `visualization.font_scale`

**Type:** Float
**Default:** `0.6`

Text size multiplier.

**Options:**

- `0.5` - Small
- `0.6` - Medium (recommended)
- `1.0` - Large

---

### `visualization.font_thickness`

**Type:** Integer
**Default:** `2`

Text line thickness in pixels.

**Options:**

- `1` - Thin
- `2` - Medium (recommended)
- `3` - Thick

---

## Performance Settings

### `performance.max_duration`

**Type:** Integer (seconds) or null
**Default:** `null`

Maximum processing time in seconds. Processing stops after this duration.

**When to use:**

- Testing: `60` (1 minute)
- Demo: `300` (5 minutes)
- Full video: `null` (no limit)

---

### `performance.verbose`

**Type:** Boolean
**Default:** `true`

Enable detailed console output.

**Output includes:**

- Frame processing progress
- FPS statistics
- Student count
- Warnings and errors

---

### `performance.frame_skip`

**Type:** Integer
**Default:** `1`

Process every Nth frame. `1` = all frames, `2` = every other frame.

**When to use:**

- Faster processing: `2` or `3`
- Real-time: `1` (all frames)

**Trade-off:**

- Higher skip = faster but less accurate tracking
- Lower skip = slower but more accurate

---

## Statistics Settings

### `statistics.show_progress`

**Type:** Boolean
**Default:** `true`

Show real-time progress updates in console.

**Example output:**

```
Progress: 45.3% (23.5s/52.0s) - FPS: 28.5 - Students: 12
```

---

### `statistics.show_final_stats`

**Type:** Boolean
**Default:** `true`

Display summary statistics at the end of processing.

**Includes:**

- Total processing time
- Average FPS
- Total frames processed
- Unique students tracked

---

### `statistics.progress_format`

**Type:** String
**Default:** `"Progress: {progress:.1f}% ({elapsed:.1f}s/{total:.1f}s) - FPS: {fps:.1f} - Students: {students}"`

Format string for progress messages.

**Available variables:**

- `{progress}` - Progress percentage
- `{elapsed}` - Elapsed time (seconds)
- `{total}` - Total time (seconds)
- `{fps}` - Current FPS
- `{students}` - Current student count

**Customization example:**

```yaml
progress_format: "Frame {frame}/{total} - Speed: {fps:.0f} FPS"
```

---

Remember: Every video is different. Tune parameters based on your specific use case!
