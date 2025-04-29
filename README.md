# Gate Traffic Counter

A computer vision solution for automated vehicle counting at gates and checkpoints. Uses YOLOv8 for reliable vehicle detection and custom tracking algorithms to monitor directional flow. Supports multiple camera setups and different monitoring orientations.

## Features

- Real-time vehicle detection and tracking using YOLOv8
- Directional flow monitoring (entry/exit counting)
- Simultaneous processing of multiple camera streams using multiprocessing
- Configurable monitoring directions (horizontal/vertical)
- Region of interest (ROI) based processing to focus on specific areas
- Support for both GPU and CPU processing

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Make sure you have the YOLOv8 model file (yolov8n.pt) in the project directory.

3. Configure cameras in `config.json` (see configuration options below).

4. Run the system:
   ```bash
   python index.py
   ```

## Configuration Options

The system is configured through the `config.json` file with the following parameters:

### Global Parameters

| Parameter | Description | Values |
|-----------|-------------|--------|
| `conf` | Detection confidence threshold | 0.0-1.0 (higher values increase precision but may reduce detection rate) |
| `model` | Path to YOLOv8 model | Path to .pt file (e.g., "yolov8n.pt", "yolov8s.pt") |
| `device` | Processing device | "cpu", "0" (first GPU), "cuda:0" (GPU 0), etc. |

### Camera Parameters

Each camera in the `cameras` array can have the following parameters:

| Parameter | Description | Values |
|-----------|-------------|--------|
| `id` | Unique camera identifier | String (e.g., "C01", "Gate1") |
| `url` | Video source | Path to video file, RTSP URL, or webcam index ("0" for default camera) |
| `roi` | Region of interest | Array of [x,y] coordinates defining a polygon |
| `direction_mode` | Primary direction of traffic flow | "horizontal" or "vertical" |
| `entry_direction` | Direction considered as entry | For horizontal: "left_to_right" or "right_to_left"<br>For vertical: "top_to_bottom" or "bottom_to_top" |

## ROI Selection Tool

The project includes a tool to help define regions of interest:

```bash
python select_roi.py --cam <camera_url> --num 4
```

This tool allows you to click on the video frame to define points for your ROI.

## GPU vs CPU Processing

The system supports both GPU and CPU processing:

- **GPU Processing**: Faster processing, recommended for real-time applications with multiple cameras. Requires CUDA-compatible NVIDIA GPU and proper CUDA installation.
  
- **CPU Processing**: Works on any system but may be slower, especially with multiple cameras.

To select the processing device, use the `device` parameter in config.json:
- "cpu": Use CPU processing
- "0" or "cuda:0": Use the first GPU
- "cuda:1": Use the second GPU (if available)
- "cuda": Use the default GPU

## Performance Considerations

- For processing multiple high-resolution camera streams, a GPU is recommended.
- Each camera runs in its own process, utilizing multiple CPU cores if available.
- Smaller YOLOv8 models (yolov8n.pt) run faster but may be less accurate than larger models.

## Example Configuration

```json
{
    "conf": 0.5,
    "model": "yolov8n.pt",
    "device": "0",
    "cameras": [
        {
            "id": "Gate1",
            "url": "rtsp://192.168.1.100:554/stream1",
            "roi": [[310, 14], [618, 18], [610, 453], [309, 445]],
            "direction_mode": "horizontal",
            "entry_direction": "right_to_left"
        },
        {
            "id": "Gate2",
            "url": "0",
            "roi": [[300, 10], [600, 10], [600, 450], [300, 450]],
            "direction_mode": "vertical",
            "entry_direction": "top_to_bottom"
        }
    ]
}
