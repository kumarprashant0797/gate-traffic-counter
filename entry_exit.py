from ultralytics import YOLO
from shapely.geometry import Polygon
from shapely.geometry.point import Point
import copy
import numpy as np
from datetime import datetime
import cv2
import json


def check_roi(roi, plate_box):
    x1, y1, x2, y2 = plate_box
    box_cent = Point(round((x1 + x2) / 2), round((y1 + y2) / 2))
    return roi.contains(box_cent)


def run_camera(cam):
    # Load config and initialize
    with open('config.json') as config_file:
        cf = json.load(config_file)
    
    model = YOLO(cf['model'])
    cam_id  = cam['id'] 
    cap = cv2.VideoCapture(cam['url'])
    
    # Get device from config (default to 'cpu' if not specified)
    device = cf.get('device', 'cpu')
    
    # Process ROI and direction settings
    roi_array = np.array(cam['roi'])
    roi_poly = Polygon(cam['roi'])
    direction_mode = cam['direction_mode']
    entry_direction = cam['entry_direction']
    height, width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    # Define colors for better readability
    GREEN, RED = (0, 255, 0), (0, 0, 255)
    BLUE, YELLOW = (255, 0, 0), (255, 255, 0)
    GRAY, BLACK = (128, 128, 128), (0, 0, 0)
    WHITE = (255, 255, 255)
    DARK_GREEN, DARK_RED = (0, 100, 0), (100, 0, 0)
    
    # Tracking variables
    entry_tracks, exit_tracks = [], []
    vehicle_positions, id_map = {}, {}
    next_id = 1
    threshold = 10  # Minimum movement to count as direction
    
    while True:
        ret, img = cap.read()
        if not ret or img is None:
            break

        # Process frame with model, now using the device from config
        results = model.track([img], classes=(2), conf=cf['conf'], 
                              persist=True, verbose=False, device=device)

        img_disp = copy.copy(img)
        cv2.polylines(img_disp, [roi_array], isClosed=True, color=BLUE, thickness=2)
        
        # Draw direction arrow
        arrow_start = (50, height-50)
        if direction_mode == 'vertical':
            if entry_direction == 'top_to_bottom':
                arrow_end = (50, height-20)
                direction_text = "Entry: Top to Bottom"
            else:
                arrow_end = (50, height-80)
                direction_text = "Entry: Bottom to Top"
        else:
            if entry_direction == 'left_to_right':
                arrow_end = (80, height-50)
                direction_text = "Entry: Left to Right"
            else:
                arrow_end = (20, height-50)
                direction_text = "Entry: Right to Left"
                
        cv2.putText(img_disp, direction_text, (10, height-60), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 1)
        cv2.arrowedLine(img_disp, arrow_start, arrow_end, GREEN, 2, tipLength=0.3)

        # Process detection results
        if len(results) > 0 and hasattr(results[0], 'boxes') and results[0].boxes is not None:
            for box in results[0].boxes:
                if box.id is None:
                    continue
                
                # Map original ID to sequential ID
                orig_id = int(box.id)
                if orig_id not in id_map:
                    id_map[orig_id] = next_id
                    next_id += 1
                box_id = id_map[orig_id]
                
                # Extract box data
                box_coords = [round(i) for i in box.xyxy[0].tolist()]
                veh_type = model.names[int(box.cls)]
                
                # Calculate center based on direction mode
                center_pos = (box_coords[1] + box_coords[3]) // 2 if direction_mode == 'vertical' else (box_coords[0] + box_coords[2]) // 2
                
                if check_roi(roi_poly, box_coords):
                    if box_id in vehicle_positions:
                        prev_pos = vehicle_positions[box_id]
                        
                        if box_id not in entry_tracks and box_id not in exit_tracks:
                            # Determine direction based on configuration
                            is_entry = is_exit = False
                            
                            if direction_mode == 'vertical':
                                if entry_direction == 'top_to_bottom':
                                    is_entry = center_pos > prev_pos + threshold
                                    is_exit = center_pos < prev_pos - threshold
                                else:  # bottom_to_top
                                    is_entry = center_pos < prev_pos - threshold
                                    is_exit = center_pos > prev_pos + threshold
                            else:  # horizontal
                                if entry_direction == 'left_to_right':
                                    is_entry = center_pos > prev_pos + threshold
                                    is_exit = center_pos < prev_pos - threshold
                                else:  # right_to_left
                                    is_entry = center_pos < prev_pos - threshold
                                    is_exit = center_pos > prev_pos + threshold
                            
                            # Record entry or exit
                            if is_entry or is_exit:
                                time_stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                if is_entry:
                                    entry_tracks.append(box_id)
                                    direction = "ENTRY"
                                else:
                                    exit_tracks.append(box_id)
                                    direction = "EXIT"
                                    
                                print(f'{time_stamp}, Camera ID: {cam_id}, Direction: {direction}, '
                                      f'Vehicle Type: {veh_type}, Entry Count: {len(entry_tracks)}, Exit Count: {len(exit_tracks)}')
                    
                    # Update position and determine box color
                    vehicle_positions[box_id] = center_pos
                    
                    if box_id in entry_tracks:
                        box_color, bg_color = GREEN, DARK_GREEN
                    elif box_id in exit_tracks:
                        box_color, bg_color = RED, DARK_RED
                    else:
                        box_color, bg_color = YELLOW, BLACK
                else:
                    box_color = GRAY
                    bg_color = BLACK
                
                # Draw bounding box
                cv2.rectangle(img_disp, (box_coords[0], box_coords[1]), 
                             (box_coords[2], box_coords[3]), box_color, 2)
                
                # Display ID
                txt_width, txt_height = cv2.getTextSize(str(box_id), cv2.FONT_HERSHEY_COMPLEX, 1, 2)[0]
                txt_width += 10
                txt_height += 10
                
                cv2.rectangle(img_disp, (box_coords[0], box_coords[1]), 
                             (box_coords[0] + txt_width, box_coords[1] - txt_height), bg_color, -1)
                cv2.putText(img_disp, str(box_id), (box_coords[0] + 5, box_coords[1] - 5), 
                           cv2.FONT_HERSHEY_COMPLEX, 1, WHITE, 2)

        # Display counts
        cv2.putText(img_disp, f"Entry: {len(entry_tracks)}", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, GREEN, 2)
        cv2.putText(img_disp, f"Exit: {len(exit_tracks)}", (20, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, RED, 2)

        cv2.imshow(cam_id, img_disp)
        if cv2.waitKey(1) == ord('q'):
            break
    
    cv2.destroyAllWindows()
    cap.release()
