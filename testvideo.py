#!/usr/bin/env python3.6

#Imports
import sys
import cv2
import time
import numpy as np
import pycuda.driver as cuda
from jetson_utils import videoSource
from yoloDet import YoloTRT
from depali import *
import matplotlib.pyplot as plt
import requests

# Aggregation and camera list
aggregation_list = get_agregation_its()
print(aggregation_list[-1])
camera_list = get_camera_agregation(aggregation_list)

# Aggregation object
agreg = Agregation(
    aggregation_list[-1]['aggregationName'], 
    aggregation_list[-1]['start'][:10], 
    aggregation_list[-1]['end'][:10], 
    2, 
    5, 
    aggregation_list[-1]['bearing'], 
    aggregation_list[-1]['tolerance_bearing'], 
    float(aggregation_list[-1]['latitude1']), 
    float(aggregation_list[-1]['longitude1']), 
    float(aggregation_list[-1]['latitude2']), 
    float(aggregation_list[-1]['longitude2']), 
    float(aggregation_list[-1]['latitude3']), 
    float(aggregation_list[-1]['longitude3']), 
    float(aggregation_list[-1]['latitude4']), 
    float(aggregation_list[-1]['longitude4']), 
    camera_list['camera_id'], 
    camera_list['name'], 
    (camera_list['latitude'], camera_list['longitude']), 
    '', 
    ''
)

agreg.compare_vectors()

confidence = 0.2

# User option for video source (video or camera)
use_camera = True 
if use_camera:
    input = videoSource("csi://0", argv=['--input-width=720', '--input-height=480'])
else:
    video_path = '/home/lab/JetsonYoloV7-TensorRT/videos/demo.mp4'
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Erreur lors de l'ouverture de la vidéo.")
        sys.exit()

#YoloTRT model
model = YoloTRT(library="yolov7/build/libmyplugins.so", engine="yolov7/build/yolov7-tiny.engine",conf=0.2, yolo_ver="v7")

# Variables for polygons
polygon_points = []
polygons = [] 
drawing = False
counting = True

# Mouse callbacks to draw polygons
def draw_polygon(event, x, y, flags, param):
    global polygon_points, polygons, drawing, counting

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(polygon_points) < 4:
            polygon_points.append((x, y))
        if len(polygon_points) == 4:
            drawing = False
            counting = True
            polygons.append(polygon_points.copy())
            polygon_points.clear()
cv2.namedWindow("Output")
cv2.setMouseCallback("Output", draw_polygon)

# Convert CUDA image to numpy array
def cudaImage_to_np(cuda_image):
    width = cuda_image.width
    height = cuda_image.height
    channels = cuda_image.channels
    np_array = np.zeros((height, width, channels), dtype=np.uint8)
    cuda.memcpy_dtoh(np_array, cuda_image.ptr)
    return np_array

# Check if a point is inside any of the polygons , Count detections within all polygons , Count detections over the entire image
def point_in_polygons(point, polygons):
    for polygon in polygons:
        if cv2.pointPolygonTest(np.array(polygon, dtype=np.int32), point, False) >= 0:
            return True
    return False

def count_detections(detections, polygons):
    counts = {"car": 0, "person": 0, "truck": 0, "bus": 0, "motorcycle": 0, "bicycle": 0,"train":0}
    for detection in detections:
        cls = detection['class']
        if cls in ["train","car", "motorcycle", "bus", "truck", "person", "bicycle"]:
            conf = detection['conf']
            if conf > confidence:
                center = (int((detection['box'][0] + detection['box'][2]) / 2), int((detection['box'][1] + detection['box'][3]) / 2))
                if point_in_polygons(center, polygons):
                    counts[cls] += 1
    return counts

def count_detections_entire_image(detections):
    counts = {"car": 0, "person": 0, "truck": 0, "bus": 0, "motorcycle": 0, "bicycle": 0,"train":0}
    for detection in detections:
        cls = detection['class']
        if cls in ["train","car", "motorcycle", "bus", "truck", "person", "bicycle"]:
            conf = detection['conf']
            if conf > confidence:
                counts[cls] += 1
    return counts

# Send or clear counter data to ITS
def send_counter_data_its(agreg, myobj):
    access_token = get_token()

    url = 'https://its.labatosbordeaux.eu/back/cdpits/aggregationRequest/data'
    headers = {'Authorization': f'Bearer {access_token}'}
    requests.post(url, json=myobj, headers=headers, verify=False)

def clear_its():
    total_counts = {"car": 0, "person": 0, "truck": 0, "bus": 0, "motorcycle": 0, "bicycle": 0,"train":0}
    myobj = {"aggregationId": getattr(agreg, 'id'),
             "carCount": total_counts['car'],
             "pedestrianCount": total_counts['person'],
             "truckCount": total_counts['truck'],
             "busCount": total_counts['bus'],
             "motoCounter": total_counts['motorcycle'],
             "bikeCounter": total_counts['bicycle'],
             "trainCounter": total_counts['train']}
    access_token = get_token()
    url = 'https://its.labatosbordeaux.eu/back/cdpits/aggregationRequest/data'
    headers = {'Authorization': f'Bearer {access_token}'}
    requests.post(url, json=myobj, headers=headers, verify=False)


# Display counts and traffic density on the frame
def show_counts_and_density(counts, polygons, frame_bgr):
    cars = counts['car']
    pedestrians = counts['person']
    bus = counts['bus']
    truck = counts['truck']
    moto = counts['motorcycle']
    bike = counts['bicycle']
    train = counts['train']

    cv2.putText(frame_bgr, f'CARS {cars}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    cv2.putText(frame_bgr, f'PEDESTRIANS {pedestrians}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    cv2.putText(frame_bgr, f'BUS {bus}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    cv2.putText(frame_bgr, f'TRUCKS {truck}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    cv2.putText(frame_bgr, f'MOTOS {moto}', (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    cv2.putText(frame_bgr, f'BIKES {bike}', (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    cv2.putText(frame_bgr, f'TRAINS {train}', (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)


MAX_FRAMES_BEFORE_SEND = 60  # sending infos to its each baatch of 60 frames
frame_count_since_send = 0
frame_count = 0
start_time = time.time()
frame_times = []
density_data = []
total_counts = None
heatmap = None

# Main loop
while True:
    frame_start_time = time.time()

    if use_camera:
        frame_cuda = input.Capture()
        if frame_cuda is None:
            break
        frame_bgr = cudaImage_to_np(frame_cuda)
        frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_RGB2BGR)
    else:
        ret, frame_bgr = cap.read()
        if not ret:
            break

    if heatmap is None:
        heatmap = np.zeros((frame_bgr.shape[0], frame_bgr.shape[1]), dtype=np.float32)

    # Detections
    detections, t = model.Inference(frame_bgr)

    for detection in detections:
        if detection['conf'] > confidence:
            x1, y1, x2, y2 = detection['box']
            heatmap[int(y1):int(y2), int(x1):int(x2)] += 1

    # Count detections in all polygons
    if counting and len(polygons) > 0:
        total_counts = {"car": 0, "person": 0, "truck": 0, "bus": 0, "motorcycle": 0, "bicycle": 0,"train":0}
        for polygon in polygons:
            counts = count_detections(detections, [polygon])
            total_counts['car'] += counts['car']
            total_counts['person'] += counts['person']
            total_counts['truck'] += counts['truck']
            total_counts['bus'] += counts['bus']
            total_counts['motorcycle'] += counts['motorcycle']
            total_counts['bicycle'] += counts['bicycle']
            total_counts['train']+= counts['train']

            color = (0, 255, 0) if sum(counts.values()) > 0 else (0, 0, 255)
            cv2.polylines(frame_bgr, [np.array(polygon, dtype=np.int32)], isClosed=True, color=color, thickness=2)

        show_counts_and_density(total_counts, polygons, frame_bgr)

    # Count detections over the entire image if no polygons are drawn
    elif counting and len(polygons) == 0:
        total_counts = count_detections_entire_image(detections)
        show_counts_and_density(total_counts, polygons, frame_bgr)

        # data to send
        myobj = {
            "aggregationId": getattr(agreg, 'id'),
            "carCount": total_counts['car'],
            "pedestrianCount": total_counts['person'],
            "truckCount": total_counts['truck'],
            "busCount": total_counts['bus'],
            "motoCounter": total_counts['motorcycle'],
            "bikeCounter": total_counts['bicycle'],
            "trainCounter": total_counts['train']
        }
        print(myobj)
        # Send data if required
        frame_count_since_send += 1
        if frame_count_since_send >= MAX_FRAMES_BEFORE_SEND:
            send_counter_data_its(agreg, myobj)
            frame_count_since_send = 0

    # Calculate FPS
    frame_count += 1
    frame_end_time = time.time()
    frame_times.append(frame_end_time - frame_start_time)
    if len(frame_times) > 10:
        frame_times.pop(0)
    fps = 1 / np.mean(frame_times)

    cv2.putText(frame_bgr, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    if total_counts is not None:
        density = sum(total_counts.values())
        density_data.append(density)

    # Display
    cv2.imshow("Output", frame_bgr)

    # key press events
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        # traffic density
        plt.figure(figsize=(10, 6))
        plt.plot(density_data)
        plt.xlabel('Frames')
        plt.ylabel('Traffic Density')
        plt.title('Traffic Density Plot')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        # Normalize heatmap
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
        heatmap = np.uint8(255 * heatmap)
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Reopen the video
        if use_camera:
            frame_cuda = input.Capture()
            frame_bgr = cudaImage_to_np(frame_cuda)
            frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_RGB2BGR)
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset video to first frame
            ret, frame_bgr = cap.read()

        # Overlay heatmap on the reopened frame
        overlay = cv2.addWeighted(frame_bgr, 0.6, heatmap_colored, 0.4, 0)

        # Display the overlay
        cv2.imshow("Heatmap", overlay)
        cv2.waitKey(0)

        clear_its()
        break    

# Release resources
if use_camera:
    input.Close()
else:
    cap.release()

cv2.destroyAllWindows()




