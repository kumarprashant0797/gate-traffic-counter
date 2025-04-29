from entry_exit import run_camera
import json
import multiprocessing

def process_camera(camera_config):
    run_camera(camera_config)

if __name__ == "__main__":
    with open('config.json') as config_file:
        cf = json.load(config_file)
    
    # Create a process for each camera
    processes = []
    for camera in cf['cameras']:
        p = multiprocessing.Process(target=process_camera, args=(camera,))
        processes.append(p)
        p.start()
    
    # Wait for all processes to complete
    for p in processes:
        p.join()