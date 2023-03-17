import cv2
import numpy as np
import time
import multiprocessing as mp
from tqdm import tqdm

def read_video(input_file):
    
    start_time = time.time()  # Start measuring time
    print("Reading video...")
    video = cv2.VideoCapture(input_file)
    frames = []
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        frames.append(frame)
    video.release()

    end_time = time.time()  # End measuring time
    print(f"Time taken to read the video: {end_time - start_time} seconds")
    # Calculate the size of video data in bytes
    video_data_size = sum([frame.nbytes for frame in frames])
    print(f"Size of video data: {video_data_size} bytes")

    return frames

def convert_color_space(frames, color_space):
    converted_frames = []
    for frame in frames:
        converted_frame = cv2.cvtColor(frame, color_space)
        converted_frames.append(converted_frame)
    return converted_frames

def bezier_curve_fitting(points, delta):
    breakpoints = points[::delta]
    segments = []
    for i in range(len(breakpoints) - 1):
        P0 = breakpoints[i]
        P2 = breakpoints[i + 1]
        m = delta + 1
        t = np.linspace(0, 1, m)
        P1 = np.sum(points[i * delta:i * delta + m] - (1 - t)**2 * P0 - t**2 * P2) / np.sum(2 * t * (1 - t))
        curve_points = (1 - t)**2 * P0 + 2 * t * (1 - t) * P1 + t**2 * P2
        segments.append((P0, P1, P2, curve_points))
    return segments
    
def compress_pixel_row(args):
    y, ycbcr_frames, width, delta, shared_progress = args
    compressed_row = []
    for x in range(width):
        points = np.array([frame[y, x] for frame in ycbcr_frames])
        fitted_curves = [bezier_curve_fitting(channel_points, delta) for channel_points in points.T]
        compressed_row.append(fitted_curves)
        with shared_progress.get_lock():
            shared_progress.value += 1
    return compressed_row

def compress_video(frames, delta):
    ycbcr_frames = convert_color_space(frames, cv2.COLOR_BGR2YCrCb)

    height, width, _ = ycbcr_frames[0].shape
    shared_progress = mp.Value('i', 0)

    with mp.Pool() as pool:
        args = [(y, ycbcr_frames, width, delta, shared_progress) for y in range(height)]
        result = pool.map_async(compress_pixel_row, args)

        with tqdm(total=height * width, desc="Compressing video", position=0) as progress_bar:
            while not result.ready():
                with shared_progress.get_lock():
                    progress_bar.n = shared_progress.value
                progress_bar.refresh()
                time.sleep(0.1)

        compressed_data = result.get()

    # Flatten the list
    compressed_data = [item for row in compressed_data for item in row]
    return compressed_data


def decompress_video(compressed_data, delta, width, height, num_frames):
    decompressed_frames = []

    for i in range(num_frames):
        decompressed_frame = np.zeros((height, width, 3), dtype=np.uint8)

        for y in range(height):
            for x in range(width):
                fitted_curves = compressed_data[y * width + x]
                t = (i % delta) / delta
                decompressed_frame[y, x] = [curve_points[int(i / delta)][3][int(t * delta)] for curve_points in fitted_curves]

        decompressed_frames.append(decompressed_frame)

    return decompressed_frames


def save_video(output_file, frames, fps):
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    for frame in frames:
        video.write(cv2.cvtColor(frame, cv2.COLOR_YCrCb2BGR))

    video.release()

def process_video(input_file, delta, compress):
    if compress:
        frames = read_video(input_file)
        fps = int(cv2.VideoCapture(input_file).get(cv2.CAP_PROP_FPS))

        compressed_data = compress_video(frames, delta)
        with open(f"{input_file}_compressed", 'wb') as f:
            pickle.dump(compressed_data, f)

        # Calculate the size of the compressed data file in bytes
        compressed_data_size = sum([len(pickle.dumps(curve)) for curve_list in compressed_data for curve in curve_list])
        print(f"Size of compressed data: {compressed_data_size} bytes")
        
    # else:
    #     with open(input_file, 'rb') as f:
    #         compressed_data = pickle.load(f)

    #     decompressed_frames = decompress_video(compressed_data, delta, frames[0].shape[1], frames[0].shape[0], len(frames))
    #     save_video(output_file, decompressed_frames, fps)

if __name__ == "__main__":
    import argparse
    import pickle

    parser = argparse.ArgumentParser(description='Video Compression using Quadratic Bézier Curve Fitting')
    parser.add_argument('input_file', type=str, help='Input video file for compression or compressed data file for decompression')
    parser.add_argument('-d', '--delta', type=int, default=8, help='Breakpoint interval (default: 8)')
    parser.add_argument('-c', '--compress', action='store_true', help='Compress the input video file')

    args = parser.parse_args()

    process_video(args.input_file, args.delta, args.compress)