import cv2
import numpy as np

def read_video(input_file):
    video = cv2.VideoCapture(input_file)
    frames = []
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        frames.append(frame)
    video.release()
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

def compress_video(input_file, delta):
    frames = read_video(input_file)
    ycbcr_frames = convert_color_space(frames, cv2.COLOR_BGR2YCrCb)

    height, width, _ = ycbcr_frames[0].shape
    compressed_data = []

    for y in range(height):
        for x in range(width):
            points = np.array([frame[y, x] for frame in ycbcr_frames])
            fitted_curves = [bezier_curve_fitting(channel_points, delta) for channel_points in points.T]
            compressed_data.append(fitted_curves)

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

if __name__ == "__main__":
    input_file = "input_video.mp4"
    output_file = "decompressed_video.mp4"
    delta = 8

    frames = read_video(input_file)
    fps = int(cv2.VideoCapture(input_file).get(cv2.CAP_PROP_FPS))

def process_video(input_file, output_file, delta, compress):
    if compress:
        frames = read_video(input_file)
        fps = int(cv2.VideoCapture(input_file).get(cv2.CAP_PROP_FPS))

        compressed_data = compress_video(input_file, delta)
        with open(output_file, 'wb') as f:
            pickle.dump(compressed_data, f)
    else:
        with open(input_file, 'rb') as f:
            compressed_data = pickle.load(f)

        decompressed_frames = decompress_video(compressed_data, delta, frames[0].shape[1], frames[0].shape[0], len(frames))
        save_video(output_file, decompressed_frames, fps)

if __name__ == "__main__":
    import argparse
    import pickle

    parser = argparse.ArgumentParser(description='Video Compression using Quadratic Bézier Curve Fitting')
    parser.add_argument('input_file', type=str, help='Input video file for compression or compressed data file for decompression')
    parser.add_argument('output_file', type=str, help='Output file for the compressed data or decompressed video')
    parser.add_argument('-d', '--delta', type=int, default=8, help='Breakpoint interval (default: 8)')
    parser.add_argument('-c', '--compress', action='store_true', help='Compress the input video file')

    args = parser.parse_args()

    process_video(args.input_file, args.output_file, args.delta, args.compress)