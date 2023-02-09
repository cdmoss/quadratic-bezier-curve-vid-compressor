import cv2
import numpy as np
import argparse

def extract_pixel_values(filepath):
    # Load the video using OpenCV
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        raise Exception("Error opening video file")

    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # Create arrays to store the pixel values
    pixel_values = np.zeros((frame_count, frame_height, frame_width, 3), dtype=np.uint8)

    # Loop through all frames in the video
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        # Store the pixel values for each frame
        pixel_values[i] = frame

    cap.release()
    return pixel_values

def quadratic_bezier_compression(pixel_values, breakpoint, filename):
    height, width, frames, channels = pixel_values.shape

    # Create an array to store the control points
    control_points = np.zeros((height, width, frames, breakpoint, channels), dtype=np.float64)

    # Loop through each pixel position
    for i in range(height):
        for j in range(width):
            for k in range(channels):
                values = pixel_values[i, j, :, k]

                # Break the set of values into n-1 sets
                sets = np.array_split(values, breakpoint - 1)

                # Calculate the control points for each set of values
                for l in range(breakpoint - 1):
                    set_values = sets[l]
                    x = np.arange(len(set_values))
                    A = np.array([x**2, x, np.ones(len(x))]).T
                    b = set_values
                    control_points[i, j, sum(sets[:l]) + np.arange(len(set_values)), l] = np.linalg.lstsq(A, b, rcond=None)[0]

    # Write the control points to a file with a .qbcc extension
    np.save(filename + '.qbcc', control_points)

    return control_points

def reconstruct_frames_from_control_points(control_points, frames_per_second, filename):
    height, width, frames, breakpoint, channels = control_points.shape
    pixel_values = np.zeros((height, width, frames, channels), dtype=np.float64)

    # Loop through each pixel position
    for i in range(height):
        for j in range(width):
            for k in range(channels):
                for l in range(breakpoint - 1):
                    start = sum(np.array_split(np.arange(frames), breakpoint - 1)[:l])
                    end = sum(np.array_split(np.arange(frames), breakpoint - 1)[:l + 1])
                    set_frames = np.arange(start, end)
                    x = set_frames.astype(np.float64) / frames_per_second
                    A = np.array([x**2, x, np.ones(len(x))]).T
                    pixel_values[i, j, start:end, k] = A @ control_points[i, j, start:end, l]

    # Create the video from the reconstructed frame data
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(filename + '.mp4', fourcc, frames_per_second, (width, height), isColor=channels > 1)
    for i in range(frames):
        frame = np.zeros((height, width, channels), dtype=np.uint8)
        for j in range(height):
            for k in range(width):
                for l in range(channels):
                    frame[j, k, l] = int(pixel_values[j, k, i, l])
        video.write(frame)
    video.release()

    return video

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("filepath", help="Path to the video file")
	parser.add_argument("breakpoint", type=int, help="Breakpoint value (integer)")
	args = parser.parse_args()
	filepath = "path/to/video.mp4"
	breakpoint = args.breakpoint
	filename = "compressed_video"
	pixel_values = extract_pixel_values(filepath)
	control_points = quadratic_bezier_compression(pixel_values, breakpoint, filename)

	print("Compressed data written to: ", filename + '.qbcc')
	control_points_file = "compressed_video.qbcc"
	frames_per_second = 30
	filename = "reconstructed_video"
	control_points = np.load(control_points_file)
	video = reconstruct_frames_from_control_points(control_points, frames_per_second, filename)
	print("Video written to file: ", filename + '.mp4')