#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <omp.h>
#include <fstream>

double video_fps = 0.0;

std::vector<cv::Mat> extractColorValues(const std::string& videoPath) {
    // Initialize the video capture object
    cv::VideoCapture cap(videoPath);

    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open the video file." << std::endl;
        return {};
    }

    // Get the frame rate of the video
    video_fps = cap.get(cv::CAP_PROP_FPS);

    std::vector<cv::Mat> frames;
    cv::Mat frame;

    while (cap.read(frame)) {
        // Determine the color space based on the number of channels
        int numChannels = frame.channels();
        if (numChannels == 1) {
            // The video is already in grayscale (luminance)
        } else if (numChannels == 3) {
            // Convert from BGR (default in OpenCV) to RGB
            cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
        } else {
            std::cerr << "Error: Unsupported number of channels: " << numChannels << std::endl;
            return {};
        }

        frames.push_back(frame.clone());
    }

    return frames;
}

std::vector<std::pair<int, int>> divideIntoSegments(const std::vector<cv::Mat>& frames, const std::vector<int>& breakpoints) {
    std::vector<std::pair<int, int>> segments;
    int start = 0;

    for (int breakpoint : breakpoints) {
        segments.push_back({start, breakpoint});
        start = breakpoint;
    }
    segments.push_back({start, static_cast<int>(frames.size())});

    return segments;
}

std::vector<cv::Mat> leastSquaresControlPoints(const std::vector<cv::Mat>& frames, int start, int end) {
    int rows = frames[0].rows;
    int cols = frames[0].cols;
    int count = end - start;

    std::vector<cv::Mat> control_points(3, cv::Mat::zeros(rows, cols, CV_32SC2));

    for (int i = start; i < end; i++) {
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                cv::Vec3b pixel_value = frames[i].at<cv::Vec3b>(r, c);
                control_points[0].at<cv::Point>(r, c) += cv::Point(i * pixel_value[0], i * pixel_value[1]);
                control_points[1].at<cv::Point>(r, c) += cv::Point(pixel_value[0], pixel_value[1]);
                control_points[2].at<cv::Point>(r, c) += cv::Point(i * pixel_value[2], pixel_value[2]);
            }
        }
    }

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            control_points[0].at<cv::Point>(r, c).x /= count;
            control_points[0].at<cv::Point>(r, c).y /= count;
            control_points[1].at<cv::Point>(r, c).x /= count;
            control_points[1].at<cv::Point>(r, c).y /= count;
            control_points[2].at<cv::Point>(r, c).x /= count;
            control_points[2].at<cv::Point>(r, c).y /= count;
        }
    }

    return control_points;
}
std::vector<std::vector<cv::Mat>> processSegments(const std::vector<cv::Mat>& frames, const std::vector<std::pair<int, int>>& segments) {
    int num_segments = segments.size();
    std::vector<std::vector<cv::Mat>> control_points(num_segments);

    #pragma omp parallel for
    for (int i = 0; i < num_segments; i++) {
        control_points[i] = leastSquaresControlPoints(frames, segments[i].first, segments[i].second);

        #pragma omp critical
        {
            int thread_num = omp_get_thread_num();
            int num_threads = omp_get_num_threads();
            int num_procs = omp_get_num_procs();
            std::cout << "Thread " << thread_num << " of " << num_threads << " (Total CPUs: " << num_procs << ") ";
            std::cout << "Processing segment " << i + 1 << " of " << num_segments << std::endl;
        }
    }

    return control_points;
}

std::vector<int> generateBreakpoints(int num_segments, int num_frames) {
    int segment_length = num_frames / num_segments;
    std::vector<int> breakpoints;

    for (int i = 1; i < num_segments; i++) {
        breakpoints.push_back(i * segment_length);
    }

    return breakpoints;
}

bool writeControlPointsToFile(const std::string& filePath, const std::vector<std::vector<cv::Mat>>& control_points) {
    std::ofstream outFile(filePath, std::ios::binary | std::ios::out);

    if (!outFile.is_open()) {
        std::cerr << "Error: Could not open the output file." << std::endl;
        return false;
    }

    int num_segments = control_points.size();
    outFile.write(reinterpret_cast<const char*>(&num_segments), sizeof(num_segments));

    for (const auto& segment_control_points : control_points) {
        int num_points = segment_control_points.size();
        outFile.write(reinterpret_cast<const char*>(&num_points), sizeof(num_points));
        for (const auto& point_mat : segment_control_points) {
            outFile.write(reinterpret_cast<const char*>(point_mat.data), point_mat.total() * point_mat.elemSize());
        }
    }

    outFile.close();
    return true;
}

std::vector<std::vector<cv::Mat>> readControlPointsFromFile(const std::string& filePath, cv::Size frame_size) {
    std::vector<std::vector<cv::Mat>> control_points;
    std::ifstream inFile(filePath, std::ios::binary | std::ios::in);

    if (!inFile.is_open()) {
        std::cerr << "Error: Could not open the input file." << std::endl;
        return control_points;
    }

    int num_segments;
    inFile.read(reinterpret_cast<char*>(&num_segments), sizeof(num_segments));
    control_points.resize(num_segments);

    for (int i = 0; i < num_segments; i++) {
        int num_points;
        inFile.read(reinterpret_cast<char*>(&num_points), sizeof(num_points));
        control_points[i].resize(num_points);
        for (auto& point_mat : control_points[i]) {
            point_mat.create(frame_size, CV_32SC2);
            inFile.read(reinterpret_cast<char*>(point_mat.data), point_mat.total() * point_mat.elemSize());
        }
    }

    inFile.close();
    return control_points;
}

cv::Mat interpolateFrame(const cv::Mat& ecp1, const cv::Mat& ecp2, const cv::Mat& mcp, double t) {
    cv::Mat frame1, frame2, frame;

    // Create a temporary matrix for each frame in the Bezier curve
    cv::addWeighted(ecp1, 1 - t, mcp, t, 0, frame1);
    cv::addWeighted(mcp, 1 - t, ecp2, t, 0, frame2);

    // Interpolate the final frame from the temporary matrices
    cv::addWeighted(frame1, 1 - t, frame2, t, 0, frame);

    return frame;
}


void createVideoFromControlPoints(const std::vector<std::vector<cv::Mat>>& control_points, cv::Size frame_size, const std::string& output_file) {
    cv::VideoWriter output_video(output_file, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), video_fps, frame_size, true);

    for (std::size_t i = 0; i < control_points.size() - 1; i++) {
        for (int t = 0; t < video_fps; t++) {
            double tt = static_cast<double>(t) / video_fps;
            cv::Mat ecp1 = control_points[i][0];
            cv::Mat mcp = control_points[i][1];
            cv::Mat ecp2 = control_points[i + 1][0];
            cv::Mat frame = interpolateFrame(ecp1, mcp, ecp2, tt);
            output_video.write(frame);
        }
    }

    output_video.release();
}

int main() {
    std::string videoPath = "./coverr-decorating-a-snowman-5925-1080p.mp4";
    std::vector<cv::Mat> frames = extractColorValues(videoPath);

    // Take the number of segments as input
    int num_segments = 8;

    // Generate the breakpoints
    std::vector<int> breakpoints = generateBreakpoints(num_segments, frames.size());

    // Divide the data into segments
    std::vector<std::pair<int, int>> segments = divideIntoSegments(frames, breakpoints);

    // Process the segments using parallel processing
    std::vector<std::vector<cv::Mat>> control_points = processSegments(frames, segments);

    // Save the control points to a binary file
    std::string control_points_file_path = "./control_points.bin";
    if (writeControlPointsToFile(control_points_file_path, control_points)) {
        std::cout << "Control points saved to: " << control_points_file_path << std::endl;
    } else {
        std::cerr << "Error: Failed to save control points." << std::endl;
    }

    cv::Size frame_size(frames[0].cols, frames[0].rows);

    // Read control points from the binary file
    std::vector<std::vector<cv::Mat>> loaded_control_points = readControlPointsFromFile(control_points_file_path, frame_size);

    

    if (loaded_control_points.empty()) {
        std::cerr << "Error: Failed to load control points." << std::endl;
    } else {
        std::cout << "Control points loaded from: " << control_points_file_path << std::endl;
    }

    // Create a new video using the loaded control points
    std::string decoded_video_path = "./decoded_video.mp4";

    createVideoFromControlPoints(loaded_control_points, frame_size, decoded_video_path);

    return 0;
}