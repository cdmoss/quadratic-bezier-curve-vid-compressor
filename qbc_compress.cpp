#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <omp.h>
#include <fstream>
#include <tuple>

double video_fps = 0.0;

struct BezierControlPoints {
    cv::Vec3f startPoint;
    cv::Vec3f middlePoint;
    cv::Vec3f endPoint;
};

std::vector<cv::Mat> readVideoData(const std::string& videoPath) {
    std::cout << "Reading video data..." << std::endl;

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

    std::cout << "Read " << frames.size() << " frames of video data" << std::endl;

    return frames;
}

std::size_t calculateUncompressedVideoSizeInBytes(const std::vector<cv::Mat>& frames) {
    std::size_t totalSizeInBytes = 0;

    for (const auto& frame : frames) {
        std::size_t frameSizeInBytes = frame.total() * frame.elemSize();
        totalSizeInBytes += frameSizeInBytes;
    }

    return totalSizeInBytes;
}

std::vector<std::vector<cv::Mat>> splitFramesIntoSegments(const std::vector<cv::Mat>& frames, int numSegments) {
    std::vector<std::vector<cv::Mat>> segments;
    int totalFrames = frames.size();

    // Calculate the size of each segment
    int segmentSize = totalFrames / numSegments;

    for (int i = 0; i < numSegments; ++i) {
        // Determine the start and end indices for the current segment
        int startIdx = i * segmentSize;
        int endIdx = (i == numSegments - 1) ? totalFrames : (i + 1) * segmentSize;

        // Create a vector to hold the frames for the current segment
        std::vector<cv::Mat> segmentFrames;

        // Add the frames to the current segment
        for (int j = startIdx; j < endIdx; ++j) {
            segmentFrames.push_back(frames[j]);
        }

        // Add the current segment to the list of segments
        segments.push_back(segmentFrames);
    }

    return segments;
}

std::vector<std::vector<BezierControlPoints>> calculateBezierControlPoints(const std::vector<cv::Mat>& segment) {
    int segmentSize = segment.size();
    int rows = segment[0].rows;
    int cols = segment[0].cols;

    std::vector<std::vector<BezierControlPoints>> controlPoints(rows, std::vector<BezierControlPoints>(cols));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            cv::Vec3f startPoint = segment.front().at<cv::Vec3b>(i, j);
            cv::Vec3f endPoint = segment.back().at<cv::Vec3b>(i, j);

            cv::Vec3f middlePoint(0, 0, 0);
            cv::Vec3f numerator(0, 0, 0);
            float denominator = 0;

            for (int frameIdx = 0; frameIdx < segmentSize; ++frameIdx) {
                float t = static_cast<float>(frameIdx) / (segmentSize - 1);
                cv::Vec3f v = segment[frameIdx].at<cv::Vec3b>(i, j);

                numerator += (v - (1 - t) * (1 - t) * startPoint - t * t * endPoint);
                denominator += 2 * t * (1 - t);
            }

            middlePoint = numerator / denominator;

            BezierControlPoints points = { startPoint, middlePoint, endPoint };
            controlPoints[i][j] = points;
        }
        // Print progress
        int progress = static_cast<int>(static_cast<float>(i) / (rows - 1) * 100);
        std::cout << "\rProgress: " << progress << "%";
        std::cout.flush();
    }

    std::cout << std::endl;
    return controlPoints;
}


std::vector<std::vector<std::vector<BezierControlPoints>>> calculateControlPointsForAllSegments(const std::vector<std::vector<cv::Mat>>& segments) {
    int numSegments = segments.size();
    std::vector<std::vector<std::vector<BezierControlPoints>>> allControlPoints(numSegments);
    std::vector<int> progress(numSegments, 0);

    // Set the number of threads to the number of segments
    omp_set_num_threads(numSegments);

    // Parallelize the loop that iterates over the segments
    #pragma omp parallel for
    for (int segmentIdx = 0; segmentIdx < numSegments; ++segmentIdx) {
        int numRows = segments[segmentIdx][0].rows;
        
        for (int row = 0; row < numRows; ++row) {
            // Update the progress for the current segment
            progress[segmentIdx] = static_cast<int>(static_cast<float>(row) / (numRows - 1) * 100);
        }

        allControlPoints[segmentIdx] = calculateBezierControlPoints(segments[segmentIdx]);
    }

    // Print the progress of each segment
    for (int i = 0; i < numSegments; ++i) {
        std::cout << "Segment " << i + 1 << " progress: " << progress[i] << "%" << std::endl;
    }

    return allControlPoints;
}

void writeControlPointsToFile(const std::string& videoPath, const std::vector<std::vector<std::vector<BezierControlPoints>>>& bezierControlPoints) {
    // Extract the video name and create the output file path
    std::string videoName = videoPath.substr(0, videoPath.find_last_of('.')) + ".qbc";
    std::ofstream outFile(videoName, std::ios::binary);

    if (!outFile.is_open()) {
        std::cerr << "Error: Could not open the output file for writing." << std::endl;
        return;
    }

    // Write the number of segments, rows, and columns
    int numSegments = bezierControlPoints.size();
    int rows = bezierControlPoints[0].size();
    int cols = bezierControlPoints[0][0].size();
    outFile.write(reinterpret_cast<const char*>(&numSegments), sizeof(int));
    outFile.write(reinterpret_cast<const char*>(&rows), sizeof(int));
    outFile.write(reinterpret_cast<const char*>(&cols), sizeof(int));

    // Write the control points

    for (const auto& segment : bezierControlPoints) {
        for (const auto& row : segment) {
            for (const auto& point : row) {
                outFile.write(reinterpret_cast<const char*>(&point.startPoint), sizeof(cv::Vec3f));
                outFile.write(reinterpret_cast<const char*>(&point.middlePoint), sizeof(cv::Vec3f));
                outFile.write(reinterpret_cast<const char*>(&point.endPoint), sizeof(cv::Vec3f));
            }
        }

    }

    std::cout << "Control points saved to: " << videoName << std::endl;
    std::cout << "Number of bytes written: " << outFile.tellp() << std::endl;
    outFile.close();
}

std::vector<std::vector<std::vector<BezierControlPoints>>> readControlPointsFromFile(const std::string& videoPath) {
    // Extract the video name and create the input file path
    std::string videoName = videoPath.substr(0, videoPath.find_last_of('.')) + ".qbc";
    std::ifstream inFile(videoName, std::ios::binary);

    if (!inFile.is_open()) {
        std::cerr << "Error: Could not open the input file for reading." << std::endl;
        return {};
    }

    // Read the number of segments, rows, and columns
    int numSegments, rows, cols;
    inFile.read(reinterpret_cast<char*>(&numSegments), sizeof(int));
    inFile.read(reinterpret_cast<char*>(&rows), sizeof(int));
    inFile.read(reinterpret_cast<char*>(&cols), sizeof(int));

    // Read the control points
    std::vector<std::vector<std::vector<BezierControlPoints>>> bezierControlPoints(numSegments, std::vector<std::vector<BezierControlPoints>>(rows, std::vector<BezierControlPoints>(cols)));

    for (auto& segment : bezierControlPoints) {
        for (auto& row : segment) {
            for (auto& point : row) {
                inFile.read(reinterpret_cast<char*>(&point.startPoint), sizeof(cv::Vec3f));
                inFile.read(reinterpret_cast<char*>(&point.middlePoint), sizeof(cv::Vec3f));
                inFile.read(reinterpret_cast<char*>(&point.endPoint), sizeof(cv::Vec3f));
            }
        }
    }

    inFile.close();
    std::cout << "Control points loaded from: " << videoName << std::endl;

    return bezierControlPoints;
}

void createInterpolatedVideo(const std::string& videoPath, const std::vector<std::vector<std::vector<BezierControlPoints>>>& bezierControlPoints, int numFrames) {
    std::string outputVideoPath = videoPath.substr(0, videoPath.find_last_of('.')) + "_interpolated.mp4";

    int numSegments = bezierControlPoints.size();
    int rows = bezierControlPoints[0].size();
    int cols = bezierControlPoints[0][0].size();
    int segmentSize = numFrames / numSegments;

    std::cout << "segmentSize: " << segmentSize << std::endl;

    cv::VideoWriter videoWriter(outputVideoPath, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), video_fps, cv::Size(cols, rows));

    if (!videoWriter.isOpened()) {
        std::cerr << "Error: Could not open the output video file for writing." << std::endl;
        return;
    }

    int totalFrames = numSegments * segmentSize;

    std::vector<cv::Mat> interpolatedFrames(totalFrames);

    for (size_t segmentIdx = 0; segmentIdx < numSegments; ++segmentIdx) {
        for (size_t frameIdx = 0; frameIdx < segmentSize; ++frameIdx) {
            float t = static_cast<float>(frameIdx) / (segmentSize - 1);
            int currentFrame = segmentIdx * segmentSize + frameIdx;
            cv::Mat frame(rows, cols, CV_8UC3);

            #pragma omp parallel for
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    const BezierControlPoints& points = bezierControlPoints[segmentIdx][i][j];
                    cv::Vec3f interpolatedPixel = (1 - t) * (1 - t) * points.startPoint + 2 * t * (1 - t) * points.middlePoint + t * t * points.endPoint;

                    // Clamp the values to [0, 255] and convert to 8-bit unsigned integer
                    frame.at<cv::Vec3b>(i, j) = cv::Vec3b(
                        static_cast<uchar>(std::clamp(interpolatedPixel[0], 0.f, 255.f)),
                        static_cast<uchar>(std::clamp(interpolatedPixel[1], 0.f, 255.f)),
                        static_cast<uchar>(std::clamp(interpolatedPixel[2], 0.f, 255.f))
                    );
                }
            }

            // Convert from RGB to BGR (default in OpenCV)
            cv::cvtColor(frame, frame, cv::COLOR_RGB2BGR);
            interpolatedFrames[currentFrame] = frame;

            // Calculate and print progress
            int progress = static_cast<int>(static_cast<float>(currentFrame) / (totalFrames - 1) * 100);
            std::cout << "\rProgress: " << progress << "%";
            std::cout.flush();
        }
    }

    // Write frames to the output video file
    for (const cv::Mat& frame : interpolatedFrames) {
        videoWriter.write(frame);
    }

    videoWriter.release();
    std::cout << std::endl << "Interpolated video saved to: " << outputVideoPath << std::endl;
}

std::size_t getFileSize(const std::string& filePath) {
    std::ifstream file(filePath, std::ifstream::ate | std::ifstream::binary);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open the file: " << filePath << std::endl;
        return 0;
    }

    std::size_t fileSize = file.tellg(); // Get the current file position (in bytes)
    file.close();

    return fileSize;
}

int main() {
    std::string videoPath = "coverr--07-022-22-gardening_0034-3713-1080p.mp4";
    std::vector<cv::Mat> frames = readVideoData(videoPath);

    std::cout << "Uncompressed video size: " << calculateUncompressedVideoSizeInBytes(frames) << " bytes" << std::endl;

    auto segments = splitFramesIntoSegments(frames, 10);

    std::vector<std::vector<std::vector<BezierControlPoints>>> bezierControlPoints = calculateControlPointsForAllSegments(segments);

    writeControlPointsToFile(videoPath, bezierControlPoints);

    std::vector<std::vector<std::vector<BezierControlPoints>>> readBezierControlPoints = readControlPointsFromFile(videoPath);

    
    // test to see if the read control points are the same as the original control points
    for (size_t i = 0; i < bezierControlPoints.size(); ++i) {
        for (size_t j = 0; j < bezierControlPoints[i].size(); ++j) {
            for (size_t k = 0; k < bezierControlPoints[i][j].size(); ++k) {
                if (bezierControlPoints[i][j][k].startPoint != readBezierControlPoints[i][j][k].startPoint) {
                    std::cout << "Error: Start points do not match." << std::endl;
                    return 1;
                }
                if (bezierControlPoints[i][j][k].middlePoint != readBezierControlPoints[i][j][k].middlePoint) {
                    std::cout << "Error: Middle points do not match." << std::endl;
                    return 1;
                }
                if (bezierControlPoints[i][j][k].endPoint != readBezierControlPoints[i][j][k].endPoint) {
                    std::cout << "Error: End points do not match." << std::endl;
                    return 1;
                }
            }
        }
    }

    createInterpolatedVideo(videoPath, bezierControlPoints, frames.size());

    return 0;
}