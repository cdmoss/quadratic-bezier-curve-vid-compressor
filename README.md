# Quadratic Bezier Curve Compression Program

This program is a simple implementation of the algorithm described in the paper https://link.springer.com/article/10.1007/s11760-010-0165-9. (Included as paper.pdf)

## Usage
Run debug.sh to compile and run the program. The program will read the data from the included video, and output the compressed data to the file \<video_name\>.qbc. Then it will automatically decode the compressed data and output the decoded data to the file \<video_name\>-interpolated.mp4

## Dependencies
- OpenCV
- C++11