# Variables
CXX = g++
CXXFLAGS = -Wall -std=c++17 -fopenmp
OPENCV_INCLUDE = `pkg-config --cflags opencv4`
OPENCV_LIBS = `pkg-config --libs opencv4`
EIGEN_INCLUDE = /usr/include/eigen3/
TARGET = qbc_compress

# Targets
all: $(TARGET)

$(TARGET): $(TARGET).o
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(TARGET).o $(OPENCV_LIBS)

$(TARGET).o: $(TARGET).cpp
	$(CXX) $(CXXFLAGS) -I$(EIGEN_INCLUDE) $(OPENCV_INCLUDE) -c $(TARGET).cpp

clean:
	rm -f *.o $(TARGET)