CC = g++
CFLAGS = -Wall -Werror -O3 -fopenmp
LIBS = `pkg-config --libs opencv4` -fopenmp
INCLUDES = `pkg-config --cflags opencv4`

TARGET = qbc_compress
SRC = qbc_compress.cpp

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(INCLUDES) -o $(TARGET) $(SRC) $(LIBS)

clean:
	rm -f $(TARGET)