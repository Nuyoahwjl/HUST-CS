CXX := g++
CXXFLAGS := -std=c++11 -Wall -Wextra -pedantic
INCLUDES := -I./include

SOURCES := main.cc helloworld.cc
OBJECTS := main.o helloworld.o

TARGET = helloworld

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $@

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $^ -o $@