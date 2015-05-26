CXX        = nvcc
LD         = $(CXX)

LIBS_PATH = -L/usr/lib/openmpi/lib
LIBS =   -lmpi_cxx -lmpi -lopen-rte -lopen-pal -ldl -lnsl -lutil -lm
INCLUDE_PATH = -I/usr/lib/openmpi/include -I/usr/lib/openmpi/include/openmpi
FLAGS = -g -arch=sm_30
TARGET = "/home/denis/Diploma/K-means/MPIandCUDA/k-means.cu"
OBIN = "/home/denis/Diploma/K-means/MPIandCUDA/k-means"

all:	$(TARGET)

$(TARGET):
	$(LD) $(INCLUDE_PATH) $(FLAGS) $(TARGET) -o $(OBIN) $(LIBS_PATH) $(LIBS)
