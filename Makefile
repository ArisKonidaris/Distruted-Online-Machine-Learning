#
# MAKEFILE FOR MY FIRST C++ DEVELOPING FOLDER
#

############################
# COMPILER FLAGS
############################

ifndef MODE
MODE=DEBUG
endif

DLIB_INCLUDE = -I/usr/local/include
DLIB_INCLUDE += -I/usr/local/cuda-9.0/include
DLIB_LIB = -L/usr/local/lib -ldlib -lpthread -lX11 -lblas -llapack -lpng16 -lz
DLIB_LIB += -L/usr/local/cuda-9.0/lib64 -lcuda -lcudart -lcusolver -lcudnn -ldl -lrt -lcublas -lcurand -lcusolver -lstdc++ -lm -lgcc_s -lc -fopenmp
DLIB_LIB += -L/usr/include/opencv2 -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_videoio -lopencv_imgcodecs
DLIB_LIB += -L/usr/lib/gcc/x86_64-linux-gnu/7 -lgomp

#TENSOR_INCLUDE = -I/usr/local/include/eigen3
#TENSOR_INCLUDE += -I/home/aris/standalone/include/third_party
#TENSOR_INCLUDE += -I/home/aris/standalone/include
#TENSOR_INCLUDE += -I/home/aris/standalone/include/nsync/public/
#TENSOR_LIB:= -lprotobuf -pthread -lpthread -ltensorflow_cc -ltensorflow_framework

MLPACK_INCLUDE:= -I/path/to/mlpack/build/include
MLPACK_LIB:= -L/path/to/mlpack/build/lib -lmlpack 

ARMA_LIB:= -O2 -larmadillo -DARMA_DONT_USE_WRAPPER -DARMA_USE_BLAS -DARMA_USE_LAPACK -DARMA_USE_HDF5

HDF5_INCLUDE:= -I/usr/include/hdf5/serial
HDF5_LIB:= -L/usr/lib/x86_64-linux-gnu/hdf5/serial \
-lhdf5_hl_cpp -lhdf5_cpp -lhdf5_hl -lhdf5_serial -ldl -laec -lsz -lz

VSAM_INCLUDE:= -I/home/aris/Desktop/Diplwmatikh/ddssim/cpp/

PYTHON_INCLUDE:= $(shell /home/aris/anaconda3/bin/python3-config --includes)
PYTHON_LIB:= $(shell /home/aris/anaconda3/bin/python3-config --ldflags)

INCLUDE = $(PYTHON_INCLUDE) $(HDF5_INCLUDE) $(MLPACK_INCLUDE) $(VSAM_INCLUDE) $(DLIB_INCLUDE)
LIB = $(ARMA_LIB) $(MLPACK_LIB) $(HDF5_LIB) $(DLIB_LIB) -lm -ljsoncpp  #-lboost_filesystem -lboost_system -lcblas -lclapack -llibf2c
CXX = g++-7
DELETE = rm -f
CXXFLAGS = -Wall -std=gnu++17 -O3 $(INCLUDE) # -fPIC

DEBUG_FLAGS = -g3
OPT_FLAGS= -g3 -Ofast #-DNDEBUG
PRODUCTION_FLAGS = -Ofast -DNDEBUG

ifeq ($(MODE),DEBUG)
CXXFLAGS+=  $(DEBUG_FLAGS)
else ifeq ($(MODE),OPT)
CXXFLAGS+= $(OPT_FLAGS)
else ifeq ($(MODE),PRODUCTION)
CXXFLAGS+= $(PRODUCTION_FLAGS)
else
$(error Unknown mode $(MODE))
endif

ifdef GPROF
CXXFLAGS += -pg -no-pie
LDFLAGS += -pg -no-pie
endif

###################################
# File lists
###################################

DDS_SOURCES = ../ddssim/cpp/hdv.cc ../ddssim/cpp/dds.cc ../ddssim/cpp/output.cc ../ddssim/cpp/eca.cc \
	../ddssim/cpp/agms.cc ../ddssim/cpp/data_source.cc ../ddssim/cpp/method.cc ../ddssim/cpp/cfgfile.cc \
	../ddssim/cpp/dsarch.cc ../ddssim/cpp/accurate.cc ../ddssim/cpp/query.cc ../ddssim/cpp/results.cc \
	../ddssim/cpp/sz_quorum.cc ../ddssim/cpp/sz_bilinear.cc ../ddssim/cpp/tods.cc ../ddssim/cpp/safezone.cc \
	../ddssim/cpp/gm_proto.cc ../ddssim/cpp/gm_szone.cc ../ddssim/cpp/gm_query.cc ../ddssim/cpp/fgm.cc \
	../ddssim/cpp/sgm.cc ../ddssim/cpp/frgm.cc

ML_SOURCES = data_structures.cc dsource.cc Machine_Learning.cc ML_GM_Proto.cc \
	ML_GM_Networks.cc ML_FGM_Networks.cc DL_GM_Networks.cc DL_FGM_Networks.cc feeders.cc
DDS_ML_SOURCES = $(DDS_SOURCES) $(ML_SOURCES)
ML_OBJ = $(ML_SOURCES:.cc=.o)

DDS_OBJ = $(DDS_SOURCES:.cc=.o)

###################################
# Rules
###################################

all: write_data write_data2 Classfrs NN_Classfrs Regrsrs NN_Regrsrs Magic

dataset:
	python3 cr_data.py

write_data : wrDataToHDF5.cc data_structures.hh
	$(CXX) $(CXXFLAGS) wrDataToHDF5.cc -o WriteData $(LIB)
	
write_data2 : writeSimpleDataToHDF5.cc data_structures.hh
	$(CXX) $(CXXFLAGS) writeSimpleDataToHDF5.cc -o WriteData2 $(LIB)
	
Classfrs : Classify.cc Classifiers.hh dsource.hh data_structures.hh
	$(CXX) $(CXXFLAGS) Classify.cc -o Classify $(LIB)
	
NN_Classfrs : NN_Classify.cc Classifiers.hh dsource.hh data_structures.hh
	$(CXX) $(CXXFLAGS) NN_Classify.cc -o NN_Classify $(LIB)
	
Regrsrs : Regress.cc Regressors.hh dsource.hh data_structures.hh
	$(CXX) $(CXXFLAGS) Regress.cc -o Regress $(LIB)
	
NN_Regrsrs : NN_Regress.cc Regressors.hh dsource.hh data_structures.hh
	$(CXX) $(CXXFLAGS) NN_Regress.cc -o NN_Regress $(LIB)
		
TensorExample : example.cc 
	$(CXX) $(CXXFLAGS) example.cc -o Tensor $(LIB)
		
Magic: dmllib Main

dmllib: libdml.a 

libdml.a: $(ML_OBJ)
	ar rcs $@ $(DDS_OBJ) $^
	ranlib $@

Main: Main.o libdml.a
	$(CXX) $(LDFLAGS) -o $@ $< -L. -ldml $(LIB)

Test: Main.cc
	$(CXX) $(CXXFLAGS) test.cc -o MatTest $(LIB)

debug: 

clean:
	$(DELETE) $(ML_OBJ)
	$(DELETE) libdml.a
	$(DELETE) Main.o
	$(DELETE) my_test_dataset.hdf5
	$(DELETE) TestFile1.h5
	#$(DELETE) TestFile2.h5
	$(DELETE) WriteData
	$(DELETE) WriteData2
	$(DELETE) Classify
	$(DELETE) NN_Classify
	$(DELETE) Regress
	$(DELETE) NN_Regress
	$(DELETE) Main
	$(DELETE) MatTest
	$(DELETE) Tensor

Write_Data:
	./WriteData
	
Write_Data2:
	./WriteData2
	
Clssfy:
	./Classify

NN_Clssfy:
	./NN_Classify

Regres:
	./Regress
	
NN_Regres:
	./NN_Regress

Run_VSam:
	./Main

Run_Test:
	./MatTest
	
Run_Tensor:
	./Tensor
	
depend: $(DDS_ML_SOURCES)
	$(CXX) $(CXXFLAGS) -MM $^ > .depend

include .depend
