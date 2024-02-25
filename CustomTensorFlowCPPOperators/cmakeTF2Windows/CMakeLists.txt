#############################################################################################

CMAKE_MINIMUM_REQUIRED(VERSION 3.9)
MESSAGE("++++ Required CMake Version: 3.9")

#############################################################################################
#CUDA 

SET (TENSORFLOW_PATH $ENV{TF_DIR})
SET (OPENCVPATH $ENV{OpenCV_Dir})
SET (CMAKE_CUDA_ARCHITECTURES 61)

#############################################################################################
#Project 

SET(PROJECT_NAME CustomTensorFlowOperators)
PROJECT(${PROJECT_NAME} CXX CUDA)
MESSAGE("++++ Project name: " ${PROJECT_NAME})

#############################################################################################
#Misc

SET(TARGET ${PROJECT_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)

SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
SET( CMAKE_EXE_LINKER_FLAGS  "-ltensorflow_framework" )
SET (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
SET (BUILD_SHARED_LIBS TRUE)

SET(projectDir "")
STRING(REPLACE "/cmakeTF2Windows" "" projectDir ${CMAKE_SOURCE_DIR})

MESSAGE("++++ Project directory: " ${projectDir})

#############################################################################################
#otherss

MESSAGE("++++ Prepare includes and libs")
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/../)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/../thirdParty/Shared/cutil/inc/)

#############################################################################################
#opencv

MESSAGE("++++ OpenCV lib: " ${OPENCVPATH})
SET (opencvLib ${OPENCVPATH}/x64/vc15/lib/opencv_world3416.lib)
INCLUDE_DIRECTORIES(${OPENCVPATH}/include/)

#############################################################################################
#tensorflow

MESSAGE("++++ Tensorflow lib")
SET (tensorflowLib ${TENSORFLOW_PATH}/python/_pywrap_tensorflow_internal.lib)
INCLUDE_DIRECTORIES(${TENSORFLOW_PATH}/include/)
INCLUDE_DIRECTORIES(${TENSORFLOW_PATH}/include/external/eigen_archive/)
INCLUDE_DIRECTORIES(${TENSORFLOW_PATH}/include/external/com_google_absl/)
INCLUDE_DIRECTORIES(${TENSORFLOW_PATH}/include/external/protobuf_archive/src/)

##############################################################################################
#ADD FILES

MESSAGE("++++ Add source files")

FILE(GLOB_RECURSE cppFiles 			
	${CMAKE_SOURCE_DIR}/../src/CudaUtils/*.cpp
	${CMAKE_SOURCE_DIR}/../src/CudaUtils/*.h

	${CMAKE_SOURCE_DIR}/../src/CustomCPUOperators/ForwardKinematicsOperator/*.cpp
	${CMAKE_SOURCE_DIR}/../src/CustomCPUOperators/ForwardKinematicsOperator/*.h
	${CMAKE_SOURCE_DIR}/../src/CustomCPUOperators/Pose2EmbeddedGraphOperator/*.cpp
	${CMAKE_SOURCE_DIR}/../src/CustomCPUOperators/Pose2EmbeddedGraphOperator/*.h
	
	${CMAKE_SOURCE_DIR}/../src/CustomGPUOperators/PerspectiveNPointOperator/*.cpp
	${CMAKE_SOURCE_DIR}/../src/CustomGPUOperators/PerspectiveNPointOperator/*.h
	${CMAKE_SOURCE_DIR}/../src/CustomGPUOperators/EmbeddedGraphOperator/*.cpp
	${CMAKE_SOURCE_DIR}/../src/CustomGPUOperators/EmbeddedGraphOperator/*.h
	${CMAKE_SOURCE_DIR}/../src/CustomGPUOperators/EmbeddedGraphARAPOperator/*.cpp
	${CMAKE_SOURCE_DIR}/../src/CustomGPUOperators/EmbeddedGraphARAPOperator/*.h
	${CMAKE_SOURCE_DIR}/../src/CustomGPUOperators/CameraProjectionOperator/*.cpp
	${CMAKE_SOURCE_DIR}/../src/CustomGPUOperators/CameraProjectionOperator/*.h
	${CMAKE_SOURCE_DIR}/../src/CustomGPUOperators/DQSkinningOperator/*.cpp
	${CMAKE_SOURCE_DIR}/../src/CustomGPUOperators/DQSkinningOperator/*.h
	${CMAKE_SOURCE_DIR}/../src/CustomGPUOperators/LBSkinningOperator/*.cpp
	${CMAKE_SOURCE_DIR}/../src/CustomGPUOperators/LBSkinningOperator/*.h
	${CMAKE_SOURCE_DIR}/../src/CustomGPUOperators/MultiViewSilhouetteLossOperator/*.cpp
	${CMAKE_SOURCE_DIR}/../src/CustomGPUOperators/MultiViewSilhouetteLossOperator/*.h
	${CMAKE_SOURCE_DIR}/../src/CustomGPUOperators/ProjectedMeshBoundaryOperator/*.cpp
	${CMAKE_SOURCE_DIR}/../src/CustomGPUOperators/ProjectedMeshBoundaryOperator/*.h
	${CMAKE_SOURCE_DIR}/../src/CustomGPUOperators/GlobalToUVSpaceOperator/*.cpp
	${CMAKE_SOURCE_DIR}/../src/CustomGPUOperators/GlobalToUVSpaceOperator/*.h

	${CMAKE_SOURCE_DIR}/../src/Lock/*.cpp
	${CMAKE_SOURCE_DIR}/../src/Lock/*.h

	${CMAKE_SOURCE_DIR}/../src/Math/*.cpp
	${CMAKE_SOURCE_DIR}/../src/Math/*.h

	${CMAKE_SOURCE_DIR}/../src/Rendering/*.cpp
	${CMAKE_SOURCE_DIR}/../src/Rendering/*.h

	${CMAKE_SOURCE_DIR}/../src/Skeletool/Camera/*.cpp
	${CMAKE_SOURCE_DIR}/../src/Skeletool/Camera/*.h
	${CMAKE_SOURCE_DIR}/../src/Skeletool/Character/*.cpp
	${CMAKE_SOURCE_DIR}/../src/Skeletool/Character/*.h
	${CMAKE_SOURCE_DIR}/../src/Skeletool/Color/*.cpp
	${CMAKE_SOURCE_DIR}/../src/Skeletool/Color/*.h
	${CMAKE_SOURCE_DIR}/../src/Skeletool/ImageVideoProcessing/*.cpp
	${CMAKE_SOURCE_DIR}/../src/Skeletool/ImageVideoProcessing/*.h
	${CMAKE_SOURCE_DIR}/../src/Skeletool/Mesh/*.cpp
	${CMAKE_SOURCE_DIR}/../src/Skeletool/Mesh/*.h
	${CMAKE_SOURCE_DIR}/../src/Skeletool/Parameters/*.cpp
	${CMAKE_SOURCE_DIR}/../src/Skeletool/Parameters/*.h

	${CMAKE_SOURCE_DIR}/../src/StringProcessing/*.cpp
	${CMAKE_SOURCE_DIR}/../src/StringProcessing/*.h
)

FILE(GLOB cudaFiles
	${CMAKE_SOURCE_DIR}/../src/CustomGPUOperators/PerspectiveNPointOperator/*.cu 	
	${CMAKE_SOURCE_DIR}/../src/CustomGPUOperators/EmbeddedGraphOperator/*.cu		
	${CMAKE_SOURCE_DIR}/../src/CustomGPUOperators/EmbeddedGraphARAPOperator/*.cu	
	${CMAKE_SOURCE_DIR}/../src/CustomGPUOperators/CameraProjectionOperator/*.cu
	${CMAKE_SOURCE_DIR}/../src/CustomGPUOperators/DQSkinningOperator/*.cu
	${CMAKE_SOURCE_DIR}/../src/CustomGPUOperators/LBSkinningOperator/*.cu
	${CMAKE_SOURCE_DIR}/../src/CustomGPUOperators/MultiViewSilhouetteLossOperator/*.cu
	${CMAKE_SOURCE_DIR}/../src/CustomGPUOperators/ProjectedMeshBoundaryOperator/*.cu
	${CMAKE_SOURCE_DIR}/../src/CustomGPUOperators/GlobalToUVSpaceOperator/*.cu
	${CMAKE_SOURCE_DIR}/../src/Skeletool/Character/*.cu
	${CMAKE_SOURCE_DIR}/../src/Skeletool/Color/*.cu
	${CMAKE_SOURCE_DIR}/../src/Skeletool/Camera/*.cu
	${CMAKE_SOURCE_DIR}/../src/Skeletool/ImageVideoProcessing/*.cu
	${CMAKE_SOURCE_DIR}/../src/Skeletool/Mesh/*.cu
	${CMAKE_SOURCE_DIR}/../src/Skeletool/Parameters/*.cu
)

##############################################################################################
#CUDA COMPILATION

MESSAGE("++++ Set CUDA compilation properties")

SET(CUDA_NVCC_FLAGS "-O3" "--default-stream per-thread" "-Xptxas -v" CACHE STRING "nvcc flags" FORCE)

##############################################################################################
#OUTPUT LIB

MESSAGE("++++ Set output lib properties")

ADD_DEFINITIONS(-DWIN32 -D_WINDOWS -DNDEBUG -DGOOGLE_CUDA -D_USE_MATH_DEFINES)

SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${projectDir}/binaries/Win64")
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${projectDir}/binaries/Win64")
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${projectDir}/binaries/Win64")

SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")

ADD_LIBRARY(${PROJECT_NAME} SHARED ${cppFiles} ${cudaFiles})

##############################################################################################
#ADD STRUCTURE IN SOLUTION

MESSAGE("++++ Add structure to the visual studio solution")
FILE( GLOB_RECURSE SRCS REGULAR_EXPRESSION "${CMAKE_SOURCE_DIR}/../src/*")


FOREACH(FILE ${SRCS}) 
    # Get the directory of the source file
    GET_FILENAME_COMPONENT(PARENT_DIR "${FILE}" DIRECTORY)

    # Remove common directory prefix to make the group
    STRING(REPLACE "${CMAKE_SOURCE_DIR}/../src/" "" GROUP "${PARENT_DIR}")

    # Make sure we are using windows slashes
    STRING(REPLACE "/" "\\" GROUP "${GROUP}")

    SOURCE_GROUP("${GROUP}" FILES "${FILE}")
endforeach()

##############################################################################################
#OUTPUT LIB

MESSAGE("++++ Specifiy the output")

TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${tensorflowLib} ${opencvLib} ${CUDA_LIBRARIES} ${CUDA_curand_LIBRARY} ${tensorflowLib})




