cmake_minimum_required(VERSION 3.3)

SET(OpenCV_STATIC OFF)
FIND_PACKAGE(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

project(FaceDet)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(SOURCE_FILES main.cpp adaboost.h adaboost.cpp features.h features.cpp image_tools.h image_tools.cpp cascadeReader.h cascadeReader.cpp adaboost_detector.cpp adaboost_detector.h)
add_executable(FaceDet ${SOURCE_FILES})
TARGET_LINK_LIBRARIES(FaceDet ${OpenCV_LIBS})

# TARGET_LINK_LIBRARIES(images ${OpenCV_LIBS})
