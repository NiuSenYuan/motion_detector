cmake_minimum_required(VERSION 3.10)
project(MotionDetection)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

add_executable(MotionDetection main.cpp motion_detector.cpp)

target_link_libraries(MotionDetection ${OpenCV_LIBS})

