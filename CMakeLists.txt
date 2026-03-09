cmake_minimum_required(VERSION 3.10)
project(VisionApp)

find_package(OpenMP REQUIRED)

find_package(OpenCV REQUIRED)

add_executable(main main.cpp)

if(OpenMP_CXX_FOUND)
    target_link_libraries(main PUBLIC OpenMP::OpenMP_CXX)
endif()

target_link_libraries(main PUBLIC ${OpenCV_LIBS})