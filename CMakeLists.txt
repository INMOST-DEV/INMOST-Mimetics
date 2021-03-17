cmake_minimum_required(VERSION 2.8)
project(Mimetics)

find_package(inmost REQUIRED)

link_directories(${INMOST_LIBRARY_DIRS})
include_directories(${INMOST_INCLUDE_DIRS})
add_definitions(${INMOST_DEFINITIONS})

add_subdirectory(ADMFD)
add_subdirectory(MFD-ES)

