cmake_minimum_required(VERSION 3.2)
project(Process)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Werror")

set(SOURCE_FILES main.c cluster.h clustering.h item.h input.txt)
add_executable(Process ${SOURCE_FILES})