cmake_minimum_required(VERSION 3.0.0)
project(engine2 VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 20)



file(GLOB TARGET_SRC "./src/*")

add_executable(main ${TARGET_SRC})

