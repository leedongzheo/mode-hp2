cmake_minimum_required(VERSION 3.16...3.26)

# 1. Tên project chuẩn
project(kiss_icp_pybind VERSION 1.2.3 LANGUAGES CXX)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(PYBIND11_NEWPYTHON ON)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# 2. [QUAN TRỌNG] Bổ sung logic tìm Pybind11 thông minh (Copy từ GenZ sang)
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
  # Nếu không tìm thấy theo cách thường, hỏi Python xem nó nằm ở đâu
  execute_process(COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
  list(APPEND CMAKE_PREFIX_PATH ${pybind11_DIR})
  # Tìm lại lần nữa
  find_package(pybind11 CONFIG REQUIRED)
endif()

# 3. Logic tìm Source C++ (Viết gọn lại giống GenZ cho đẹp)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/kiss_icp/)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../cpp/kiss_icp ${CMAKE_CURRENT_BINARY_DIR}/kiss_icp)
else()
  # Báo lỗi nếu không thấy folder code C++
  message(FATAL_ERROR "Could not find local KISS-ICP C++ source at ../cpp/kiss_icp. Please check directory structure.")
endif()

# 4. Add thư mục binding
add_subdirectory(kiss_icp/pybind)
