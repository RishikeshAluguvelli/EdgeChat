cmake_minimum_required(VERSION 3.22)
project(edgechat CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# llama.cpp (b10988) is checked out by scripts/setup-llama-android.sh into android/third_party/llama.cpp.
set(LLAMA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../third_party/llama.cpp")
if (NOT EXISTS "${LLAMA_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "llama.cpp source not found at ${LLAMA_DIR}. Run scripts/setup-llama-android.sh first.")
endif()

# Library-only build: no tools/tests/examples, no curl, static ggml + llama linked into our JNI library.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
set(LLAMA_SUBPROCESS OFF CACHE BOOL "" FORCE)
set(MTMD_VIDEO OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
# arm64-v8a: runtime detection picks dotprod / i8mm / SVE kernels on phones that have them.
if (ANDROID_ABI STREQUAL "arm64-v8a")
    set(GGML_CPU_ARM_ARCH "armv8.2-a+dotprod+fp16" CACHE STRING "" FORCE)
endif()

add_subdirectory("${LLAMA_DIR}" llama_build EXCLUDE_FROM_ALL)
add_subdirectory("${LLAMA_DIR}/tools/mtmd" mtmd_build EXCLUDE_FROM_ALL)

add_library(edgechat SHARED
    edgechat_engine.cpp
    jni.cpp)

target_include_directories(edgechat PRIVATE
    "${LLAMA_DIR}/include"
    "${LLAMA_DIR}/ggml/include"
    "${LLAMA_DIR}/tools/mtmd")

target_compile_options(edgechat PRIVATE -O2 -fexceptions -frtti)
target_link_libraries(edgechat PRIVATE llama ggml mtmd android log)
