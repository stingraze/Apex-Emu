cmake_minimum_required(VERSION 3.18)
project(gpu-x86-emulator LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(GNUInstallDirs)

# --- Optional CUDA backend ---
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(GPUEMU_CUDA 1)
    message(STATUS "CUDA found: building GPU backend")
else()
    message(STATUS "CUDA not found: CPU-only backend")
endif()

set(CORE_SOURCES
    src/x86/decode.cpp
    src/x86/decode_extended.cpp
    src/translator.cpp
    src/virt/memory.cpp
    src/virt/vcpu.cpp
    src/cpu_executor.cpp
    src/emulator.cpp
    src/linux/syscalls.cpp
    src/guest/elf_loader.cpp
)

set(CORE_HEADERS
    include/x86/types.hpp
    include/x86/state.hpp
    include/x86/decode.hpp
    include/ir/opcode.hpp
    include/translator.hpp
    include/virt/memory.hpp
    include/virt/vcpu.hpp
    include/cpu_executor.hpp
    include/emulator.hpp
    cuda/ir_interpreter.cuh
)

add_library(gpuemu_core STATIC ${CORE_SOURCES} ${CORE_HEADERS})
target_include_directories(gpuemu_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

if(GPUEMU_CUDA)
    target_compile_definitions(gpuemu_core PUBLIC GPUEMU_CUDA)
    set_source_files_properties(cuda/executor.cu PROPERTIES LANGUAGE CUDA)

    add_library(gpuemu_cuda STATIC cuda/executor.cu)
    target_include_directories(gpuemu_cuda PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    target_link_libraries(gpuemu_cuda PUBLIC gpuemu_core)
    set_target_properties(gpuemu_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
    )
endif()

# --- Minimal Linux/BusyBox guest image ---
find_program(GAS_EXECUTABLE as)
find_program(LD_EXECUTABLE ld)
set(GUEST_ELF ${CMAKE_BINARY_DIR}/guest/minlinux.elf)

find_program(GCC_EXECUTABLE gcc)
set(GUEST_CFLAGS -nostdlib -fno-pic -fno-stack-protector -fcf-protection=none
                 -fno-asynchronous-unwind-tables -O0)
set(SIMPLE_OBJ ${CMAKE_BINARY_DIR}/guest/simple.o)
set(PARALLEL_OBJ ${CMAKE_BINARY_DIR}/guest/parallel.o)
set(DEMO_OBJ ${CMAKE_BINARY_DIR}/guest/demo.o)

if(GCC_EXECUTABLE)
    add_custom_command(
        OUTPUT ${SIMPLE_OBJ}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/guest
        COMMAND ${GCC_EXECUTABLE} -c ${GUEST_CFLAGS}
                ${CMAKE_SOURCE_DIR}/guest/simple.c -o ${SIMPLE_OBJ}
        DEPENDS ${CMAKE_SOURCE_DIR}/guest/simple.c
        COMMENT "Compiling guest/simple.c to simple.o"
    )
    add_custom_command(
        OUTPUT ${PARALLEL_OBJ}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/guest
        COMMAND ${GCC_EXECUTABLE} -c ${GUEST_CFLAGS}
                ${CMAKE_SOURCE_DIR}/guest/parallel.c -o ${PARALLEL_OBJ}
        DEPENDS ${CMAKE_SOURCE_DIR}/guest/parallel.c
        COMMENT "Compiling guest/parallel.c to parallel.o"
    )
    add_custom_command(
        OUTPUT ${DEMO_OBJ}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/guest
        COMMAND ${GCC_EXECUTABLE} -c ${GUEST_CFLAGS}
                ${CMAKE_SOURCE_DIR}/guest/demo.c -o ${DEMO_OBJ}
        DEPENDS ${CMAKE_SOURCE_DIR}/guest/demo.c
        COMMENT "Compiling guest/demo.c to demo.o"
    )
    add_custom_target(guest_objects ALL DEPENDS ${SIMPLE_OBJ} ${PARALLEL_OBJ} ${DEMO_OBJ})
endif()

if(GAS_EXECUTABLE AND LD_EXECUTABLE)
    add_custom_command(
        OUTPUT ${GUEST_ELF}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/guest
        COMMAND ${GAS_EXECUTABLE} -64 ${CMAKE_SOURCE_DIR}/guest/minlinux.S
                -o ${CMAKE_BINARY_DIR}/guest/minlinux.o
        COMMAND ${LD_EXECUTABLE} -nostdlib -T ${CMAKE_SOURCE_DIR}/guest/link.ld
                ${CMAKE_BINARY_DIR}/guest/minlinux.o -o ${GUEST_ELF}
        DEPENDS ${CMAKE_SOURCE_DIR}/guest/minlinux.S ${CMAKE_SOURCE_DIR}/guest/link.ld
        COMMENT "Building minimal Linux/BusyBox guest ELF"
    )
    add_custom_target(guest_image ALL DEPENDS ${GUEST_ELF})
else()
    message(WARNING "as/ld not found; guest image will not be built automatically")
endif()

add_executable(decode_test src/decode_test.cpp)
target_link_libraries(decode_test PRIVATE gpuemu_core)

add_executable(gpuemu src/main.cpp)
target_link_libraries(gpuemu PRIVATE gpuemu_core)
if(GPUEMU_CUDA)
    target_link_libraries(gpuemu PRIVATE gpuemu_cuda)
    target_compile_definitions(gpuemu PRIVATE GPUEMU_CUDA)
endif()

if(TARGET guest_image)
    add_dependencies(gpuemu guest_image)
    target_compile_definitions(gpuemu PRIVATE GUEST_ELF_DEFAULT="${GUEST_ELF}")
endif()

if(TARGET guest_objects)
    add_dependencies(gpuemu guest_objects)
    target_compile_definitions(gpuemu PRIVATE SIMPLE_OBJ_DEFAULT="${SIMPLE_OBJ}")
endif()

# Allow CPU executor to include the CUDA header without nvcc
target_compile_options(gpuemu_core PRIVATE -Wno-unused-parameter)
