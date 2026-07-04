# Bare-metal ARM cross toolchain file for arm-none-eabi-gcc.
#
# Assumes arm-none-eabi-gcc is on PATH (e.g. from STM32CubeCLT's
# GNU-tools-for-STM32, or the Arm GNU toolchain). Override the prefix with
#   -DTOOLCHAIN_PREFIX=/path/to/arm-none-eabi-
# if it is not on PATH.

set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR arm)

if(NOT DEFINED TOOLCHAIN_PREFIX)
    set(TOOLCHAIN_PREFIX "arm-none-eabi-")
endif()

# On Windows the executables have a .exe suffix; let CMake find them.
set(CMAKE_C_COMPILER   ${TOOLCHAIN_PREFIX}gcc)
set(CMAKE_ASM_COMPILER ${TOOLCHAIN_PREFIX}gcc)
set(CMAKE_CXX_COMPILER ${TOOLCHAIN_PREFIX}g++)
set(CMAKE_OBJCOPY      ${TOOLCHAIN_PREFIX}objcopy CACHE INTERNAL "")
set(CMAKE_SIZE         ${TOOLCHAIN_PREFIX}size    CACHE INTERNAL "")

# Do not try to link a full executable during the compiler check (there is no
# startup/linker script available at that point); a static library is enough.
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
