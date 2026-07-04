# ============================================================================
#  53L9A1_LiveDepth  -  NUCLEO-H533RE + X-NUCLEO-53L9A1 (VL53L9CX)
#
#  IDE-free build using CMake + arm-none-eabi-gcc.
#
#  Configure & build (Ninja):
#     cmake -B build -G Ninja \
#           -DCMAKE_TOOLCHAIN_FILE=cmake/gcc-arm-none-eabi.cmake \
#           -DCMAKE_BUILD_TYPE=Release
#     cmake --build build
#
#  Output: build/53L9A1_LiveDepth.elf  (+ .bin, .hex)
#  Flash:  STM32_Programmer_CLI -c port=SWD -w build/53L9A1_LiveDepth.hex -rst
# ============================================================================
cmake_minimum_required(VERSION 3.20)

project(53L9A1_LiveDepth C ASM)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ---- repo layout ----------------------------------------------------------
# This project lives at Projects/NUCLEO-H533RE/Applications/53L9A1/53L9A1_LiveDepth
# so the package root is five directories up.
get_filename_component(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../../.." ABSOLUTE)
set(PROJ_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

set(TARGET 53L9A1_LiveDepth)
set(LINKER_SCRIPT "${PROJ_DIR}/STM32CubeIDE/STM32H533RETX_FLASH.ld")

# ---- MCU / architecture flags --------------------------------------------
set(MCU_FLAGS
    -mcpu=cortex-m33
    -mfpu=fpv5-sp-d16
    -mfloat-abi=hard
    -mthumb)

add_compile_definitions(
    USE_HAL_DRIVER
    USE_FULL_LL_DRIVER
    USE_NUCLEO_H533RE
    STM32H533xx)

add_compile_options(
    ${MCU_FLAGS}
    -Wall
    -ffunction-sections
    -fdata-sections
    -fno-common
    $<$<CONFIG:Debug>:-Og>
    $<$<CONFIG:Debug>:-g3>
    $<$<CONFIG:Release>:-O2>
    $<$<CONFIG:Release>:-g0>)

# ---- include paths --------------------------------------------------------
include_directories(
    ${PROJ_DIR}/Inc
    ${REPO_ROOT}/Drivers/STM32H5xx_HAL_Driver/Inc
    ${REPO_ROOT}/Drivers/STM32H5xx_HAL_Driver/Inc/Legacy
    ${REPO_ROOT}/Drivers/CMSIS/Device/ST/STM32H5xx/Include
    ${REPO_ROOT}/Drivers/CMSIS/Include
    ${REPO_ROOT}/Drivers/BSP/STM32H5xx_Nucleo
    ${REPO_ROOT}/Drivers/BSP/Components/vl53l9
    ${REPO_ROOT}/Utilities/vl53l9-common
    ${REPO_ROOT}/Utilities/vl53l9-common/vl53l9
    ${REPO_ROOT}/Middlewares/ST/media-object/media-c/header-lib/include/public
    ${REPO_ROOT}/Middlewares/ST/media-object/transform-c/header-lib/include/public
    ${REPO_ROOT}/Middlewares/ST/vl53l9-transform-c/vl53l9-transform-c-lib/include/private
    ${REPO_ROOT}/Middlewares/ST/vl53l9-transform-c/vl53l9-transform-c-lib/include/public)

# ---- sources --------------------------------------------------------------
set(HAL_DIR ${REPO_ROOT}/Drivers/STM32H5xx_HAL_Driver/Src)
set(HAL_SOURCES
    ${HAL_DIR}/stm32h5xx_hal.c
    ${HAL_DIR}/stm32h5xx_hal_cortex.c
    ${HAL_DIR}/stm32h5xx_hal_dma.c
    ${HAL_DIR}/stm32h5xx_hal_dma_ex.c
    ${HAL_DIR}/stm32h5xx_hal_exti.c
    ${HAL_DIR}/stm32h5xx_hal_flash.c
    ${HAL_DIR}/stm32h5xx_hal_flash_ex.c
    ${HAL_DIR}/stm32h5xx_hal_gpio.c
    ${HAL_DIR}/stm32h5xx_hal_i3c.c
    ${HAL_DIR}/stm32h5xx_hal_icache.c
    ${HAL_DIR}/stm32h5xx_hal_pcd.c
    ${HAL_DIR}/stm32h5xx_hal_pcd_ex.c
    ${HAL_DIR}/stm32h5xx_hal_pwr.c
    ${HAL_DIR}/stm32h5xx_hal_pwr_ex.c
    ${HAL_DIR}/stm32h5xx_hal_rcc.c
    ${HAL_DIR}/stm32h5xx_hal_rcc_ex.c
    ${HAL_DIR}/stm32h5xx_hal_tim.c
    ${HAL_DIR}/stm32h5xx_hal_tim_ex.c
    ${HAL_DIR}/stm32h5xx_hal_uart.c
    ${HAL_DIR}/stm32h5xx_hal_uart_ex.c
    ${HAL_DIR}/stm32h5xx_hal_usart.c
    ${HAL_DIR}/stm32h5xx_hal_usart_ex.c
    ${HAL_DIR}/stm32h5xx_ll_dma.c
    ${HAL_DIR}/stm32h5xx_ll_exti.c
    ${HAL_DIR}/stm32h5xx_ll_gpio.c
    ${HAL_DIR}/stm32h5xx_ll_rcc.c
    ${HAL_DIR}/stm32h5xx_ll_ucpd.c
    ${HAL_DIR}/stm32h5xx_ll_usb.c
    ${HAL_DIR}/stm32h5xx_ll_utils.c
    ${HAL_DIR}/stm32h5xx_util_i3c.c)

set(BSP_SOURCES
    ${REPO_ROOT}/Drivers/BSP/Components/vl53l9/vl53l9.c
    ${REPO_ROOT}/Drivers/BSP/STM32H5xx_Nucleo/stm32h5xx_nucleo.c)

file(GLOB TRANSFORM_SOURCES
    ${REPO_ROOT}/Middlewares/ST/vl53l9-transform-c/vl53l9-transform-c-lib/src/*.c
    ${REPO_ROOT}/Middlewares/ST/vl53l9-transform-c/vl53l9-transform-c-lib/src/algo/*.c)

set(COMMON_SOURCES
    ${REPO_ROOT}/Utilities/vl53l9-common/platform/platform_utils.c
    ${REPO_ROOT}/Utilities/vl53l9-common/vl53l9/vl53l9_device.c
    ${REPO_ROOT}/Utilities/vl53l9-common/vl53l9/vl53l9_platform.c
    ${REPO_ROOT}/Utilities/vl53l9-common/vl53l9/vl53l9_utils.c)

set(APP_SOURCES
    ${PROJ_DIR}/Src/main.c
    ${PROJ_DIR}/Src/stm32h5xx_hal_msp.c
    ${PROJ_DIR}/Src/stm32h5xx_it.c
    ${PROJ_DIR}/Src/vl53l9_app.c
    ${PROJ_DIR}/Src/system_stm32h5xx.c
    ${PROJ_DIR}/STM32CubeIDE/Application/User/syscalls.c
    ${PROJ_DIR}/STM32CubeIDE/Application/User/sysmem.c)

set(STARTUP ${PROJ_DIR}/STM32CubeIDE/Application/Startup/startup_stm32h533retx.s)
set_property(SOURCE ${STARTUP} PROPERTY LANGUAGE ASM)

add_executable(${TARGET}
    ${APP_SOURCES}
    ${COMMON_SOURCES}
    ${BSP_SOURCES}
    ${TRANSFORM_SOURCES}
    ${HAL_SOURCES}
    ${STARTUP})

set_target_properties(${TARGET} PROPERTIES SUFFIX ".elf")

# ---- link -----------------------------------------------------------------
target_link_options(${TARGET} PRIVATE
    ${MCU_FLAGS}
    -T${LINKER_SCRIPT}
    --specs=nano.specs
    -Wl,--gc-sections
    -Wl,-Map=${TARGET}.map,--cref
    -Wl,--print-memory-usage
    -static
    -Wl,--start-group -lc -lm -Wl,--end-group)

# ---- post-build: size + .hex/.bin ----------------------------------------
add_custom_command(TARGET ${TARGET} POST_BUILD
    COMMAND ${CMAKE_SIZE} $<TARGET_FILE:${TARGET}>
    COMMAND ${CMAKE_OBJCOPY} -O ihex   $<TARGET_FILE:${TARGET}> ${TARGET}.hex
    COMMAND ${CMAKE_OBJCOPY} -O binary $<TARGET_FILE:${TARGET}> ${TARGET}.bin
    COMMENT "Generating ${TARGET}.hex and ${TARGET}.bin")
