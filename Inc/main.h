/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.h
  * @brief          : Header for main.c file.
  *                   This file contains the common defines of the application.
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */

/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef __MAIN_H
#define __MAIN_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "stm32h5xx_hal.h"

#include "stm32h5xx_nucleo.h"
#include <stdio.h>

#include "stm32h5xx_ll_bus.h"
#include "stm32h5xx_ll_cortex.h"
#include "stm32h5xx_ll_rcc.h"
#include "stm32h5xx_ll_system.h"
#include "stm32h5xx_ll_utils.h"
#include "stm32h5xx_ll_pwr.h"
#include "stm32h5xx_ll_gpio.h"
#include "stm32h5xx_ll_dma.h"
#include "stm32h5xx_ll_exti.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

/* USER CODE END Includes */

/* Exported types ------------------------------------------------------------*/
/* USER CODE BEGIN ET */

/* USER CODE END ET */

/* Exported constants --------------------------------------------------------*/
/* USER CODE BEGIN EC */

/* USER CODE END EC */

/* Exported macro ------------------------------------------------------------*/
/* USER CODE BEGIN EM */

/* USER CODE END EM */

void HAL_TIM_MspPostInit(TIM_HandleTypeDef *htim);

/* Exported functions prototypes ---------------------------------------------*/
void Error_Handler(void);

/* USER CODE BEGIN EFP */

/* USER CODE END EFP */

/* Private defines -----------------------------------------------------------*/
/*
 * X-NUCLEO-53L9A1 signal mapping on NUCLEO-H533RE (Nucleo-64, MB1814).
 *
 * The shield connects through the Arduino UNO R3 header. The Arduino header
 * positions carry different STM32 pins than on the NUCLEO-H563ZI, so the pins
 * below differ from the H563ZI reference project:
 *
 *   Signal            Arduino   H533RE pin   Function
 *   ----------------  --------  -----------  ---------------------------------
 *   I3C1_SCL          D15       PB6 (AF3)    sensor I3C clock   (set in MSP)
 *   I3C1_SDA          D14       PB7 (AF3)    sensor I3C data    (set in MSP)
 *   XSHUT             D1        PB14         sensor shutdown (active low)
 *   GPIO1 / INTR      D0        PB15 -> EXTI15  data-ready interrupt (falling)
 *   sensor clock/PWM  D11       PA7 (TIM3_CH2, AF2) ~12.5 MHz  (set in MSP)
 *   SYNC_IN           A3        PB0          sensor sync input (driven high)
 */
#define XSHUT_Pin              GPIO_PIN_14
#define XSHUT_GPIO_Port        GPIOB

#define INTR_Pin               GPIO_PIN_15
#define INTR_GPIO_Port         GPIOB
#define INTR_EXTI_IRQn         EXTI15_IRQn

#define SYNC_IN_Pin            GPIO_PIN_0
#define SYNC_IN_GPIO_Port      GPIOB

/* Optional debug/profiling outputs (not wired to the shield) */
#define DEBUG_GPIO_1_Pin       GPIO_PIN_0
#define DEBUG_GPIO_1_GPIO_Port GPIOC
#define DEBUG_GPIO_2_Pin       GPIO_PIN_1
#define DEBUG_GPIO_2_GPIO_Port GPIOC

/* USER CODE BEGIN Private defines */

/* USER CODE END Private defines */

#ifdef __cplusplus
}
#endif

#endif /* __MAIN_H */
