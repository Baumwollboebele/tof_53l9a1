/**
 ******************************************************************************
 * @file    vl53l9_app.c
 * @brief   VL53L9CX raw multi-channel acquisition with runtime-tunable settings.
 *
 *          Adapted for NUCLEO-H533RE. Reads the RAW sensor frame (distance,
 *          amplitude, ambient) directly - no transform pipeline - so it fits in
 *          RAM even at full resolution (54x42, binning=2). Frames and sensor
 *          state are published into a RAM mailbox (g_live) that the PC reads
 *          live over SWD; the PC changes settings by writing g_ctrl over SWD.
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

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "vl53l9.h"
#include "vl53l9_device.h"
#include "vl53l9_interface.h"
#include "vl53l9_utils.h"

#define CONF_DEVICE_ID (0)

/* Maximum raw resolution is binning=2 -> 54 x 42. */
#define MAX_W  54
#define MAX_H  42
#define MAX_PX (MAX_W * MAX_H)
#define RAW_BUF_BYTES 16384 /* >= raw buffer size for binning=2 */

#define FSM_STANDBY   2
#define FSM_STREAMING 3

/* SWD mailbox: frames + sensor state published to the PC. */
typedef struct {
    uint32_t magic; /* 'LIVE' = 0x4556494C */
    uint32_t heartbeat;
    uint32_t fsm;              /* 1=ready_to_boot 2=standby 3=streaming */
    uint32_t frames;
    uint32_t seq;              /* bumped on every new frame */
    uint32_t dropped;
    uint32_t applied_seq;      /* echoes g_ctrl.cmd_seq once applied */
    uint32_t usecase;          /* active use case (0..3) */
    uint32_t binning;          /* active binning (2,4,...) */
    uint32_t frame_period_us;  /* active frame period */
    uint32_t exposure_ms;      /* active exposure */
    uint32_t temperature;      /* last frame temperature */
    uint32_t frame_counter;    /* last sensor frame counter */
    uint16_t width;
    uint16_t height;
    uint16_t distance[MAX_PX];  /* mm */
    uint16_t amplitude[MAX_PX]; /* signal / intensity */
    uint16_t ambient[MAX_PX];   /* ambient light */
} live_mailbox_t;

/* SWD control block: PC writes these, then bumps cmd_seq to request apply. */
typedef struct {
    volatile uint32_t cmd_seq;         /* bump to apply the settings below */
    volatile uint32_t usecase;         /* 0=AR_RANGE 1=AR_PRECISION (both 54x42),
                                          2=AF_RANGE 3=AF (both 24x20) */
    volatile uint32_t frame_period_us; /* 0 = use the use case default */
    volatile uint32_t exposure_ms;     /* 0 = use the use case default */
} live_ctrl_t;

volatile live_mailbox_t g_live __attribute__((used));
/* default: use case 1 (AR_PRECISION) = full resolution 54x42 */
volatile live_ctrl_t g_ctrl __attribute__((used)) = { .cmd_seq = 1, .usecase = 1,
                                                       .frame_period_us = 0, .exposure_ms = 0 };

static uint8_t g_raw_buffer[RAW_BUF_BYTES];

static void handle_error(void);

/* raw frame dimensions for a given binning (matches the driver's raw layout) */
static void raw_dims(uint8_t binning, uint16_t *w, uint16_t *h) {
    switch (binning) {
    case 2:  *w = 54; *h = 42; break;
    case 4:  *w = 24; *h = 24; break;
    case 6:  *w = 18; *h = 14; break;
    case 8:  *w = 12; *h = 10; break;
    case 12: *w = 8;  *h = 8;  break;
    case 24: *w = 4;  *h = 4;  break;
    default: *w = 0;  *h = 0;  break;
    }
}

/* Bring a (possibly streaming) sensor to STANDBY. A pending unread frame keeps
 * the sensor from stopping, so a frame is drained (ack) before STOP_STREAM.
 * Returns the last observed FSM state. */
static uint8_t to_standby(vl53l9_device_t *p_dev) {
    vl53l9_status_t st = { 0 };
    for (uint32_t i = 0; i < 3000u; i++) {
        vl53l9_get_status(p_dev, &st);
        g_live.fsm = st.fsm;
        if (st.fsm != FSM_STREAMING) {
            break; /* standby or ready_to_boot - nothing more to stop */
        }
        uint8_t rdy = 0;
        if (vl53l9_poll_frame(p_dev, &rdy) == 0 && rdy) {
            (void)vl53l9_get_frame(p_dev, NULL, 0); /* ack pending frame */
        }
        (void)vl53l9_stop(p_dev);
    }
    return st.fsm;
}

/* Configure the sensor for a use case and start streaming. The sensor must be
 * in STANDBY. Fills the raw frame size / dimensions for the selected binning. */
static int configure_start(vl53l9_device_t *p_dev, uint32_t usecase, uint32_t frame_period_us,
                           uint32_t exposure_ms, uint16_t *p_raw_size, uint16_t *p_w, uint16_t *p_h) {
    int ret;
    if (usecase >= VL53L9_NB_USECASES) {
        usecase = 0;
    }

    vl53l9_profile_t profile = g_ranging_profiles[usecase];
    if (frame_period_us) {
        profile.frame_period_us = frame_period_us;
    }
    if (exposure_ms) {
        profile.exposure_ms = exposure_ms;
    }

    ret = vl53l9_utils_set_profile(p_dev, &profile);
    if (ret) {
        return ret;
    }

    vl53l9_get_raw_buffer_size(profile.binning, p_raw_size);
    raw_dims(profile.binning, p_w, p_h);

    g_live.usecase = usecase;
    g_live.binning = profile.binning;
    g_live.frame_period_us = profile.frame_period_us;
    g_live.exposure_ms = profile.exposure_ms;

    ret = vl53l9_set_sync_mode(p_dev, VL53L9_SYNC_AUTONOMOUS);
    if (ret) {
        return ret;
    }
    return vl53l9_start(p_dev);
}

void vl53l9_app(void) {
    int ret;
    vl53l9_device_t *p_dev = &device[CONF_DEVICE_ID];

    g_live.magic = 0x4556494CU;

    /* sensor reset + I3C dynamic addressing */
    platform_power_reset(CONF_DEVICE_ID);
    if (p_dev->bus_type & PLATFORM_BUS_I3C) {
        platform_assign_dynamic_address();
    }

    /*
     * Bring-up. On a cold sensor it comes up in READY_TO_BOOT and we do a full
     * init. If it is already running (XSHUT reset is ineffective on this board)
     * we bring it to STANDBY instead and skip the boot.
     */
    vl53l9_status_t st = { 0 };
    vl53l9_get_status(p_dev, &st);
    g_live.fsm = st.fsm;
    if (st.fsm == FSM_STREAMING) {
        to_standby(p_dev);
        vl53l9_get_status(p_dev, &st);
        g_live.fsm = st.fsm;
    }
    if (st.fsm != FSM_STANDBY) { /* READY_TO_BOOT (or unknown) -> full init */
        ret = vl53l9_init(p_dev);
        if (ret) {
            handle_error();
        }
    }

    /* Read calibration data once. We don't use it (no transform pipeline) but
     * this call also performs the clock switching the sensor needs to range. */
    static uint8_t calib_data[VL53L9_CALIB_DATA_SIZE];
    (void)vl53l9_get_calib_data(p_dev, calib_data);

    /* initial configuration (sensor is in STANDBY here) */
    uint16_t raw_size = 0, cur_w = 0, cur_h = 0;
    ret = configure_start(p_dev, g_ctrl.usecase, g_ctrl.frame_period_us, g_ctrl.exposure_ms,
                          &raw_size, &cur_w, &cur_h);
    if (ret) {
        handle_error();
    }
    uint32_t applied_seq = g_ctrl.cmd_seq;
    g_live.applied_seq = applied_seq;

    while (1) {

        /* apply a settings change requested by the PC */
        if (g_ctrl.cmd_seq != applied_seq) {
            to_standby(p_dev);
            ret = configure_start(p_dev, g_ctrl.usecase, g_ctrl.frame_period_us, g_ctrl.exposure_ms,
                                  &raw_size, &cur_w, &cur_h);
            if (ret) {
                handle_error();
            }
            applied_seq = g_ctrl.cmd_seq;
            g_live.applied_seq = applied_seq;
        }

        /* wait for a frame (poll FRAME_READY over I3C); also react to new commands */
        uint8_t ready = 0;
        do {
            ret = vl53l9_poll_frame(p_dev, &ready);
            if (ret) {
                handle_error();
            }
            g_live.heartbeat++;
            if (g_ctrl.cmd_seq != applied_seq) {
                break; /* a reconfiguration was requested */
            }
        } while (!ready);
        if (!ready) {
            continue;
        }

        /* synchronous raw read + acknowledge */
        ret = vl53l9_get_frame(p_dev, g_raw_buffer, raw_size);
        if (ret) {
            g_live.dropped++;
            continue;
        }
        (void)vl53l9_get_frame(p_dev, NULL, 0); /* ack */

        vl53l9_frame_t fr = { 0 };
        if (vl53l9_utils_parse_frame(g_raw_buffer, raw_size, &fr)) {
            g_live.dropped++;
            continue;
        }

        uint32_t px = (uint32_t)cur_w * (uint32_t)cur_h;
        if (px == 0 || px > MAX_PX) {
            g_live.dropped++;
            continue;
        }

        for (uint32_t i = 0; i < px; i++) {
            g_live.distance[i] = (uint16_t)fr.p_distance[i].value; /* 15-bit mm */
            g_live.amplitude[i] = fr.p_amplitude[i];
            g_live.ambient[i] = fr.p_ambient[i];
        }
        g_live.width = cur_w;
        g_live.height = cur_h;
        g_live.temperature = fr.p_metadata->temperature;
        g_live.frame_counter = fr.p_metadata->frame_counter;
        g_live.frames++;
        g_live.seq++;
    }
}

static void handle_error(void) {
    vl53l9_status_t status = { 0 };
    vl53l9_get_status(&device[CONF_DEVICE_ID], &status);
    while (1)
        ;
}
