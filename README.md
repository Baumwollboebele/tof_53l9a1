# 53L9A1_LiveDepth — VL53L9CX live depth streaming (NUCLEO-H533RE)

This application runs the **VL53L9CX 3D ToF LiDAR** on the **X-NUCLEO-53L9A1**
expansion board mounted on a **NUCLEO-H533RE**, and streams the processed depth
map over the ST-Link virtual COM port so a PC can display it **live**.

It is a port of the ST reference example
`Projects/NUCLEO-H563ZI/.../53L9A1_PostprocessSingle`, adapted for the
NUCLEO-H533RE and changed to emit a compact **binary depth stream** (instead of
ASCII art) that the included Python viewer renders as a color image.

---

## 1. Hardware setup

1. Plug the **X-NUCLEO-53L9A1** onto the **NUCLEO-H533RE** Arduino headers.
2. Leave the shield's clock-source switch **SW1 on INT** (sensor internal clock),
   which matches the device descriptor in `Utilities/vl53l9-common/vl53l9/vl53l9_device.c`.
   (The firmware also drives a ~12.5 MHz clock on Arduino D11/PA7 in case SW1 is
   set to the external MCU clock — harmless when SW1 = INT.)
3. Connect the Nucleo's **ST-Link USB** to the PC. This provides both the
   programming interface and the virtual COM port used for the depth stream.

### Pin mapping (why this differs from the H563ZI example)

The shield talks over the Arduino UNO R3 header. On the Nucleo-64 (H533RE) those
header positions land on **different STM32 pins** than on the Nucleo-144
(H563ZI), so the pins were remapped:

| Signal            | Arduino | H563ZI (orig) | **H533RE (this project)** |
|-------------------|---------|---------------|---------------------------|
| I3C1 SCL          | D15     | PB8 (AF3)     | **PB6 (AF3)**             |
| I3C1 SDA          | D14     | PB9 (AF3)     | **PB7 (AF3)**             |
| XSHUT             | D1      | PB6           | **PB14**                  |
| GPIO1 / INTR      | D0      | PB7 → EXTI7   | **PB15 → EXTI15**         |
| Sensor clock (PWM)| D11     | PB5 (TIM3_CH2)| **PA7 (TIM3_CH2, AF2)**   |
| SYNC_IN           | A3      | PB1           | **PB0**                   |
| VCOM UART         | —       | USART3        | **USART2 (PA2/PA3)**      |

The VCOM UART is handled automatically by the Nucleo BSP because the project
defines `USE_NUCLEO_H533RE`.

---

## 2. Resolution / RAM note (important)

The VL53L9CX can output up to **54×42** (binning=2). That full-resolution
transform pipeline needs **~400 KB of heap**, which the STM32H533RE **cannot**
provide (it has **272 KB** of RAM total).

Therefore this project defaults to **binning=4 → 24×20** output
(`CONF_USECASE = VL53L9_USECASE_AF_RANGE` in `Src/vl53l9_app.c`), which fits
comfortably. To experiment with other profiles, change `CONF_USECASE`:

* `VL53L9_USECASE_AF_RANGE`, `VL53L9_USECASE_AF` → binning 4 (24×20) — **fit**
* `VL53L9_USECASE_AR_RANGE`, `VL53L9_USECASE_AR_PRECISION` → binning 2 (54×42) — **will not fit** on H533RE

(The 54×42 profiles run fine on the larger NUCLEO-H563ZI.)

---

## 3. Build & flash (CMake + arm-none-eabi-gcc)

No IDE required. You need `arm-none-eabi-gcc`, `cmake` and a generator
(`ninja` or `make`) on your PATH. All three ship with **STM32CubeCLT**
(`GNU-tools-for-STM32`, `CMake`, `Ninja` folders) — add those `bin` dirs to
PATH, or install the Arm GNU toolchain + CMake yourself.

From this project directory
(`Projects/NUCLEO-H533RE/Applications/53L9A1/53L9A1_LiveDepth`):

```
cmake -B build -G Ninja -DCMAKE_TOOLCHAIN_FILE=cmake/gcc-arm-none-eabi.cmake -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Outputs land in `build/`:
* `53L9A1_LiveDepth.elf` — for debugging
* `53L9A1_LiveDepth.hex` / `.bin` — for flashing
* `53L9A1_LiveDepth.map` — link map

Reference build result: **FLASH ~16 % (84 KB / 512 KB), RAM ~52 % (144 KB /
272 KB)** — the RAM figure includes the 128 KB heap reserved for the transform
pipeline.

Flash it with the ST CLI (also part of STM32CubeCLT):

```
STM32_Programmer_CLI -c port=SWD -w build/53L9A1_LiveDepth.hex -rst
```

Notes:
* If `arm-none-eabi-gcc` is not on PATH, pass its prefix explicitly:
  `-DTOOLCHAIN_PREFIX=C:/ST/STM32CubeCLT_x.y.z/GNU-tools-for-STM32/bin/arm-none-eabi-`
* Use `-G "Unix Makefiles"` (or `"MinGW Makefiles"`) instead of `-G Ninja` if
  you prefer `make`.
* The build reuses the shared `Drivers/`, `Middlewares/` and `Utilities/`
  sources from the package, plus the H533 startup
  (`startup_stm32h533retx.s`) and linker script (`STM32H533RETX_FLASH.ld`).

A ready-made **STM32CubeIDE** project is also included under `STM32CubeIDE/`
(open it via *File → Open Projects from File System…*) if you ever want the GUI.

### Flash & run with pyOCD

One-time: install the STM32H5 device pack so pyOCD knows the target:

```
pyocd pack install stm32h533retx
```

Then, with the board connected via ST-Link USB, flash and run (pyOCD resets the
MCU after programming, so it starts streaming immediately):

```
pyocd flash --target stm32h533retx build/53L9A1_LiveDepth.hex
```

A `pyocd.yaml` in this folder already sets `target_override: stm32h533retx`, so
from here you can also just run `pyocd flash build/53L9A1_LiveDepth.hex`,
`pyocd reset`, etc. without repeating `--target`. Check the probe is seen with
`pyocd list`.

### VS Code tasks

`.vscode/tasks.json` provides ready-made tasks (Terminal → Run Task…):

* **Build** — configure (first time) + `cmake --build build`  (default build task, `Ctrl+Shift+B`)
* **Flash & Run (pyOCD)** — builds then flashes `build/53L9A1_LiveDepth.hex` (default test task)
* **pyOCD: Install H533 pack (one-time)** — runs `pyocd pack install stm32h533retx`
* **Reset & Run (pyOCD)** — resets the board without reflashing
* **Live view (Python)** — launches `tools/live_view.py`

`.vscode/launch.json` adds a **Debug (pyOCD)** configuration (needs the
*Cortex-Debug* extension and `arm-none-eabi-gdb`, which ships with STM32CubeCLT).

Open **this folder** (`53L9A1_LiveDepth`) as the VS Code workspace so the task
paths resolve. `arm-none-eabi-gcc`, `cmake`, `ninja` and `pyocd` must be on PATH
(STM32CubeCLT provides the first three; `pip install pyocd` for the last).

---

## 4. Run the live viewer (PC) — SWD path (WORKING on this board)

On this board the ST-Link **VCOM UART does not carry data** (TX never reaches the
host at any baud). So the working transport reads each depth frame **directly
out of the target's RAM over the SWD debug link** with pyOCD — no UART involved.
The firmware publishes every frame into a fixed global mailbox `g_live`.

```
cd tools
pip install -r requirements.txt          # pyocd, numpy, matplotlib
python swd_view.py                        # auto-finds g_live from the .elf
```

You get a live color depth image. Options: `--min/--max` (mm color scale),
`--cmap`, `--addr 0x........` (skip the ELF symbol lookup).

### Desktop app (recommended): `tof_gui.py`

```
python tof_gui.py                       # YOLO detector (needs ultralytics)
python tof_gui.py --detector heuristic  # no ML deps
```

A PySide6/Qt application that puts everything in one window:
* crisp upscaled image (OpenCV colormap) on the left;
* grouped controls on the right — **Display** (channel, colormap, size, max
  distance, **min intensity**, **Sobel-edge overlay**), **Sensor**
  (resolution/use case, frame rate), **Object detection** (detect on/off, show
  only detected, confidence, class filter, **edge-refine mask**);
* **Min intensity** hides pixels whose **amplitude** (signal/confidence) is below
  a threshold — so weak, unreliable returns are removed from every channel view
  (including distance). The threshold is adaptive (a percentage of the frame's
  99th-percentile amplitude) and the actual cutoff value is shown next to it;
* **Processing** group — **Temporal** (per-pixel median over the last N frames,
  ignoring invalid pixels: strong at removing flying pixels + noise) and
  **Edge-preserving smooth** (bilateral filter — denoises while keeping depth
  edges sharp). Both are applied to all channels before display/detection;
* an **SNR** view (amplitude ÷ ambient = signal / noise-floor) is selectable in
  the channel list as a confidence map;
* a **3D point cloud** checkbox opens a rotatable pop-out window — each pixel +
  its distance + the sensor field-of-view becomes an XYZ point, coloured by
  distance. Needs `pip install pyqtgraph PyOpenGL`; set the horizontal FoV with
  `--fov` (vertical is derived from the aspect ratio, so it's approximate);
* a **Motion / presence** group: **Learn background** snapshots the current depth
  as a reference, **Background subtract** then flags pixels that differ from it
  (foreground, tinted green), **Show only foreground** blanks the rest, and the
  Info panel reports **presence** (yes/no + foreground %) and a **motion** metric
  (mean depth change between frames);
* **Sobel edges:** depth discontinuities = object boundaries, so a Sobel
  gradient-magnitude + Otsu binarization (computed on the *upscaled* depth for
  crisp lines) can be overlaid, and can **refine the heuristic mask** — it cuts
  the foreground at strong depth edges, which cleans boundaries and separates
  adjacent objects at different depths (verified: two touching objects split
  into just the nearest one). Utilities live in `segmenter.py`
  (`sobel_magnitude`, `sobel_binary`, `DepthBlobSegmenter(edge_refine=True)`);
* live **stats** (resolution, fps, temperature, dropped, detected objects) and
  **pixel hover** read-out (distance / amplitude / ambient under the cursor).

Sensor settings are only re-applied on release / actual change, so dragging a
slider doesn't thrash the sensor. Detection is throttled so the view stays
smooth. Needs `pip install PySide6` (and `ultralytics` for YOLO).

### Crisp OpenCV viewer: `tof_cv.py`

```
python tof_cv.py                       # YOLO detector (needs ultralytics)
python tof_cv.py --detector heuristic  # no ML deps
```

An OpenCV window that **smoothly upscales** the low-res ToF frame (cubic) with a
colormap for a clean, crisp image — much nicer than the raw 54×42 blocks. All
controls are trackbars in the "controls" window: **channel**, **use case /
resolution**, **fps**, output **size**, **dmax** (distance scale), **colormap**,
**detect** (object detection on/off), **show only** (keep only detected regions),
and **conf%**. Object detection runs on the *upscaled* image, which segments
better than the tiny raw frame. Press **q** or **Esc** to quit.

### matplotlib viewer + control panel: `tof_app.py`

```
python tof_app.py
```

A richer app (also SWD-based) that shows the **raw sensor channels** and lets you
tune the sensor live:
* **Channel**: Distance (mm) / Amplitude (signal) / Ambient.
* **Resolution / use case**: full **54×42** (range or precision) or **24×20**
  (range or fast). Full 54×42 is the chip's native resolution and is the default.
* **Frame rate** slider.
* **Hover** any pixel to read its exact distance / amplitude / ambient.
* **Detect objects** checkbox — runs an object detector and draws boxes + labels
  (+ segmentation masks). Multi-class.
* **Show only detected** checkbox — blanks everything outside the detected
  objects (keeps only the detected regions of the image).
* Live stats: resolution, measured fps, temperature, dropped frames, detected objects.

### Object-detection backends (`tools/detector.py`)

Pluggable `Detector` returning `Detection(label, score, box, mask)`:

* **`yolo`** (default) — Ultralytics YOLO on the upscaled amplitude image, COCO
  classes, boxes + masks (with a `-seg` model). Needs `pip install ultralytics`
  (pulls in torch). Weights auto-download on first use; that first detection can
  take a while. Enable with the checkbox or run `--detector yolo`.
  Options: `--model yolov8n-seg.pt`, `--conf 0.25`, `--classes 0` (e.g. person only).
* **`heuristic`** — no-dependency depth foreground blob (from `segmenter.py`),
  returned as one detection. Use `--detector heuristic`.

Add your own model by subclassing `Detector` and returning it from
`make_detector()`.

> **Reality check:** YOLO is trained on RGB photos; a 54×42 ToF amplitude map
> upscaled to the model input is out of that domain, so DL detections will be
> sparse/unreliable. The detection→overlay→show-only plumbing is model-agnostic,
> so the practical path to good results is to fine-tune a small detector on ToF
> data and drop it into `detector.py`. For a robust "where is the nearest thing"
> right now, `--detector heuristic` works without any model.

The firmware publishes distance + amplitude + ambient (up to 54×42) into `g_live`
and applies settings written to `g_ctrl` — both over SWD, no UART. Full resolution
works because this build reads the **raw** frame directly (the heavy transform
pipeline, which did not fit in RAM at 54×42, is not used).

> If the app shows "no frame yet (power-cycle the board?)", the sensor is stuck
> in a running state from a previous session (its power stays on across MCU
> resets, and the board's XSHUT reset line is mis-wired). **Unplug the USB for a
> few seconds and plug it back in** — a cold boot resets the sensor cleanly.

Requirements: the board connected via ST-Link, and **no other debugger holding
the probe** (close STM32CubeIDE debug sessions / gdbservers first). `swd_view.py`
attaches without halting or resetting, so it does not disturb the running app.

Quick sanity check without the GUI:
```
pyocd commander -t stm32h533retx -c "read32 0x20000050 32"
# 0x20000050 should start with 4556494c ('LIVE'); the frames/seq words climb.
```

### Notes on what was wrong on this board (and how it's handled)

The X-NUCLEO-53L9A1's non-I3C Arduino signals land on different STM32 pins on
this Nucleo-64 than on the reference NUCLEO-H563ZI, so several things needed
workarounds (all already applied in the firmware):
* **Clock**: the HSE-from-ST-Link assumption hangs the H533RE; the firmware runs
  the PLL from HSI instead (still 250 MHz).
* **Reset (XSHUT)**: the XSHUT toggle does not reset the sensor here, so the app
  detects an already-running sensor and recovers in software.
* **Data-ready (INT) pin**: not wired to a usable pin, so the app **polls** the
  sensor's FRAME_READY register over I3C instead of using the EXTI.
* **Acquisition**: uses a simple synchronous read (the async DMA path raced).
* **UART**: unused; frames go out over SWD (above).

## 5. (Legacy) UART viewer — does NOT work on this board

Requires Python 3.

```
cd Projects/NUCLEO-H533RE/Applications/53L9A1/53L9A1_LiveDepth/tools
pip install -r requirements.txt
python live_view.py                 # auto-detects the ST-Link COM port
# or specify explicitly:
python live_view.py --port COM7               # Windows
python live_view.py --port /dev/ttyACM0       # Linux
```

Options: `--baud` (default **921600**, must match the firmware),
`--min/--max` (color scale in mm, default 0–2000), `--cmap` (default `turbo`).

Close the plot window to quit. Invalid pixels (no target) are shown black.

> The firmware VCOM baud is set in `Src/main.c` (`STREAM_BAUDRATE`, 921600). If
> you change it there, pass the same value to `live_view.py --baud`.

---

## 5. Stream protocol

One little-endian packet per processed frame:

| offset | type       | field         |
|--------|------------|---------------|
| 0      | `char[4]`  | magic `L9DF`  |
| 4      | `uint16`   | width         |
| 6      | `uint16`   | height        |
| 8      | `uint16`   | format (0 = uint16 depth in mm) |
| 10     | `uint32`   | frame_counter |
| 14     | `uint16[width*height]` | depth in mm (0 = invalid/no target) |

The viewer resynchronizes on the `L9DF` magic, so it can be started/stopped at
any time.
