<!-- SPDX-License-Identifier: GPL-3.0-or-later -->
<!-- Copyright 2026 the ScarletNX contributors -->
# The Mali calibration run

`run.py` times the GT 1030 corpus (`../gt1030/spv`, `../gt1030/manifest.json`) on a Mali GPU of
an Android phone and fits ucas's cost classes from it, the way `../gt1030/run.py` does on a GT
1030.  The corpus maths, the interleaved runs and the fit are the GT 1030 script's, imported
from it; only the transport differs: every run happens on the device, over adb.

## The device binary

`snx-shaderperf` links only Vulkan's headers and opens the loader at run time, so the NDK
builds it for the phone as it is:

```sh
NDK=$HOME/Android/android-ndk-r28c
$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/clang \
  --target=aarch64-linux-android28 -O2 -std=c99 \
  -I .cache/cpm/vulkan-headers/3208/include -I tools/shaderperf \
  tools/shaderperf/main.c tools/shaderperf/gpu.c tools/shaderperf/reflect.c \
  tools/shaderperf/stats.c tools/shaderperf/texfmt.c \
  -lm -ldl -o snx-shaderperf-arm64
adb push snx-shaderperf-arm64 /data/local/tmp/snx-shaderperf
adb shell chmod 755 /data/local/tmp/snx-shaderperf
```

## The run

```sh
tests/mali-g715/run.py --push --out tests/mali-g715/results/$(date +%F)
```

Close every game first: the phone's compositor shares the GPU with the shaders being timed,
and the fit is a difference of two timings from the same run, so a busy GPU widens the
intervals rather than shifting them.

## What the device reports

The Mali driver has `VK_KHR_pipeline_executable_properties` and no
`VK_KHR_performance_query` and no `pipelineStatisticsQuery`, so `snx-shaderperf` gives the
stage's Arithmetic, FMA, CVT, SFU, Load Store and register counts but no invocations a
repetition: it counts the dispatch's invocations instead.  Subgroups are 16 lanes on Valhall,
so a "warp" is 16 lanes when cycles are read from the timings.

Latency builds are skipped: they read the SM clock through `KHR_shader_clock` and dump the
buffer, which needs a pull per dump.  The GT 1030 model's `load`, `store`, `branch` and
`diverge` classes come from throughput tests alone here.
