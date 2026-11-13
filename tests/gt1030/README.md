<!-- SPDX-License-Identifier: GPL-3.0-or-later -->
# GT 1030 calibration corpus

SPIR-V shaders that measure what each ucas cost class costs on the real GPU, an NVIDIA GeForce GT 1030
(GP108, sm_61, 3 SMs, warp 32), so that the prelude's `gt1030` model (src/parse.c) can be checked and fitted.
They are timed with ScarletNX's `snx-shaderperf` (`~/ScarletNX/tools/shaderperf`).

`make test` does not run these: they need the GPU.

## Files

- `gen.py` holds every test (a table) and writes `src/` (GLSL), `spv/` (SPIR-V, built with glslangValidator
  without optimisation and checked by spirv-val) and `manifest.json`. Edit the table, then run `./gen.py`.
- `run.py` times the corpus and fits the costs. It writes `results.json`, `report.md`, `model.ucas` (a
  `model gt1030_measured { ... }` block) and `costs.json` to `~/ucas/work/gt1030/run-<time>/`.
- `results/` keeps the report of a run, for reference.

## Method

A test is one step, an expression of the chain values, written out UNROLL times inside a loop of
`pc.iters` iterations. Every test is built twice, `NAME.a.spv` with UNROLL = A and `NAME.b.spv` with
UNROLL = 2A, so `t(b) - t(a)` is the time of A x C steps a lane alone: the loop, the prologue, the store and
the dispatch cancel. `run.py` runs `fadd.a fadd.b NAME.a NAME.b` in one snx-shaderperf invocation
(interleaved samples, same push constants), and a step's cost is its time over an fadd's from the same run.
That is the unit of the ucas model: issue cost per warp instruction, fadd = 1.

- C independent chains (8 for ALU tests) give each warp instruction-level parallelism, and 65536
  invocations fill the SMs, so latency is hidden and the ratio is throughput.
- Chain j reads chains j+1, j+2, j+3 (a ring, updated in order). The driver's compiler reassociates
  even float adds of loop invariants (`a + kf` 128 times became one multiply-add), but it cannot shorten a
  ring. Inputs come from a noise buffer and push constants, so nothing is a compile-time constant.
- The `Binary Size` the driver reports (VK_KHR_pipeline_executable_properties) gives the bytes of machine
  code a step (`B/step`; sm_61 packs 3 instructions and a control word in 32 bytes, ~10.7 B an
  instruction). It shows fusion (LOP3, ISCADD, XMAD, FFMA) and emulation (division, fp64, MATCH) without the
  disassembly, which the driver does not give.
- Latency: tests with `.la/.lb` builds run one warp (`--size 32`) with one serial chain, read the SM clock
  around the loop (`clockARB`, KHR_shader_clock) and store it; `run.py` reads it back with `--dump`. The result
  is SM clocks a step, free of the clock speed.
- `probe_sm.spv` records each warp's `gl_SMIDNV`, `gl_WarpIDNV`, `gl_SMCountNV`, `gl_WarpsPerSMNV`
  (NV_shader_sm_builtins) and clocks, for the SM count and the occupancy.

The fit: every class that a primary test (`fit=True`) uses is an unknown, fadd is 1, and least squares over
the primary tests gives the costs. Every other test is a check: priced by the fitted costs and by the prelude,
next to what it measured. Classes the ucas model does not have are named `x_...` (DP4A, shuffles, MATCH,
shared loads, texture fetches...): each gets a cost from its test, ready for a model that grows them.

## Coverage

- 32-bit integer: add, sub, logic (and the LOP3 fusions), shifts (variable, constant, ISCADD), rotates,
  multiply (and IMAD, multiply-high, by constants and powers of two), division and remainder (variable,
  constant, power of two), compare + select, select, min/max/clamp, bitCount, findMSB/LSB, bitfieldReverse,
  bitfieldExtract/Insert.
- Conversions (f32<->int, floor/fract/roundEven, f32->f16->f32, f32<->f64), bitcasts, 16-bit and 8-bit
  integers (shaderInt16/Int8: no narrow ALU on sm_61).
- 32-bit float: add, mul, fma, contracted and `precise` multiply-add, free negate/abs modifiers, min/max/clamp,
  compare + select, mix, length, division (variable, reciprocal, constant), sqrt, inversesqrt, exp2/log2,
  exp/log, sin/cos/tan, asin/acos/atan/atan2, sinh/tanh, pow.
- 64-bit: f64 add/mul/fma/div/sqrt (1/32 rate), u64 add/mul/shift.
- NVIDIA and other extensions: DP4A through SPV_KHR_integer_dot_product (`spirv_instruction` in GLSL),
  subgroup reduce/scan/clustered/shuffle/shuffle-xor/broadcast/quad/ballot/vote/rotate,
  NV_shader_subgroup_partitioned (partition = MATCH, partitioned add), KHR_shader_clock (SM and device
  clocks), NV_shader_sm_builtins, EXT_shader_atomic_float (global and shared float atomics),
  KHR_shader_atomic_int64, KHR_shader_expect_assume.
- Memory: coalesced DRAM loads, cache-resident loads, broadcast loads, uniform buffer (uniform and divergent
  addresses), shared memory (and 32-way bank conflicts), stores, atomics (global with and without the result,
  contended, compare-and-swap, f32, u64; shared, contended, f32), barriers (128 and 1024 invocations,
  subgroup barrier), texel fetch, bilinear rgba8 and rgba32f, image loads.
- Control: uniform and divergent branches (arms of 2 and 8 FFMAs, `[[dont_flatten]]`), a flattened branch,
  an expect hint.
- Graphics stages, ALU: 22 of the steps above (fadd, fmul, fma, iadd, xor, shift, imul, udiv, udiv by a
  constant, compare-select, min, bitCount, conversions, fdiv, sqrt, exp2, sin, pow, f64 add, DP4A, subgroup add
  and shuffle) again in a vertex, a geometry and a fragment shader (`vert_*`, `geom_*`, `frag_*`), each against
  the fadd of its own stage; derivatives (dFdx, dFdyFine, fwidth) and implicit-LOD sampling in the fragment
  stage.
- Graphics stages, I/O (`io` in the manifest): the cost of what a stage reads and writes, one thing varied a
  group. Vertex: varyings written (1-16 vec4), attributes read (1-8 vec4), the attribute format (rgba32f,
  rgba16f, rgba16 unorm, rgba8 unorm/snorm, a2bgr10, b10g11r11), indexing (none, sequential, random). Geometry:
  points emitted a point (1-32), vec4 outputs a point (1-8), invocations against emitted vertices. Fragment:
  inputs interpolated (1-8 vec4), render target format (8 formats), blending (none, alpha, add on rgba8,
  rgba16f, rgba32f), MSAA (1-8 samples), render targets (1-8), depth attachments (none, d16, d24s8, d32).

## Formats and fixed interfaces

The interface of a stage costs time apart from its ALU, so each ALU test keeps it fixed and the A/2A
difference cancels it: the vertex shader reads one vec4 attribute as rgba32_sfloat and writes gl_Position and
one vec4, the geometry shader reads points (the built-in vertex shader's vec4) and emits one point, both into
rasterizer discard; the fragment shader reads one vec4 of the full-screen triangle (1920x1080) and writes one
rgba8_unorm target without blending. The I/O groups then vary one of these at a time. Groups whose items differ
only in the shader ("run") share one snx-shaderperf run; items that need other options (formats, blending,
samples, other vertex inputs or render targets: "sweep") are runs of their own, so their ratios carry the clock
drift between runs. The report gives each item in ns a vertex, input primitive or pixel, as a ratio to the
group's first item, and in fadd steps of its stage.

## Running

```sh
cd ~/ucas/tests/gt1030
./gen.py                       # after editing the table
./run.py                       # ~12 minutes; --only a,b to run some tests or I/O groups, --runs N,
                               # --no-io (ALU only), --io-only (the I/O groups only)
./run.py --analyse ~/ucas/work/gt1030/run-...   # re-fit a finished run
```

Notes:

- Xorg shares this GPU, and the SM clock moves (1.2 to 1.9 GHz). Ratios inside one run are steady; do not
  compare times of two runs.
- The graphics tests need snx-shaderperf with geometry shaders and `--attr` (ScarletNX branch `shaderperf`).
- snx-shaderperf enables extensions given with `--ext` but not their feature structs: the float atomics run
  with VK_EXT_shader_atomic_float enabled and its features left at their defaults.
- shaderFloat16 is false on this GPU, so there are no f16 arithmetic tests, only conversions.

## The probes ucas waits for

`probe.py` is one run of what is still unmeasured (the gt1030 issues #115-#122, #144, #145, #147, #156, #166,
#168, #184-#188, #207-#212): each probe takes its issue's method, a step against fadd as above, a draw or a
dispatch timed whole, or values read back (the filter's weight grid and precision, the blender's and the UNORM
conversion's rounding, the BC4 palette, REPEAT at large coordinates, vulkaninfo's subTexelPrecisionBits). It
writes `probe.json`, `report.md` (by issue) and `model.ucas`, a `model gt1030_probed : gt1030 { ... }` with
the prices and parameters the probes set and the other measures as comments; ucas reads that file back at the
end. What snx-shaderperf cannot set up yet (logic ops, stencil operations, MIN/MAX and advanced blending) is
named in the report with what it needs.

```sh
cd ~/ucas/tests/gt1030
./probe.py                     # every probe; --issue 184,186 or --only mul16_mad for some, --list for all
./probe.py --quick             # a smoke run on a small grid: it goes through, the numbers mean nothing
./test_probe.py                # the checks that need no GPU (about 30 s)
ucas ~/ucas/work/gt1030/probe-.../model.ucas    # then `use gt1030_probed`
```

It needs glslangValidator, spirv-val, snx-shaderperf (`--sp`, as run.py) and vulkaninfo for #117. A draw or
a dispatch timed whole is put in fadd steps by a fadd timed in a run of its own just before it, and every run's
SM clock is in the report.

`test_probe.py` checks what needs no GPU: no step is a subgroup operation with a uniform result alone (it
folds), clamps keep their bounds ordered, every subgroup operation reads a chain value, the walked tables keep
a warp's lanes on entries of their own, the filter analysis reads a weight grid and its rounding from synthetic
answers, units convert by their own fadd, and every step compiles and passes spirv-val.
