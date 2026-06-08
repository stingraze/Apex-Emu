# GPU x86-64 Emulator

A CUDA-accelerated x86-64 (long mode) emulator with a virtualization layer that translates guest instructions into GPU-friendly IR for parallel execution.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Host (CPU)                                                 │
│  ┌──────────────┐   ┌────────────────┐   ┌──────────────┐ │
│  │ x86 Decoder  │──▶│ IR Translator  │──▶│ Virt Layer   │ │
│  └──────────────┘   └────────────────┘   │ (traps, MMIO)│ │
│                                           └──────┬───────┘ │
└──────────────────────────────────────────────────┼─────────┘
                                                   │
┌──────────────────────────────────────────────────┼─────────┐
│  Device (GPU)                                    ▼         │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  vcpu_run_kernel — one CUDA thread per virtual CPU   │  │
│  │  Each thread executes translated IR in parallel        │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────┘
```

### Layers

1. **Decode** (`src/x86/decode.cpp`) — Parses x86-64 machine code into structured instructions (ModR/M, REX, basic blocks).

2. **Translate** (`src/translator.cpp`) — Lowers x86 ops to 16-byte GPU IR records (`ir::Instr`). Straight-line sequences run efficiently; branches become `Jmp`/`Jcc` with resolved IR indices.

3. **Virtualize** (`src/virt/`) — Guest physical memory, per-vCPU state, syscall traps dispatched to host.

4. **Execute** (`cuda/executor.cu`, `cuda/ir_interpreter.cuh`) — CUDA kernel runs one thread per vCPU. All vCPUs share the same translated IR but have independent register files and memory views.

### Parallelism model

| Mode | Description |
|------|-------------|
| **Multi-vCPU** | N independent x86 guests run simultaneously — one CUDA thread each. This is the primary GPU speedup path. |
| **IR batching** | Instructions within a basic block are sequential per thread, but thousands of threads execute concurrently. |
| **Future** | Speculative multi-block tracing, JIT superblocks, warp-cooperative memory ops. |

## Build

### Requirements

- CMake ≥ 3.18
- C++17 compiler (g++ or clang++)
- CUDA Toolkit ≥ 11.0 (optional; CPU fallback works without GPU)

### Compile

```bash
mkdir build && cd build
cmake ..
cmake --build . -j
```

Without CUDA, the emulator runs on CPU using the same IR interpreter.

<<<<<<< Updated upstream
8 vCPU test (from ~/alpha/gpu-cpu-emulator):
```
./build/gpuemu -p 8 guest/parallel.o
```
And its output (in text)
```
GPU x86-64 Emulator — Compiled C Guest
======================================
Loading compiled guest: build/guest/println.o
vCPUs: 8

Backend: CUDA GPU | entry=0x10017b | IR ops=151

=== println guest (compiled C) ====== println guest (compiled C) ====== println guest (compiled C) ====== println guest (compiled C) ====== println guest (compiled C) ====== println guest (compiled C) ====== println guest (compiled C) ====== println guest (compiled C) ===







a = 17, b = 25a = 17, b = 25a = 17, b = 25a = 17, b = 25a = 17, b = 25a = 17, b = 25a = 17, b = 25a = 17, b = 25







a + b =a + b =a + b =a + b =a + b =a + b =a + b =a + b =







42







a * b (loop) =a * b (loop) =a * b (loop) =a * b (loop) =a * b (loop) =a * b (loop) =a * b (loop) =a * b (loop) =







425







Done.Done.Done.Done.Done.Done.Done.Done.








--- 8 guest(s) finished (exit_code=42, halted=1) ---

```
### Run the Mandelbrot vCPU demo (up to 32vCPU
```
cmake -S . -B build && cmake --build build -j
./build/gpuemu --mandel -p 8
```

=======
### Run the Linux / BusyBox demo (Untested)

```bash
cd build
cmake --build .
./gpuemu
```

This boots a **minimal bare-metal Linux userland** guest (`guest/minlinux.S`) that simulates:

- Kernel boot messages
- BusyBox `/bin/sh` init
- `echo` and `uname -a` applets (via `write` syscalls)
- Clean `exit`

Custom guest path:

```bash
./gpuemu linux /path/to/minlinux.elf
```

The guest ELF is built automatically by CMake (`as` + `ld` with `guest/link.ld`).

## Supported x86-64 subset

- `mov` (reg↔reg, reg↔mem, reg←imm)
- `add`, `sub`, `and`, `or`, `xor` (reg/mem forms)
- `cmp`, `test`
- `push`, `pop`
- `lea`
- `jmp`, conditional jumps (`jcc`)
- `nop`, `hlt`, `syscall`

## Project layout

```
include/          Public headers (x86, IR, virt, emulator)
src/              Host implementation
cuda/             GPU IR interpreter + CUDA kernels
```

How it works today
CMake compiles every file under guest/ with gcc (-nostdlib -O0, etc.) into build/guest/*.o:

Guest	Object	How to run
parallel.c
parallel.o
```
./build/gpuemu --guest parallel -p 4
simple.c
simple.o
./build/gpuemu --guest simple
```
println.c
println.o
```
./build/gpuemu (default)
```

demo.c
demo.o
```
./build/gpuemu --guest demo
```

mandel.c
mandel.o
```
./build/gpuemu --mandel -p 8
```
gpubench.S
gpubench.o
```
./build/gpuemu --bench -p 1024
```

You can also pass any .o path directly:

```
./build/gpuemu build/guest/parallel.o -p 4
```
Important: -p / --parallel means number of vCPUs, not “run parallel.c”. That’s why it felt like only --mandel was wired up — mandel is the only guest with a dedicated mode; the rest use the generic loader.

New CLI helpers
Added:
```
--list-guests — shows all built-in guests and example commands
--guest NAME — picks a guest by name (parallel, simple, println, demo, mandel, gpubench)
cmake --build build
./build/gpuemu --list-guests
./build/gpuemu --guest parallel -p 4
./build/gpuemu --mandel -p 8
Adding your own guest .c
```

Put guest/myprog.c in the repo with a _start() entry (same pattern as simple.c / parallel.c).
Add a add_custom_command block in CMakeLists.txt (copy the parallel.o rule).
Append it to GUEST_OBJ_LIST.
Run it: ./build/gpuemu build/guest/myprog.o -p N
Or extend the kGuests[] table in main.cpp if you want a --guest myprog alias.

Why mandel “just works” but println is flaky
--mandel uses extra host setup: row bands in RCX/RDX, per-vCPU memory snapshots, and ASCII framebuffer printing.
Generic guests (run_guest) only set RDI = vcpu_id and run. That’s enough for parallel.c and simple.c (inline asm + syscalls).
println.c / demo.c use C stack locals and division loops for number printing — those hit known IR decoder gaps (same class of bugs we worked around in mandelbrot). Mandelbrot works because it avoids those patterns.
So: parallel and simple work now; println/demo need more IR support (or asm-style syscalls like parallel.c) before they’re reliable.

## Extending

- **More instructions**: Add cases to `Decoder::decode_one` and `Translator::emit_x86`.
- **Self-modifying code**: Re-translate on write to code pages (virt layer hook).
- **Multi-block scheduling**: Use `vcpu_batch_kernel` with per-vCPU IR regions.
- **Linux guest**: Expand syscall handler in `VirtualCpu::handle_syscall`.

## License

MIT

