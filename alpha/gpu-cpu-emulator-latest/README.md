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

### Run the Linux / BusyBox demo

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

## Extending

- **More instructions**: Add cases to `Decoder::decode_one` and `Translator::emit_x86`.
- **Self-modifying code**: Re-translate on write to code pages (virt layer hook).
- **Multi-block scheduling**: Use `vcpu_batch_kernel` with per-vCPU IR regions.
- **Linux guest**: Expand syscall handler in `VirtualCpu::handle_syscall`.

## License

MIT
