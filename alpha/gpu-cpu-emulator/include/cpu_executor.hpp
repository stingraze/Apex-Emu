#pragma once

#include "ir/opcode.hpp"
#include "x86/state.hpp"
#include <cstdint>

namespace cpuexec {

// CPU fallback: same IR interpreter, for dev machines without CUDA.
int run_ir(x86::CpuState& cpu, uint8_t* mem, uint64_t mem_size,
           const ir::Instr* code, uint32_t code_len,
           uint32_t& pc, uint32_t max_steps);

}  // namespace cpuexec
