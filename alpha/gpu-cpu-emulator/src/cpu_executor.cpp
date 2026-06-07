#include "cpu_executor.hpp"
#include "cuda/ir_interpreter.cuh"
#include <cstring>

namespace cpuexec {

int run_ir(x86::CpuState& cpu, uint8_t* mem, uint64_t mem_size,
           const ir::Instr* code, uint32_t code_len,
           uint32_t& pc, uint32_t max_steps) {
    return gpuexec::exec_ir(cpu, mem, mem_size, code, code_len, pc, max_steps);
}

}  // namespace cpuexec
