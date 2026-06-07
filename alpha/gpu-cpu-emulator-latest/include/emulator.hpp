#pragma once

#include "cpu_executor.hpp"
#include "ir/opcode.hpp"
#include "translator.hpp"
#include "virt/memory.hpp"
#include "virt/vcpu.hpp"
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

namespace gpuemu {

struct FlatIr {
    std::vector<ir::Instr> code;
    uint64_t entry_pc = 0;
    std::unordered_map<uint64_t, uint32_t> pc_to_idx;

    uint32_t ir_index_for(uint64_t guest_pc) const {
        const auto it = pc_to_idx.find(guest_pc);
        return it != pc_to_idx.end() ? it->second : 0;
    }
};

// Flatten translated basic blocks into a single IR stream with resolved jump indices.
FlatIr flatten_ir(const ir::TranslationUnit& tu);

struct RunResult {
    bool halted = false;
    bool syscall_trap = false;
    int64_t exit_code = 0;
    uint32_t steps = 0;
};

class Emulator {
public:
    Emulator();

    // Load guest machine code at guest address.
    void load_code(uint64_t guest_addr, const uint8_t* bytes, size_t len);

    // Load a static ELF or relocatable .o guest into guest RAM.
    bool load_guest(const char* path, uint64_t& entry_out);

    // Add a virtual CPU; returns vCPU index.
    uint32_t add_vcpu(uint64_t entry_pc, const x86::CpuState* initial = nullptr);

    // Translate loaded code to GPU IR.
    void translate(uint64_t entry_pc);

    // Run all vCPUs (parallel on GPU when available).
    RunResult run(uint32_t max_steps_per_vcpu = 1 << 20);

    virt::GuestMemory& memory() { return mem_; }
    const FlatIr& ir() const { return flat_; }
    bool using_gpu() const { return use_gpu_; }

    virt::VirtualCpu& vcpu(uint32_t idx) { return *vcpus_.at(idx); }

private:
    void run_cpu(uint32_t max_steps);
    void run_gpu(uint32_t max_steps);
    void handle_traps();

    virt::GuestMemory mem_;
    std::vector<std::unique_ptr<virt::VirtualCpu>> vcpus_;
    FlatIr flat_;
    uint64_t code_base_ = 0x1000;
    size_t code_size_ = 0;
    bool use_gpu_ = false;
    bool translated_ = false;
    void* cuda_handle_ = nullptr;
};

}  // namespace gpuemu
