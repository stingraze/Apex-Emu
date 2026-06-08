#include "emulator.hpp"
#include "guest/elf_loader.hpp"
#include <cstring>
#include <iostream>
#include <unordered_map>

#ifdef GPUEMU_CUDA
extern "C" {
int gpuemu_cuda_available();
void* gpuemu_cuda_create(uint32_t num_vcpus, uint64_t mem_size,
                          const ir::Instr* ir, uint32_t ir_len);
int gpuemu_cuda_upload(void* h, uint32_t idx, const x86::CpuState* cpu, const uint8_t* mem);
int gpuemu_cuda_set_pc(void* h, uint32_t idx, uint32_t pc);
int gpuemu_cuda_run(void* h, uint32_t steps);
int gpuemu_cuda_download(void* h, uint32_t idx, x86::CpuState* cpu, uint8_t* mem);
void gpuemu_cuda_destroy(void* h);
const char* gpuemu_cuda_device_name();
}
#endif

namespace gpuemu {

FlatIr flatten_ir(const ir::TranslationUnit& tu) {
    FlatIr flat;
    flat.entry_pc = tu.entry_pc;

    // Map guest PC -> IR index for each instruction start.
    std::unordered_map<uint64_t, uint32_t> pc_to_idx;
    uint32_t idx = 0;

    for (const auto& bb : tu.blocks) {
        for (size_t i = 0; i < bb.code.size(); ++i) {
            const uint64_t gpc = i < bb.guest_pcs.size() ? bb.guest_pcs[i] : bb.guest_pc + i;
            pc_to_idx[gpc] = idx + static_cast<uint32_t>(i);
        }
        idx += static_cast<uint32_t>(bb.code.size());
    }
    flat.pc_to_idx = pc_to_idx;

    // Second pass: emit with resolved jump targets.
    idx = 0;
    for (const auto& bb : tu.blocks) {
        for (auto ins : bb.code) {
            if (ins.op == ir::Opcode::Jmp || ins.op == ir::Opcode::Jcc ||
                ins.op == ir::Opcode::Call) {
                const uint64_t target_pc = static_cast<uint64_t>(ins.imm);
                auto it = pc_to_idx.find(target_pc);
                ins.imm = (it != pc_to_idx.end()) ? static_cast<int64_t>(it->second) : 0;
            }
            flat.code.push_back(ins);
            ++idx;
        }
    }
    return flat;
}

Emulator::Emulator() {
#ifdef GPUEMU_CUDA
    use_gpu_ = gpuemu_cuda_available() != 0;
#endif
}

void Emulator::load_code(uint64_t guest_addr, const uint8_t* bytes, size_t len) {
    mem_.load(guest_addr, bytes, len);
    code_base_ = guest_addr;
    code_size_ = len;
    translated_ = false;
}

bool Emulator::load_guest(const char* path, uint64_t& entry_out) {
    auto img = guest::load_guest_file(mem_, path);
    if (!img.ok) {
        std::cerr << "ELF load failed: " << img.error << "\n";
        return false;
    }
    code_base_ = img.code_addr;
    code_size_ = img.code_size;
    entry_out = img.entry;
    translated_ = false;
    return true;
}

uint32_t Emulator::add_vcpu(uint64_t entry_pc, const x86::CpuState* initial) {
    auto vc = std::make_unique<virt::VirtualCpu>(mem_, entry_pc);
    if (initial) {
        const uint64_t rsp = vc->state().gpr[static_cast<int>(x86::Reg::RSP)];
        vc->state() = *initial;
        vc->state().rip = entry_pc;
        vc->state().gpr[static_cast<int>(x86::Reg::RSP)] = rsp;
    }
    vcpus_.push_back(std::move(vc));
    return static_cast<uint32_t>(vcpus_.size() - 1);
}

void Emulator::translate(uint64_t entry_pc) {
    (void)entry_pc;
    const uint8_t* code = mem_.host_ptr() + code_base_;
    Translator tr;
    auto tu = tr.translate(code, code_size_, code_base_);
    flat_ = flatten_ir(tu);
    translated_ = true;

    if (flat_.code.empty()) {
        std::cerr << "Translation failed: no IR ops (code_addr=0x" << std::hex << code_base_
                  << " code_size=" << std::dec << code_size_ << ")\n";
    }

#ifdef GPUEMU_CUDA
    if (cuda_handle_) {
        gpuemu_cuda_destroy(cuda_handle_);
        cuda_handle_ = nullptr;
    }
#else
    (void)cuda_handle_;
#endif
}

bool Emulator::init_gpu_backend() {
    return init_gpu();
}

bool Emulator::init_gpu() {
#ifdef GPUEMU_CUDA
    if (!use_gpu_ || flat_.code.empty() || vcpus_.empty()) {
        return false;
    }
    if (cuda_handle_) {
        gpuemu_cuda_destroy(cuda_handle_);
        cuda_handle_ = nullptr;
    }
    cuda_handle_ = gpuemu_cuda_create(
        static_cast<uint32_t>(vcpus_.size()),
        mem_.size(),
        flat_.code.data(),
        static_cast<uint32_t>(flat_.code.size()));
    if (!cuda_handle_) {
        std::cerr << "CUDA executor init failed; using CPU fallback\n";
        use_gpu_ = false;
        return false;
    }
    return true;
#else
    return false;
#endif
}

void Emulator::handle_traps() {
    for (auto& vc : vcpus_) {
        if (vc->state().syscall_pending) {
            vc->handle_syscall();
        }
    }
}

RunResult Emulator::run(uint32_t max_steps_per_vcpu) {
    RunResult result;
    if (!translated_ || flat_.code.empty()) {
        return result;
    }

    if (use_gpu_ && !cuda_handle_) {
        init_gpu();
    }
    const bool on_gpu = use_gpu_ && cuda_handle_ != nullptr;
    if (on_gpu) {
        run_gpu(max_steps_per_vcpu);
    } else {
        run_cpu(max_steps_per_vcpu);
    }

    handle_traps();

    bool all_halted = true;
    for (auto& vc : vcpus_) {
        if (!vc->state().halt) all_halted = false;
        if (vc->state().halt) {
            result.exit_code = vc->state().syscall_ret;
        }
    }
    result.halted = all_halted;
    result.steps = max_steps_per_vcpu;
    return result;
}

const uint8_t* Emulator::vcpu_memory(uint32_t idx) const {
    if (idx >= vcpu_mem_.size()) {
        return mem_.host_ptr();
    }
    return vcpu_mem_[idx].data();
}

void Emulator::run_cpu(uint32_t max_steps) {
    const uint32_t n = static_cast<uint32_t>(vcpus_.size());
    const uint64_t slice = mem_.size();
    std::vector<std::vector<uint8_t>> mem_slices(n);
    for (uint32_t i = 0; i < n; ++i) {
        mem_slices[i].resize(slice);
        std::memcpy(mem_slices[i].data(), mem_.host_ptr(), slice);
    }

    std::vector<uint32_t> pcs(n);
    for (uint32_t i = 0; i < n; ++i) {
        pcs[i] = flat_.ir_index_for(vcpus_[i]->state().rip);
    }
    const uint32_t chunk = 4096;
    uint32_t total = 0;

    while (total < max_steps) {
        bool any_running = false;
        for (uint32_t i = 0; i < n; ++i) {
            auto& cpu = vcpus_[i]->state();
            if (cpu.halt) continue;
            any_running = true;

            const int st = cpuexec::run_ir(
                cpu, mem_slices[i].data(), slice,
                flat_.code.data(), static_cast<uint32_t>(flat_.code.size()),
                pcs[i], chunk);

            if (st == 2) {
                cpu.syscall_pending = 1;
                vcpus_[i]->handle_syscall();
            }
        }
        total += chunk;
        if (!any_running) break;
    }

    vcpu_mem_ = std::move(mem_slices);
}

void Emulator::run_gpu(uint32_t max_steps) {
#ifdef GPUEMU_CUDA
    if (!cuda_handle_) return;

    for (size_t i = 0; i < vcpus_.size(); ++i) {
        gpuemu_cuda_upload(cuda_handle_, static_cast<uint32_t>(i),
                           &vcpus_[i]->state(), mem_.host_ptr());
        gpuemu_cuda_set_pc(cuda_handle_, static_cast<uint32_t>(i),
                           flat_.ir_index_for(vcpus_[i]->state().rip));
    }

    const uint32_t chunk = 4096;
    uint32_t total = 0;
    while (total < max_steps) {
        gpuemu_cuda_run(cuda_handle_, chunk);

        bool any_running = false;
        for (size_t i = 0; i < vcpus_.size(); ++i) {
            x86::CpuState cpu{};
            gpuemu_cuda_download(cuda_handle_, static_cast<uint32_t>(i),
                                 &cpu, mem_.host_ptr());
            vcpus_[i]->state() = cpu;

            if (cpu.syscall_pending) {
                vcpus_[i]->handle_syscall();
                gpuemu_cuda_upload(cuda_handle_, static_cast<uint32_t>(i),
                                   &vcpus_[i]->state(), mem_.host_ptr());
            }
            if (!vcpus_[i]->state().halt) any_running = true;
        }
        total += chunk;
        if (!any_running) break;
    }

    vcpu_mem_.resize(vcpus_.size());
    for (size_t i = 0; i < vcpus_.size(); ++i) {
        vcpu_mem_[i].resize(mem_.size());
        gpuemu_cuda_download(cuda_handle_, static_cast<uint32_t>(i),
                             &vcpus_[i]->state(), vcpu_mem_[i].data());
    }
#else
    run_cpu(max_steps);
#endif
}

}  // namespace gpuemu
