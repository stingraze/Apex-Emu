#pragma once

#include "virt/memory.hpp"
#include "x86/state.hpp"
#include <cstdint>
#include <functional>

namespace virt {

// Virtualization layer: one vCPU with trap dispatch to host.
class VirtualCpu {
public:
    using SyscallHandler = std::function<int64_t(x86::CpuState&, GuestMemory&)>;

    VirtualCpu(GuestMemory& mem, uint64_t entry, SyscallHandler handler = {});

    x86::CpuState& state() { return state_; }
    const x86::CpuState& state() const { return state_; }
    GuestMemory& memory() { return mem_; }

    // Resolve effective address for ModR/M operand.
    uint64_t resolve_addr(const x86::CpuState& cpu, uint8_t base, uint32_t disp,
                          bool rip_rel, uint64_t insn_end) const;

    // Host-side syscall dispatch after GPU trap.
    void handle_syscall();

    void set_syscall_handler(SyscallHandler h) { syscall_handler_ = std::move(h); }

private:
    GuestMemory& mem_;
    x86::CpuState state_{};
    SyscallHandler syscall_handler_;
};

}  // namespace virt
