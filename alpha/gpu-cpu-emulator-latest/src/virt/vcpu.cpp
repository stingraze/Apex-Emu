#include "virt/vcpu.hpp"
#include "linux/syscalls.hpp"

namespace virt {

VirtualCpu::VirtualCpu(GuestMemory& mem, uint64_t entry, SyscallHandler handler)
    : mem_(mem), syscall_handler_(std::move(handler)) {
    state_.rip = entry;
    state_.gpr[static_cast<int>(x86::Reg::RSP)] = mem_.size() - 0x100;
}

uint64_t VirtualCpu::resolve_addr(const x86::CpuState& cpu, uint8_t base, uint32_t disp,
                                   bool rip_rel, uint64_t insn_end) const {
    if (rip_rel) return insn_end + static_cast<int32_t>(disp);
    return cpu.gpr[base] + disp;
}

void VirtualCpu::handle_syscall() {
    if (syscall_handler_) {
        state_.syscall_ret = syscall_handler_(state_, mem_);
        state_.gpr[0] = static_cast<uint64_t>(state_.syscall_ret);
    } else {
        linux_guest::dispatch(state_, mem_);
    }
    state_.syscall_pending = 0;
}

}  // namespace virt
