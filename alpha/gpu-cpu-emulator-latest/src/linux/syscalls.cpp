#include "linux/syscalls.hpp"
#include <cerrno>
#include <cstring>
#include <iostream>

namespace linux_guest {

namespace {

constexpr int64_t kEnosys = -38;

uint64_t guest_ptr(const virt::GuestMemory& mem, uint64_t addr, uint64_t len) {
    if (addr + len > mem.size()) return 0;
    return addr;
}

int64_t sys_write(virt::GuestMemory& mem, int64_t fd, uint64_t buf, uint64_t len) {
    if (fd != 1 && fd != 2) return kEnosys;
    if (!guest_ptr(mem, buf, len)) return -14;  // EFAULT
    std::cout.write(reinterpret_cast<const char*>(mem.host_ptr() + buf),
                    static_cast<std::streamsize>(len));
    std::cout.flush();
    return static_cast<int64_t>(len);
}

int64_t sys_read(virt::GuestMemory&, int64_t fd, uint64_t, uint64_t) {
    if (fd == 0) return 0;  // EOF on stdin
    return kEnosys;
}

}  // namespace

int64_t dispatch(x86::CpuState& cpu, virt::GuestMemory& mem) {
    const int64_t nr = cpu.syscall_num;
    const int64_t a0 = static_cast<int64_t>(cpu.gpr[static_cast<int>(x86::Reg::RDI)]);
    const int64_t a1 = static_cast<int64_t>(cpu.gpr[static_cast<int>(x86::Reg::RSI)]);
    const int64_t a2 = static_cast<int64_t>(cpu.gpr[static_cast<int>(x86::Reg::RDX)]);
    const int64_t a3 = static_cast<int64_t>(cpu.gpr[static_cast<int>(x86::Reg::R10)]);
    (void)a3;

    int64_t ret = kEnosys;

    switch (nr) {
        case 0:   // read
            ret = sys_read(mem, a0, static_cast<uint64_t>(a1), static_cast<uint64_t>(a2));
            break;
        case 1:   // write
            ret = sys_write(mem, a0, static_cast<uint64_t>(a1), static_cast<uint64_t>(a2));
            break;
        case 12:  // brk — no-op allocator stub
            ret = a0 ? a0 : static_cast<int64_t>(mem.size() / 2);
            break;
        case 60:  // exit
            cpu.halt = 1;
            ret = a0;
            break;
        case 231: // exit_group
            cpu.halt = 1;
            ret = a0;
            break;
        default:
            ret = kEnosys;
            break;
    }

    cpu.syscall_ret = ret;
    cpu.gpr[0] = static_cast<uint64_t>(ret);
    cpu.syscall_pending = 0;
    return ret;
}

}  // namespace linux_guest
