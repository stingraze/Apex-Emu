#pragma once

#include "virt/memory.hpp"
#include "x86/state.hpp"
#include <cstdint>

namespace linux_guest {

// Minimal Linux x86-64 syscall emulation for bare-metal userland demos.
int64_t dispatch(x86::CpuState& cpu, virt::GuestMemory& mem);

}  // namespace linux_guest
