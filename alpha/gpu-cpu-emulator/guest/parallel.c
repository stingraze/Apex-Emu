// Parallel demo: host passes vCPU id in RDI. Each instance prints one line.
// Build with the same flags as simple.c (see CMakeLists.txt guest_objects).

void _start(void) {
    __asm__ volatile(
        "mov $1, %%rax\n"
        "mov $1, %%rdi\n"
        "lea msg(%%rip), %%rsi\n"
        "lea endmsg(%%rip), %%rdx\n"
        "sub %%rsi, %%rdx\n"
        "syscall\n"

        "mov $60, %%rax\n"
        "mov $0, %%rdi\n"
        "syscall\n"

        "msg:\n"
        ".asciz \"vcpu running\\n\"\n"
        "endmsg:\n"
        ::: "rax", "rdi", "rsi", "rdx", "memory");
}
