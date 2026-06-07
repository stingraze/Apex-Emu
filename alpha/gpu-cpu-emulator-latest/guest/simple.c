// Minimal x86-64 guest compiled from C (not hand-written bytecode).
// Computes 17 + 25, prints a message, exits with the sum (42) as the exit code.

void _start(void) {
    __asm__ volatile(
        "mov $17, %%rbx\n"
        "add $25, %%rbx\n"

        "mov $1, %%rax\n"
        "mov $1, %%rdi\n"
        "lea msg(%%rip), %%rsi\n"
        "lea endmsg(%%rip), %%rdx\n"
        "sub %%rsi, %%rdx\n"
        "syscall\n"

        "mov %%rbx, %%rdi\n"
        "mov $60, %%rax\n"
        "syscall\n"

        "msg:\n"
        ".asciz \"Hello from compiled x86-64 C!\\n\"\n"
        "endmsg:\n"
        ::: "rax", "rbx", "rdi", "rsi", "rdx", "memory");
}
