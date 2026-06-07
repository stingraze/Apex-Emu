// Guest program: compile with
//   gcc -c -nostdlib -fno-pic -fno-stack-protector -fcf-protection=none
//        -fno-asynchronous-unwind-tables -O0 guest/demo.c -o demo.o

typedef unsigned long u64;

static u64 sys_write(int fd, const char* buf, u64 len) {
    u64 ret;
    __asm__ volatile(
        "mov %1, %%rdi\n"
        "mov %2, %%rsi\n"
        "mov %3, %%rdx\n"
        "mov $1, %%rax\n"
        "syscall\n"
        "mov %%rax, %0\n"
        : "=r"(ret)
        : "r"((u64)fd), "r"(buf), "r"(len)
        : "rax", "rdi", "rsi", "rdx", "rcx", "r11", "memory");
    return ret;
}

static void println(const char* msg) {
    u64 n = 0;
    while (msg[n]) n++;
    sys_write(1, msg, n);
    sys_write(1, "\n", 1);
}

static u64 mod10(u64 n) {
    while (n >= 10) n -= 10;
    return n;
}

static u64 div10(u64 n) {
    u64 q = 0;
    while (n >= 10) {
        n -= 10;
        q++;
    }
    return q;
}

static void print_u64(u64 v) {
    char tmp[24];
    int i = 0;
    if (v == 0) {
        tmp[i++] = '0';
    } else {
        while (v > 0) {
            tmp[i++] = (char)('0' + mod10(v));
            v = div10(v);
        }
    }
    char out[24];
    int j = 0;
    while (i > 0) out[j++] = tmp[--i];
    out[j] = '\0';
    println(out);
}

void _start(void) {
    const u64 a = 17;
    const u64 b = 25;
    const u64 sum = a + b;
    u64 prod = 0;
    u64 i;
    for (i = 0; i < b; i++) prod += a;

    println("=== x86-64 .o guest (demo.c) ===");
    println("a = 17, b = 25");
    println("a + b =");
    print_u64(sum);
    println("a * b (loop mul) =");
    print_u64(prod);
    println("Done.");

    __asm__ volatile("mov $60, %%rax\nxor %%rdi, %%rdi\nsyscall" ::: "rax", "rdi");
}
