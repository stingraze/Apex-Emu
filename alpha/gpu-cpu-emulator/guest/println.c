// Guest with println() — compiled from C.
// Build: gcc -c -nostdlib -fno-pic -fno-stack-protector -fcf-protection=none
//           -fno-asynchronous-unwind-tables -O0 guest/println.c -o println.o

typedef unsigned long u64;

static const char newline[] = "\n";

static void write_bytes(const char* buf, u64 len) {
    __asm__ volatile(
        "mov $1, %%rax\n"
        "mov $1, %%rdi\n"
        "mov %0, %%rsi\n"
        "mov %1, %%rdx\n"
        "syscall\n"
        :
        : "r"(buf), "r"(len)
        : "rax", "rdi", "rsi", "rdx", "memory");
}

static void println(const char* msg) {
    u64 n = 0;
    while (msg[n]) {
        n++;
    }
    write_bytes(msg, n);
    write_bytes(newline, 1);
}

static u64 mod10(u64 n) {
    while (n >= 10) {
        n -= 10;
    }
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
    char tmp[20];
    u64 i = 0;

    if (v == 0) {
        tmp[i++] = '0';
    } else {
        while (v > 0) {
            tmp[i++] = (char)('0' + mod10(v));
            v = div10(v);
        }
    }

    char out[20];
    u64 j = 0;
    while (i > 0) {
        out[j++] = tmp[--i];
    }
    out[j] = '\0';
    println(out);
}

void _start(void) {
    const u64 a = 17;
    const u64 b = 25;
    const u64 sum = a + b;
    u64 prod = 0;
    u64 k;

    for (k = 0; k < b; k++) {
        prod += a;
    }

    println("=== println guest (compiled C) ===");
    println("a = 17, b = 25");
    println("a + b =");
    print_u64(sum);
    println("a * b (loop) =");
    print_u64(prod);
    println("Done.");

    __asm__ volatile(
        "mov $60, %%rax\n"
        "mov %0, %%rdi\n"
        "syscall\n"
        :
        : "r"(sum)
        : "rax", "rdi");
}
