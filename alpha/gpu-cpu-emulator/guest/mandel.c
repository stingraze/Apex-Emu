// Parallel Mandelbrot — compiled C guest, fixed-point (no switch/jump tables).
typedef long i64;
typedef unsigned long u64;

#define WIDTH   64
#define HEIGHT  32
#define FB_BASE 0x200000UL
#define MAX_ITER 12
#define LIMIT   (4L * 256L * 256L)

static char pick(u64 it) {
    u64 idx = it >> 1;
    if (idx > 7) {
        return '#';
    }
    if (idx == 0) return ' ';
    if (idx == 1) return '.';
    if (idx == 2) return ':';
    if (idx == 3) return '-';
    if (idx == 4) return '=';
    if (idx == 5) return '+';
    if (idx == 6) return '*';
    return '#';
}

void _start(void) {
    u64 y0;
    u64 y1;
    u64 y;
    u64 x;

    __asm__ volatile("mov %%rcx, %0" : "=r"(y0));
    __asm__ volatile("mov %%rdx, %0" : "=r"(y1));

    y = y0;
    while (y < y1) {
        i64 cy = 300 - ((i64)y << 4) - ((i64)y << 1) - (i64)y;
        x = 0;
        while (x < WIDTH) {
            i64 cx = -512 + ((i64)x << 3) + ((i64)x << 2) + ((i64)x << 1);
            u64 it = 0;
            i64 zx = 0;
            i64 zy = 0;

            while (it < MAX_ITER) {
                i64 zx2 = zx * zx;
                i64 zy2 = zy * zy;
                if (((zx2 >> 8) + (zy2 >> 8)) >= LIMIT) {
                    break;
                }
                i64 zxy = (zx * zy) >> 8;
                zx = (zx2 >> 8) - (zy2 >> 8) + cx;
                zy = zxy + zxy + cy;
                it++;
            }

            {
                char ch = pick(it);
                volatile char* p = (volatile char*)(FB_BASE + y * WIDTH + x);
                *p = ch;
            }
            x++;
        }
        y++;
    }

    __asm__ volatile(
        "mov $60, %%rax\n"
        "xor %%rdi, %%rdi\n"
        "syscall\n"
        ::: "rax", "rdi");
}
