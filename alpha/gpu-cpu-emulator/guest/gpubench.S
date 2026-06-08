# GPU showcase: N parallel x86-64 guests, one CUDA thread each.
# Host sets RDI = vCPU id. Each core runs a deterministic checksum loop;
# exit(tally) = id×ITER + Σ((seed+i)&7) lets the host verify CPU == GPU.

    .equ ITER, 200000

    .text
    .globl _start
_start:
    push    %rbp
    mov     %rsp, %rbp
    push    %rbx
    push    %r12
    push    %r13
    push    %r15

    mov     %rdi, %r15          # vCPU id

    test    %r15, %r15
    jnz     .Lskip_banner
    mov     $1, %rax
    mov     $1, %rdi
    lea     banner(%rip), %rsi
    lea     banner_end(%rip), %rdx
    sub     %rsi, %rdx
    syscall
.Lskip_banner:

    xor     %r12, %r12
    mov     %r15, %r13          # seed = id (unique per guest)
    xor     %rbx, %rbx
.Louter:
    cmp     $ITER, %rbx
    jae     .Lexit

    mov     %r13, %rax
    add     %rbx, %rax
    and     $7, %rax
    add     %rax, %r12

    add     $3, %r13
    add     $1, %rbx
    jmp     .Louter

.Lexit:
    mov     %r12, %rdi
    mov     $60, %rax
    syscall

    pop     %r15
    pop     %r13
    pop     %r12
    pop     %rbx
    pop     %rbp
    ret

    .section .rodata
banner:
    .ascii ">>> N parallel x86-64 guests on GPU (one CUDA thread each) <<<\n"
banner_end:
