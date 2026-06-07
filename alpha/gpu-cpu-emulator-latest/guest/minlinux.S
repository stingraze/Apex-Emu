# Minimal bare-metal Linux userland demo for the GPU x86-64 emulator.
# Uses only write(1) and exit syscalls — mimics BusyBox init boot sequence.
    .text
    .global _start

    .macro WRITE_MSG label, len
    mov    $1, %rax
    mov    $1, %rdi
    lea    \label(%rip), %rsi
    mov    $\len, %rdx
    syscall
    .endm

_start:
    WRITE_MSG msg_boot,      boot_len
    WRITE_MSG msg_mount,     mount_len
    WRITE_MSG msg_init,      init_len
    WRITE_MSG msg_shell,     shell_len
    WRITE_MSG msg_echo_cmd,  echo_cmd_len
    WRITE_MSG msg_hello,     hello_len
    WRITE_MSG msg_uname_cmd, uname_cmd_len
    WRITE_MSG msg_uname,     uname_len
    WRITE_MSG msg_done,      done_len

    mov    $60, %rax
    xor    %rdi, %rdi
    syscall

    .section .rodata

msg_boot:
    .ascii "[    0.000000] Booting minimal Linux on GPU x86-64 emulator\n"
    boot_len = . - msg_boot

msg_mount:
    .ascii "[    0.002100] VFS: Mounted root (initramfs) on /dev/root\n"
    mount_len = . - msg_mount

msg_init:
    .ascii "[    0.003400] init: starting BusyBox v1.36 (emulated)\n"
    init_len = . - msg_init

msg_shell:
    .ascii "\n# /bin/sh (BusyBox) — type 'help' for applets\n"
    shell_len = . - msg_shell

msg_echo_cmd:
    .ascii "# echo \"Hello from emulated Linux\"\n"
    echo_cmd_len = . - msg_echo_cmd

msg_hello:
    .ascii "Hello from emulated Linux\n"
    hello_len = . - msg_hello

msg_uname_cmd:
    .ascii "# uname -a\n"
    uname_cmd_len = . - msg_uname_cmd

msg_uname:
    .ascii "Linux gpuemu 6.6.0 #1 SMP x86_64 GNU/Linux (BusyBox init)\n"
    uname_len = . - msg_uname

msg_done:
    .ascii "\n# init: system ready — powering down\n"
    done_len = . - msg_done
