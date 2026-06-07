/* Link minimal Linux guest at a fixed kernel-like load address. */
ENTRY(_start)

SECTIONS
{
    . = 0x100000;

    .text : {
        *(.text)
    }

    .rodata : {
        *(.rodata)
        *(.rodata.*)
    }
}
