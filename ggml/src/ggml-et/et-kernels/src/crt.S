.section .text.init, "ax", @progbits
.global _start
_start:
    # initialize global pointer
.option push
.option norelax
    la    gp, __global_pointer$
.option pop
    # Firmware sets stack pointer before launch
    # bss not allowed, no init
    # et_kernel_entry is the profile.c wrapper, which snapshots PMCs and then
    # calls the kernel body (its entry_point, renamed to user_kernel at build).
    call  et_kernel_entry
    li    a2, 0  /* KERNEL_RETURN_SUCCESS (0) */
    mv    a1, a0
    li    a0, 8  /* SYSCALL_RETURN_FROM_KERNEL (8) */
    ecall
