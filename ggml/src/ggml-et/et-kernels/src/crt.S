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
    # Save arguments
    addi sp, sp, -16
    sd a0, 0(sp)
    sd a1, 8(sp)

    call et_profile_start

    # Restore arguments
    ld a0, 0(sp)
    ld a1, 8(sp)
    addi sp, sp, 16
    call  entry_point

    # Save return value
    addi sp, sp, -16
    sd a0, 0(sp)

    call et_profile_end

    # Restore return value
    ld a0, 0(sp)
    addi sp, sp, 16

    li    a2, 0  /* KERNEL_RETURN_SUCCESS (0) */
    mv    a1, a0
    li    a0, 8  /* SYSCALL_RETURN_FROM_KERNEL (8) */
    ecall
