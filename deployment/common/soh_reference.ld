/* Reference memory budget, NOT a selected STM32 part's complete memory map. */
ENTRY(Reset_Handler)
MEMORY {
 FLASH (rx) : ORIGIN = 0x08000000, LENGTH = 64K
 RAM (rwx) : ORIGIN = 0x20000000, LENGTH = 16K
}
_estack = ORIGIN(RAM) + LENGTH(RAM);
SECTIONS {
 .isr_vector : { KEEP(*(.isr_vector)) } > FLASH
 .text : { *(.text*) *(.rodata*) . = ALIGN(4); } > FLASH
 .ARM.extab : { *(.ARM.extab*) } > FLASH
 .ARM.exidx : { *(.ARM.exidx*) } > FLASH
 .data : { . = ALIGN(8); _sdata = .; *(.data*) . = ALIGN(8); _edata = .; } > RAM AT> FLASH
 _sidata = LOADADDR(.data);
 .bss (NOLOAD) : { . = ALIGN(8); _sbss = .; *(.bss*) *(COMMON) . = ALIGN(8); _ebss = .; } > RAM
 .stack_reserve (NOLOAD) : { . = ALIGN(8); . += 2048; } > RAM
 /DISCARD/ : { *(.comment) *(.note*) }
}
ASSERT(SIZEOF(.data) + SIZEOF(.bss) + SIZEOF(.stack_reserve) <= LENGTH(RAM), "RAM budget exceeded")
