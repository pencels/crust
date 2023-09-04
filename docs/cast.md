# Type cast expressions

Adapted from the [Operator Expressions](https://doc.rust-lang.org/reference/expressions/operator-expr.html#type-cast-expressions) section of the Rust Reference.

| Type of `e`| `U` | Cast performed by `e as U` |
|-|-|-|
|Integer or Float type|Integer or Float type|[Numeric cast](#numeric-cast)|
|`bool` or `char`|Integer type|[Primitive to integer cast](#primitive-to-integer-cast)|
|`*T`| `*V` where `V` is sized\* | [Pointer to pointer cast](#pointer-to-pointer-cast)|
|`*T` where `T` is sized|Integer type|[Pointer to address cast](#pointer-to-address-cast)|
|`*[T; N]`|`*T`|[Array to pointer cast](#array-to-pointer-cast)|
|Integer type|`*T` where `T` is sized|[Address to pointer cast](#address-to-pointer-cast)|
|Function item|Function pointer|[Function item to function pointer cast](#function-item-to-function-pointer-cast)|

\* or `T` and `V` are compatible unsized types, e.g. both slices

## Semantics

### Numeric cast

- Casting between two integers of the same size is a no-op.
- Casting from a larger integer to a smaller integer will truncate.
- Casting from a smaller integer to a larger integer will
    - zero-extend if the source is unsigned
    - sign-extend if the source is signed
- Casting from a float to an integer will defer to the `fpto(s|u)i` LLVM instructions
    - `s` if the target is signed
    - `u` if the target is unsigned
- Casting from an integer to float will defer to the `(s|u)itofp` LLVM instructions
    - `s` if the source is signed
    - `u` if the source is unsigned
- Casting from `float` to `double` is lossless
- Casting from `double` to `float` may lose some precision

### Primitive to integer cast

- `false` casts to `0`, `true` casts to `1`
- `char` casts to the value of the code point, then uses a numeric cast if needed

### Pointer to address cast

Produces the machine address of the referenced memory. If the integer type is smaller than the pointer type, the address may be truncated. Using `usize` avoids this.

### Address to pointer cast

Interprets the integer as a memory address and produces a pointer referencing that memory.

### Pointer to pointer cast

A no-op, simply uses the same pointer address of the source.

### Array to pointer cast

Extracts the address of the array in memory and uses this as the pointer value.

### Function item to function pointer cast

Extracts the address of the function in memory and uses this as the pointer value.

