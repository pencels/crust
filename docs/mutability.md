Similar to Rust, variables in Crust are immutable by default.

```rust
let x = 1;
x = x + 1; // Error: must define x as mutable
```

Ascribing a variable with `mut` will make it mutable.

```rust
let mut x = 1;
x = x + 1; // OK
```

## Pointer mutability

Pointers are immutable by default: this means that their target is immutable.

```rust
let x = 1;
let ptr = &x; // type is *int
*ptr = 2; // Error: Cannot change value since the pointer is immutable
```

To make a pointer a _mutable_ pointer, use the `&mut` operator. This also requires that the target is mutable.

```rust
let mut x = 1; // needs to be `mut`
let ptr = &mut x; // use `&mut` to get a pointer `*mut int`
*ptr = 2; // OK
```

For pointers to pointers, immutability trumps mutability if there are multiple levels of indirection.

```rust
let mut x = 1;
let mut_ptr = &mut x; // type is *mut int
let ptr_ptr = &mut_ptr; // type is **mut int

**ptr_tr = 2; // Error: Cannot mutate behind an immutable ptr
```

