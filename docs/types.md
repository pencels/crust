## Primitive types

### Boolean type

`bool` takes on one of two values, called `true` and `false`. Its theoretical bit width is 1, but can vary depending on implementation.

### Integer types

|Type|Bit Width|Signed?|
|-|-|-|
|`byte`|8|Y|
|`char`|8|N|
|`short`|16|Y
|`ushort`|16|N|
|`int`|32|Y|
|`uint`|32|N|
|`long`|64|Y|
|`ulong`|64|N|

### Floating-point types

|Type|IEEE-754-2008 type|
|-|-|
|`float`|binary32|
|`double`|binary64|

### Machine-dependent integer types

The `usize` type is an unsigned integer type with the same number of bits as the platform's pointer type.

`isize` is similar in bit layout but is interpreted as a signed integer.

## Pointer types

A pointer is the address of some type `T`. Pointers have an associated mutability, which refers to whether the target type can be modified through the pointer. Written `*T` for normal pointers and `*mut T` for mutable pointers.

## Array types

An array is a fixed-size sequence of `N` elements of type `T`. It is written `[T; N]` where `T` is a type and `N` is an integer that is cast to a `usize`.

Arrays have bounds-checking applied to any accesses.

## Slice types

A slice is a "view" into a sequence of elements of type `T`. It is written as `[T]`.

`str` is a special slice type which acts as an alias to `[char]`.

Slice types are generally used through pointer types. For example:

- `*[T]`: A constant pointer to the slice of `T`.
- `*mut [T]`: A mutable pointer to the slice of `T` which supports mutating the contents of the slice.

Slices have bounds-checking applied to any accesses at run-time.

## Struct types

A struct is a heterogenous collection of other types. Each type is named in a _field_ of the struct.

```rust
struct Person {
    first_name: *str,
    age: uint,
    favorite_color: Color,
}
```

Fields of the struct can be accessed via their name:

- `s.first_name`
- `s.age`
- `s.favorite_color`
- etc.

## Tuple types

A tuple is a heterogenous collection of other types. It is similar to a struct type, but its fields are anonymous. It is written as a comma-delimited list surrounded by parentheses: `(T1, T2, ...)`.

- The zero-length tuple, also called _unit_, is written `()`. 
- The type of a tuple with a single element is written `(T,)`.

Members of a tuple are accessed via their index:

- `t.0`
- `t.1`
- ... and so on.

## Function and function pointer types

A function type is functionally equivalent to its pointer type. Function pointers cannot be mutable.

```rust
fn id(x: int) -> int {
    x
}

fn main() {
    let f = id; // type: fn(int) -> int
}
```
