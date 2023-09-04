Most types have a fixed size that is known at compile time and is called _sized_. A type with a size that is only known at run-time is called _unsized_.

In Rust, these are handled via a trait called `Sized`, but as Crust doesn't have a trait system, these are simply kept track of in the compiler.

Examples of unsized types are slices: `[T]` and `str`. 

- Pointers to unsized types are sized, but have twice the size of pointers to sized types. These are also known as _fat pointers_. Along with the pointer to the data, they store the number of elements in the slice.
- Structs may contain an unsized type as the last field; this makes the struct itself unsized.

Variables, function parameters, and static items _must_ be sized.

|Type|Sized?|
|-|-|
|`char`|Y|
|`*char`|Y|
|`str`|N|
|`*str`|Y|
|`*mut str`|Y|
|`[char]`|N|
|`[char; N]`|Y|
|`*[char]`|Y|
|`*[char; N]`|Y|
