Most types have a fixed size that is known at compile time and are called _sized_. A type with a size that is only known at run-time is called _unsized_.

In Rust, these are handled via a trait called `Sized`, but as Crust doesn't have a trait system, these are simply kept track of in the compiler.

Examples of unsized types are slices: `[T]` and `str`. Slices refer to a "view" of memory which is dynamically sized based on the size of the view at runtime.

Pointers to slices (`*[T]` and `*str`) are sized, but are twice the size of pointers to sized types. These are also known as _fat pointers_. Along with the pointer to the data, `*T`, they store the number of elements in the slice as a `usize`.

Variables, function parameters, and static items _must_ be sized.

Array and slice elements _must_ be sized.

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
