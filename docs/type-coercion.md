# Type coercions

A _type coercion_ is an implicit operation which change the type of a value. They happen at specific locations.

Any conversions allowed by coercion can also be performed by the type cast operator, `as`.

## Coercion sites

Coercion sites are locations in the code where an expression of type `T` will be implicitly coerced to a target type `U`.

- `let` statements where an explicit type is given:
    ```rust
    let x: U = e;
    ```
    `e` is coerced to `U`.

- `static` variable declarations, similar to `let`

- Arguments for function calls: each expression is coerced to the type of the corresponding parameter.

- Instantiations of struct fields: each expression is coerced to teh type of the corresponding struct field.

- Function returns: as the last expression of the function body, or in a `return` statement, the expression is coerced to the return type of the function.

If the expression in a coercion site is a _coercion-propagating expression_, then the relevant sub-expressions are also coercion sites. The coercion-propagating expressions are:

- Array literals, where the array is in a coercion site to type `[U; N]`. Each sub-expression in the literal is then a coercion site to `U`.

- Tuples, where the the tuple is a coercion site to type `(U_0, U_1, ..., U_N)`. Each sub-expression `e_i` is then a coercion site to `U_i`.

- Parenthesized sub-expressions `(e)`: if the expression is in a coercion site to type `U`, then the sub-expression `e` is a coercion site to `U`.

- Blocks: if a block has type `U` then the last expression in the block is a coercion site to `U`. e.g.
    - `if`/`else` blocks
    - function bodies

## Coercion types

Coercion is allowed between the following types:

- for [primitive types](./types.md#primitive-types): `T` to `U` if `T` is smaller than `U`

- `*mut T` to `*T`

- Functions to function pointers
