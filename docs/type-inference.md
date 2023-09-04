Crust's type inference is rudimentary. It only exists to reduce unneeded boilerplate at `let` definitions.

```rust
let x: int = 6;
let x = 6; // infers x: int
```

Crust cannot infer the type of a binding past the declaration site.

```rust
let x; // syntax error: a type and/or initializer must be provided
x = 6; 
```
