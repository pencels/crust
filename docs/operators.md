
## Arithmetic Operators

### operator `+`

This operator is commutative, so the left and right operands may be reversed.

|Left|Right|Result|
|-|-|-|
|Integer type `T` \*|Integer type `U` \*|Integer type `V` determined as the largest integer type between `T` and `U`|
|Integer type `T` \*|Float type `U`|Float type `U`|
|Float type `T` \*|Float type `U` \*|Float type `V` determined as the largest floating-point type between `T` and `U`|
|Pointer type `*T`|`isize`|Pointer type `*T`|

\* These operands are coerced to the result type before they are used in the operation.


### operator `-`

This operator is not commutative.

|Left|Right|Result|
|-|-|-|
|Integer type `T` \*|Integer type `U` \*|Integer type `V` determined as the largest common integer type between `T` and `U`|
|Integer type `T` \*|Float type `U`|Float type `U`|
|Float type `T`|Integer type `U` \*|Float type `T`|
|Float type `T` \*|Float type `U` \*|Float type `V` determined as the largest floating-point type between `T` and `U`|
|Pointer type `*T`|`isize`|Pointer type `*T`|
|Pointer type `*T`|Pointer type `*T`|`isize`|

\* These operands are coerced to the result type before they are used in the operation.

### operators `^` `&` `|`

These operators are commutative.

|Left|Right|Result|
|-|-|-|
|`bool`|`bool`|`bool`|
|Integer type `T` \*|Integer type `U` \*|Integer type `V` determined as the largest common integer type between `T` and `U`|

### operators `>>` `<<`

These operators are not commutative.

|Left|Right|Result|
|-|-|-|
|Integer type `T` (signed)|`usize`|Integer type `T` (sign-extend)|
|Integer type `T` (unsigned)|`usize`|Integer type `T` (zero-extend)|

### operators `&&` `||`

These operators are not commutative.

|Left|Right|Result|
|-|-|-|
|`bool`|`bool`|`bool`|

### operators `*` `/` `%`

### operators `==` `!=` `>` `>=` `<` `<=`

