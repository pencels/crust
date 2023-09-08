#include <stdio.h>
#include <string.h>

typedef __int32_t i32;
typedef __int8_t i8;

typedef struct slice_ptr {
    void* slice;
    i32 len;
} slice_ptr;

typedef struct Range {
    i32 start;
    i32 end;
} Range;

typedef struct RangeFrom {
    i32 start;
} RangeFrom;

typedef struct RangeTo {
    i32 end;
} RangeTo;

slice_ptr slice_from_raw_parts(void* ptr, i32 len) {
    slice_ptr ret = {
        .slice = ptr,
        .len = len,
    };
    return ret;
}

void print_char(i8 c) {
    printf("%c", c);
}

void print_int(i32 i) {
    printf("%d\n", i);
}

void print_int_slice(slice_ptr /* : *[int] */ s) {
    printf("[");
    if (s.len > 0) {
        i32* ptr = s.slice;
        printf("%d", *ptr);
        for (i32 i = 1; i < s.len; i++) {
            printf(", %d", ptr[i]);
        }
    }
    printf("]\n");
}

void print_str_slice(slice_ptr /* : *[*str] */ s) {
    printf("[");
    if (s.len > 0) {
        slice_ptr* ptr = s.slice;
        printf("%.*s", ptr[0].len, (char*) ptr[0].slice);
        for (i32 i = 1; i < s.len; i++) {
            printf(", %.*s", ptr[i].len, (char*) ptr[i].slice);
        }
    }
    printf("]\n");
}

void print(slice_ptr /* : *str */ s) {
    printf("%.*s\n", s.len, (char*) s.slice);
}

void print_int_ptr(i32* ptr) {
    printf("%d\n", *ptr);
}

void for_each(slice_ptr s, void (*action)(void*)) {
    void** slice = s.slice;
    for (i32 i = 0; i < s.len; i++) {
        action(slice[i]);
    }
}

FILE* get_stdin() {
    return stdin;
}

extern int __crust__main(slice_ptr args);

int main(int argc, char* argv[]) {
    slice_ptr args[argc];

    for (int i = 0; i < argc; i++) {
        args[i] = slice_from_raw_parts(argv[i], strlen(argv[i]));
    }
    
    return __crust__main(slice_from_raw_parts(args, argc));
}
