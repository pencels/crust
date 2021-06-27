use std::{
    collections::{BTreeMap, HashMap, HashSet, VecDeque},
    fmt::{Debug, Display, Formatter, Result},
    sync::Arc,
};

pub trait PrettyPrint<Ctx> {
    fn fmt(&self, f: &mut Formatter, ctx: &Ctx) -> Result;
}

pub struct Delimited<'a, T>(
    pub &'static str,
    pub &'static str,
    pub &'static str,
    pub &'a [T],
);

impl<'a, T: Display> Display for Delimited<'a, T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        let &Delimited(start, delim, end, values) = self;
        write!(f, "{}", start)?;
        if let Some((head, tail)) = values.split_first() {
            write!(f, "{}", head)?;
            for v in tail {
                write!(f, "{}", delim)?;
                write!(f, "{}", v)?;
            }
        }
        write!(f, "{}", end)
    }
}

impl<'a, Ctx, T: PrettyPrint<Ctx>> PrettyPrint<Ctx> for Delimited<'a, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter, ctx: &Ctx) -> std::fmt::Result {
        let &Delimited(start, delim, end, values) = self;
        write!(f, "{}", start)?;
        if let Some((head, tail)) = values.split_first() {
            write!(f, "{}", &Pretty(head, ctx))?;
            for v in tail {
                write!(f, "{}", delim)?;
                write!(f, "{}", &Pretty(v, ctx))?;
            }
        }
        write!(f, "{}", end)
    }
}

pub struct Pretty<'ctx, T, Ctx>(pub T, pub &'ctx Ctx);

impl<'ctx, T: PrettyPrint<Ctx>, Ctx> Debug for Pretty<'ctx, T, Ctx> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        self.0.fmt(f, self.1)
    }
}

impl<'ctx, T: PrettyPrint<Ctx>, Ctx> Display for Pretty<'ctx, T, Ctx> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        self.0.fmt(f, self.1)
    }
}

impl<T: PrettyPrint<Ctx>, Ctx> PrettyPrint<Ctx> for &'_ T {
    fn fmt(&self, f: &mut Formatter, ctx: &Ctx) -> Result {
        <T as PrettyPrint<Ctx>>::fmt(self, f, ctx)
    }
}

impl<T: PrettyPrint<Ctx>, Ctx> PrettyPrint<Ctx> for Arc<T> {
    fn fmt(&self, f: &mut Formatter, ctx: &Ctx) -> Result {
        <T as PrettyPrint<Ctx>>::fmt(self, f, ctx)
    }
}

impl<T: PrettyPrint<Ctx>, Ctx> PrettyPrint<Ctx> for Option<T> {
    fn fmt(&self, f: &mut Formatter, ctx: &Ctx) -> Result {
        match self {
            Some(t) => {
                let mut d = f.debug_tuple("Some");
                d.field(&Pretty(t, ctx));
                d.finish()
            }
            None => write!(f, "None"),
        }
    }
}

impl<T: PrettyPrint<Ctx>, Ctx> PrettyPrint<Ctx> for &'_ [T] {
    fn fmt(&self, f: &mut Formatter, ctx: &Ctx) -> Result {
        let mut h = f.debug_list();

        for i in *self {
            h.entry(&Pretty(i, ctx));
        }

        h.finish()
    }
}

impl<T: PrettyPrint<Ctx>, Ctx> PrettyPrint<Ctx> for Vec<T> {
    fn fmt(&self, f: &mut Formatter, ctx: &Ctx) -> Result {
        let mut h = f.debug_list();

        for i in self {
            h.entry(&Pretty(i, ctx));
        }

        h.finish()
    }
}

impl<T: PrettyPrint<Ctx>, Ctx> PrettyPrint<Ctx> for VecDeque<T> {
    fn fmt(&self, f: &mut Formatter, ctx: &Ctx) -> Result {
        let mut h = f.debug_list();

        for i in self {
            h.entry(&Pretty(i, ctx));
        }

        h.finish()
    }
}

impl<K: PrettyPrint<Ctx>, V: PrettyPrint<Ctx>, Ctx> PrettyPrint<Ctx> for BTreeMap<K, V> {
    fn fmt(&self, f: &mut Formatter, ctx: &Ctx) -> Result {
        let mut h = f.debug_map();

        for (k, v) in self {
            h.entry(&Pretty(k, ctx), &Pretty(v, ctx));
        }

        h.finish()
    }
}

impl<K: PrettyPrint<Ctx>, V: PrettyPrint<Ctx>, Ctx> PrettyPrint<Ctx> for HashMap<K, V> {
    fn fmt(&self, f: &mut Formatter, ctx: &Ctx) -> Result {
        let mut h = f.debug_map();

        for (k, v) in self {
            h.entry(&Pretty(k, ctx), &Pretty(v, ctx));
        }

        h.finish()
    }
}

impl<K: PrettyPrint<Ctx>, Ctx> PrettyPrint<Ctx> for HashSet<K> {
    fn fmt(&self, f: &mut Formatter, ctx: &Ctx) -> Result {
        let mut h = f.debug_set();

        for k in self {
            h.entry(&Pretty(k, ctx));
        }

        h.finish()
    }
}

macro_rules! simple_pretty {
    ($tr:path; $($ty:ty),*) => {$(
        impl<Ctx> PrettyPrint<Ctx> for $ty {
            fn fmt(&self, f: &mut Formatter, _: &Ctx) -> Result {
                <Self as $tr>::fmt(self, f)
            }
        }
    )*};
    ($($ty:ty,)*) => {
        simple_pretty! { $($ty),* }
    }
}

simple_pretty!(std::fmt::Display; bool, i64, u64, usize, char, str, &'static str, String);
simple_pretty!(std::fmt::Debug; std::path::PathBuf);

macro_rules! tuple_pretty {
    ($($ty:ident),*; $($idx:tt),*) => {
        impl<$($ty: PrettyPrint<Ctx>),* , Ctx> PrettyPrint<Ctx> for ($($ty,)*) {
            fn fmt(&self, f: &mut Formatter, ctx: &Ctx) -> Result {
                let mut h = f.debug_tuple("");
                $(h.field(&Pretty(&self.$idx, ctx));)*
                h.finish()
            }
        }
    };
}

tuple_pretty!(A; 0);
tuple_pretty!(A, B; 0, 1);
tuple_pretty!(A, B, C; 0, 1, 2);
tuple_pretty!(A, B, C, D; 0, 1, 2, 3);
