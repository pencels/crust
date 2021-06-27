pub mod pretty;
mod span;

use std::{
    fmt::Debug,
    hash::Hash,
    marker::PhantomData,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

pub use span::Span;

use codespan_reporting::files::{Files, SimpleFiles};

pub type FileId = <SimpleFiles<String, String> as Files<'static>>::FileId;
pub type P<T> = Arc<T>;

pub struct Id<T: ?Sized>(usize, PhantomData<T>);

impl<T: ?Sized> Id<T> {
    pub fn id(&self) -> usize {
        self.0
    }
}

impl<T: ?Sized> Clone for Id<T> {
    fn clone(&self) -> Self {
        Id(self.0, PhantomData)
    }
}

impl<T: ?Sized> Copy for Id<T> {}

impl<T: ?Sized> PartialEq for Id<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0.eq(&other.0)
    }
}

impl<T: ?Sized> Eq for Id<T> {}

impl<T: ?Sized> Hash for Id<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state)
    }
}

impl<T: ?Sized> From<usize> for Id<T> {
    fn from(u: usize) -> Self {
        Self(u, PhantomData)
    }
}

impl<T: ?Sized> Debug for Id<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

static NEXT_ID: AtomicUsize = AtomicUsize::new(0);

pub fn fresh_id<T: From<usize>>() -> T {
    NEXT_ID.fetch_add(1, Ordering::Relaxed).into()
}
