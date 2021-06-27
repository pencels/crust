use std::cmp::{max, min};

use codespan_derive::{IntoLabel, Label};
use codespan_reporting::diagnostic::LabelStyle;

use super::FileId;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct Span {
    pub file_id: FileId,
    pub start: usize,
    pub end: usize,
}

impl IntoLabel for Span {
    type FileId = super::FileId;

    fn into_label(&self, style: LabelStyle) -> Label<Self::FileId> {
        Label::new(style, self.file_id, self.start..self.end)
    }
}

impl Span {
    pub fn new(file_id: FileId, start: usize, end: usize) -> Span {
        assert!(start <= end);
        Span {
            file_id,
            start,
            end,
        }
    }

    pub fn unite(self, next: Span) -> Span {
        assert!(self.file_id == next.file_id);
        Span {
            file_id: self.file_id,
            start: min(self.start, next.start),
            end: max(self.end, next.end),
        }
    }

    pub fn dummy() -> Span {
        Span {
            file_id: 0,
            start: 0,
            end: 0,
        }
    }
}
