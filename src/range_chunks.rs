use std::{
    iter::successors,
    ops::{Add, Range},
};

pub trait RangeChunk<T>
where
    Self: Sized,
{
    fn chunks(self, chunk_size: T) -> impl Iterator<Item = Self>;
}

impl<T> RangeChunk<T> for Range<T>
where
    T: Copy + Add<Output = T> + Ord,
{
    /// Split a `Range` into chunks of `chunk_size`
    fn chunks(self, chunk_size: T) -> impl Iterator<Item = Self> {
        successors(Some(self.start), move |&i| {
            let next_start = i + chunk_size;
            (next_start < self.end).then_some(next_start)
        })
        .map(move |block_start| {
            let block_end = (block_start + chunk_size).min(self.end);
            block_start..block_end
        })
    }
}
