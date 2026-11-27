//! The running engine.
//!
//! Audio goes in (the song on air and, once it is known, the next one),
//! audio comes out without a gap, and next to it a stream of events: when
//! the transition will start, where the one is, when it ends, where the DJ
//! may talk, and once a second what is going on.
//!
//! The engine never sees more of the song on air than `lookahead_s` ahead of
//! what it has played. Inside that window it plans the transition
//! (`sone-plan`), has it rendered ahead of time (a `Renderer`, in practice
//! `sone-render` on the stems of both songs), and then plays three things one
//! after the other without a seam: the song as it is, the rendered
//! transition, the next song. The next song leaves the transition at the
//! tempo it was matched to and is brought back to its own tempo over
//! sixteen bars with its pitch unchanged (a `Glide`, in practice the
//! stretcher of `sone-dsp`).
//!
//! No threads, no clock, no files: the caller pulls blocks of audio and
//! gets the events that fall into each block. That is what a Durable Object
//! in a Worker needs.

pub mod engine;
pub mod talk;

pub use engine::{Clip, Config, Glide, Live, Renderer, Song, When};
