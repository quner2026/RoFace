//! Storage module for face data persistence

pub mod traits;
pub mod sqlite;

pub use traits::{FaceStorage, FaceRecord, SearchResult};
pub use sqlite::{SqliteStorage, TaskRecord};
