//! SQLite storage implementation

use std::path::Path;

use async_trait::async_trait;
use sqlx::{sqlite::{SqlitePool, SqlitePoolOptions}, Row};
use anyhow::{Result, Context};
use tracing::{info, debug};

use super::traits::{FaceStorage, FaceRecord, SearchResult, cosine_similarity};

/// SQLite-based face storage
pub struct SqliteStorage {
    pool: SqlitePool,
}

impl SqliteStorage {
    /// Create a new SQLite storage
    pub async fn new(db_path: &str) -> Result<Self> {
        // Ensure parent directory exists
        if let Some(parent) = Path::new(db_path).parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Create connection pool
        let database_url = format!("sqlite:{}?mode=rwc", db_path);
        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect(&database_url)
            .await
            .context("Failed to connect to SQLite database")?;

        let storage = Self { pool };
        storage.initialize().await?;

        Ok(storage)
    }

    /// Initialize database schema
    async fn initialize(&self) -> Result<()> {
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS faces (
                face_id TEXT PRIMARY KEY,
                person_id TEXT NOT NULL,
                person_name TEXT NOT NULL,
                category TEXT DEFAULT 'default',
                embedding BLOB NOT NULL,
                metadata TEXT,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            )
            "#,
        )
        .execute(&self.pool)
        .await?;

        // Create indexes
        sqlx::query(
            r#"
            CREATE INDEX IF NOT EXISTS idx_faces_person_id ON faces(person_id)
            "#,
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            r#"
            CREATE INDEX IF NOT EXISTS idx_faces_category ON faces(category)
            "#,
        )
        .execute(&self.pool)
        .await?;

        // Migration: add category column if it doesn't exist
        let _ = sqlx::query("ALTER TABLE faces ADD COLUMN category TEXT DEFAULT 'default'")
            .execute(&self.pool)
            .await;

        // Tasks table for history
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                task_type TEXT NOT NULL,
                result TEXT,
                image_path TEXT,
                faces_json TEXT,
                duration_ms INTEGER NOT NULL,
                created_at INTEGER NOT NULL
            )
            "#,
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            r#"
            CREATE INDEX IF NOT EXISTS idx_tasks_created_at ON tasks(created_at DESC)
            "#,
        )
        .execute(&self.pool)
        .await?;

        info!("SQLite database initialized");
        Ok(())
    }
}

#[async_trait]
impl FaceStorage for SqliteStorage {
    async fn save_face(&self, record: &FaceRecord) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO faces (face_id, person_id, person_name, category, embedding, metadata, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            "#,
        )
        .bind(&record.face_id)
        .bind(&record.person_id)
        .bind(&record.person_name)
        .bind(&record.category)
        .bind(&record.embedding)
        .bind(&record.metadata)
        .bind(record.created_at)
        .bind(record.updated_at)
        .execute(&self.pool)
        .await?;

        debug!("Saved face: {}", record.face_id);
        Ok(())
    }

    async fn get_face(&self, face_id: &str) -> Result<Option<FaceRecord>> {
        let row = sqlx::query(
            r#"
            SELECT face_id, person_id, person_name, category, embedding, metadata, created_at, updated_at
            FROM faces
            WHERE face_id = ?
            "#,
        )
        .bind(face_id)
        .fetch_optional(&self.pool)
        .await?;

        match row {
            Some(row) => Ok(Some(FaceRecord {
                face_id: row.get("face_id"),
                person_id: row.get("person_id"),
                person_name: row.get("person_name"),
                category: row.get("category"),
                embedding: row.get("embedding"),
                metadata: row.get("metadata"),
                created_at: row.get("created_at"),
                updated_at: row.get("updated_at"),
            })),
            None => Ok(None),
        }
    }

    async fn get_faces_by_person(&self, person_id: &str) -> Result<Vec<FaceRecord>> {
        let rows = sqlx::query(
            r#"
            SELECT face_id, person_id, person_name, category, embedding, metadata, created_at, updated_at
            FROM faces
            WHERE person_id = ?
            ORDER BY created_at DESC
            "#,
        )
        .bind(person_id)
        .fetch_all(&self.pool)
        .await?;

        let records: Vec<FaceRecord> = rows
            .into_iter()
            .map(|row| FaceRecord {
                face_id: row.get("face_id"),
                person_id: row.get("person_id"),
                person_name: row.get("person_name"),
                category: row.get("category"),
                embedding: row.get("embedding"),
                metadata: row.get("metadata"),
                created_at: row.get("created_at"),
                updated_at: row.get("updated_at"),
            })
            .collect();

        Ok(records)
    }

    async fn delete_face(&self, face_id: &str) -> Result<bool> {
        let result = sqlx::query("DELETE FROM faces WHERE face_id = ?")
            .bind(face_id)
            .execute(&self.pool)
            .await?;

        Ok(result.rows_affected() > 0)
    }

    async fn delete_person(&self, person_id: &str) -> Result<u64> {
        let result = sqlx::query("DELETE FROM faces WHERE person_id = ?")
            .bind(person_id)
            .execute(&self.pool)
            .await?;

        Ok(result.rows_affected())
    }

    async fn search_similar(
        &self,
        embedding: &[f32],
        threshold: f32,
        limit: usize,
    ) -> Result<Vec<SearchResult>> {
        // For SQLite, we need to load all embeddings and compute similarity in Rust
        // For production with large datasets, consider using vector search extensions
        let rows = sqlx::query(
            r#"
            SELECT face_id, person_id, person_name, embedding
            FROM faces
            "#,
        )
        .fetch_all(&self.pool)
        .await?;

        let mut results: Vec<SearchResult> = rows
            .into_iter()
            .filter_map(|row| {
                let stored_embedding: Vec<u8> = row.get("embedding");
                let stored: Vec<f32> = stored_embedding
                    .chunks_exact(4)
                    .map(|chunk| {
                        let arr: [u8; 4] = chunk.try_into().unwrap();
                        f32::from_le_bytes(arr)
                    })
                    .collect();

                let similarity = cosine_similarity(embedding, &stored);
                
                if similarity >= threshold {
                    Some(SearchResult {
                        face_id: row.get("face_id"),
                        person_id: row.get("person_id"),
                        person_name: row.get("person_name"),
                        similarity,
                    })
                } else {
                    None
                }
            })
            .collect();

        // Sort by similarity descending
        results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());

        // Limit results
        results.truncate(limit);

        Ok(results)
    }

    async fn list_faces(&self, offset: i64, limit: i64) -> Result<Vec<FaceRecord>> {
        let rows = sqlx::query(
            r#"
            SELECT face_id, person_id, person_name, category, embedding, metadata, created_at, updated_at
            FROM faces
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
            "#,
        )
        .bind(limit)
        .bind(offset)
        .fetch_all(&self.pool)
        .await?;

        let records: Vec<FaceRecord> = rows
            .into_iter()
            .map(|row| FaceRecord {
                face_id: row.get("face_id"),
                person_id: row.get("person_id"),
                person_name: row.get("person_name"),
                category: row.get("category"),
                embedding: row.get("embedding"),
                metadata: row.get("metadata"),
                created_at: row.get("created_at"),
                updated_at: row.get("updated_at"),
            })
            .collect();

        Ok(records)
    }

    async fn count_faces(&self) -> Result<i64> {
        let row = sqlx::query("SELECT COUNT(*) as count FROM faces")
            .fetch_one(&self.pool)
            .await?;

        Ok(row.get("count"))
    }

    async fn update_face(&self, record: &FaceRecord) -> Result<bool> {
        let result = sqlx::query(
            r#"
            UPDATE faces
            SET person_id = ?, person_name = ?, category = ?, embedding = ?, metadata = ?, updated_at = ?
            WHERE face_id = ?
            "#,
        )
        .bind(&record.person_id)
        .bind(&record.person_name)
        .bind(&record.category)
        .bind(&record.embedding)
        .bind(&record.metadata)
        .bind(record.updated_at)
        .bind(&record.face_id)
        .execute(&self.pool)
        .await?;

        Ok(result.rows_affected() > 0)
    }
}

impl SqliteStorage {
    /// List faces by category
    pub async fn list_faces_by_category(&self, category: &str, offset: i64, limit: i64) -> Result<Vec<FaceRecord>> {
        let rows = sqlx::query(
            r#"
            SELECT face_id, person_id, person_name, category, embedding, metadata, created_at, updated_at
            FROM faces
            WHERE category = ?
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
            "#,
        )
        .bind(category)
        .bind(limit)
        .bind(offset)
        .fetch_all(&self.pool)
        .await?;

        let records: Vec<FaceRecord> = rows
            .into_iter()
            .map(|row| FaceRecord {
                face_id: row.get("face_id"),
                person_id: row.get("person_id"),
                person_name: row.get("person_name"),
                category: row.get("category"),
                embedding: row.get("embedding"),
                metadata: row.get("metadata"),
                created_at: row.get("created_at"),
                updated_at: row.get("updated_at"),
            })
            .collect();

        Ok(records)
    }

    /// Get all distinct categories
    pub async fn get_categories(&self) -> Result<Vec<String>> {
        let rows = sqlx::query(
            r#"
            SELECT DISTINCT COALESCE(category, 'default') as category
            FROM faces
            ORDER BY category
            "#,
        )
        .fetch_all(&self.pool)
        .await?;

        let categories: Vec<String> = rows
            .into_iter()
            .map(|row| row.get("category"))
            .collect();

        Ok(categories)
    }

    /// Count faces by category
    pub async fn count_faces_by_category(&self, category: &str) -> Result<i64> {
        let row = sqlx::query("SELECT COUNT(*) as count FROM faces WHERE category = ?")
            .bind(category)
            .fetch_one(&self.pool)
            .await?;

        Ok(row.get("count"))
    }
}

/// Task record for history
#[derive(Debug, Clone)]
pub struct TaskRecord {
    pub id: String,
    pub task_type: String,
    pub result: Option<String>,
    pub image_path: Option<String>,
    pub faces_json: Option<String>,
    pub duration_ms: i64,
    pub created_at: i64,
}

impl SqliteStorage {
    /// Save a task record
    pub async fn save_task(&self, task: &TaskRecord) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO tasks (id, task_type, result, image_path, faces_json, duration_ms, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            "#,
        )
        .bind(&task.id)
        .bind(&task.task_type)
        .bind(&task.result)
        .bind(&task.image_path)
        .bind(&task.faces_json)
        .bind(task.duration_ms)
        .bind(task.created_at)
        .execute(&self.pool)
        .await?;

        debug!("Saved task: {}", task.id);
        Ok(())
    }

    /// List recent tasks
    pub async fn list_tasks(&self, limit: i64, task_type: Option<&str>) -> Result<Vec<TaskRecord>> {
        let rows = if let Some(t) = task_type {
            sqlx::query(
                r#"
                SELECT id, task_type, result, image_path, faces_json, duration_ms, created_at
                FROM tasks
                WHERE task_type = ?
                ORDER BY created_at DESC
                LIMIT ?
                "#,
            )
            .bind(t)
            .bind(limit)
            .fetch_all(&self.pool)
            .await?
        } else {
            sqlx::query(
                r#"
                SELECT id, task_type, result, image_path, faces_json, duration_ms, created_at
                FROM tasks
                ORDER BY created_at DESC
                LIMIT ?
                "#,
            )
            .bind(limit)
            .fetch_all(&self.pool)
            .await?
        };

        let records: Vec<TaskRecord> = rows
            .into_iter()
            .map(|row| TaskRecord {
                id: row.get("id"),
                task_type: row.get("task_type"),
                result: row.get("result"),
                image_path: row.get("image_path"),
                faces_json: row.get("faces_json"),
                duration_ms: row.get("duration_ms"),
                created_at: row.get("created_at"),
            })
            .collect();

        Ok(records)
    }

    /// Delete a single task by ID
    pub async fn delete_task(&self, task_id: &str) -> Result<bool> {
        // Get the task first to find image path
        let row = sqlx::query("SELECT image_path FROM tasks WHERE id = ?")
            .bind(task_id)
            .fetch_optional(&self.pool)
            .await?;

        if let Some(row) = row {
            // Delete image file if exists
            if let Some(path) = row.get::<Option<String>, _>("image_path") {
                if let Some(filename) = path.strip_prefix("/data/tasks/") {
                    let file_path = std::path::Path::new("data/tasks").join(filename);
                    let _ = std::fs::remove_file(file_path);
                }
            }
        }

        let result = sqlx::query("DELETE FROM tasks WHERE id = ?")
            .bind(task_id)
            .execute(&self.pool)
            .await?;

        Ok(result.rows_affected() > 0)
    }

    /// Delete all tasks
    pub async fn delete_all_tasks(&self) -> Result<u64> {
        // Delete all task images
        let tasks_dir = std::path::Path::new("data/tasks");
        if tasks_dir.exists() {
            if let Ok(entries) = std::fs::read_dir(tasks_dir) {
                for entry in entries.flatten() {
                    let _ = std::fs::remove_file(entry.path());
                }
            }
        }

        let result = sqlx::query("DELETE FROM tasks")
            .execute(&self.pool)
            .await?;

        Ok(result.rows_affected())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_sqlite_storage() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let storage = SqliteStorage::new(db_path.to_str().unwrap()).await.unwrap();

        // Create embedding
        let embedding: Vec<u8> = vec![1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        let record = FaceRecord {
            face_id: "test-face-001".to_string(),
            person_id: "person-001".to_string(),
            person_name: "Test Person".to_string(),
            category: Some("default".to_string()),
            embedding,
            metadata: Some(r#"{"key": "value"}"#.to_string()),
            created_at: 1234567890,
            updated_at: 1234567890,
        };

        // Save
        storage.save_face(&record).await.unwrap();

        // Get
        let retrieved = storage.get_face("test-face-001").await.unwrap();
        assert!(retrieved.is_some());
        let retrieved = retrieved.unwrap();
        assert_eq!(retrieved.person_name, "Test Person");

        // Count
        let count = storage.count_faces().await.unwrap();
        assert_eq!(count, 1);

        // Delete
        let deleted = storage.delete_face("test-face-001").await.unwrap();
        assert!(deleted);

        // Verify deleted
        let retrieved = storage.get_face("test-face-001").await.unwrap();
        assert!(retrieved.is_none());
    }
}
