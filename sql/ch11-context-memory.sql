-- ch11:三层记忆中的长期记忆事实源与增量抽取游标。
CREATE TABLE IF NOT EXISTS long_term_memories (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  user_id VARCHAR(64) NOT NULL,
  memory_type VARCHAR(32) NOT NULL,
  subject_key VARCHAR(128) NOT NULL DEFAULT 'general',
  content TEXT NOT NULL,
  content_hash CHAR(64) NOT NULL,
  source_conversation_id BIGINT NULL,
  source_message_ids JSON NULL,
  confidence DOUBLE NOT NULL DEFAULT 1,
  status VARCHAR(16) NOT NULL DEFAULT 'active',
  supersedes_id BIGINT NULL,
  vectorize_status VARCHAR(16) NOT NULL DEFAULT 'pending',
  expires_at DATETIME NULL,
  last_accessed_at DATETIME NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  INDEX idx_memory_user_status (user_id, status),
  INDEX idx_memory_subject (user_id, memory_type, subject_key),
  INDEX idx_memory_hash (content_hash)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS memory_consolidations (
  conversation_id BIGINT PRIMARY KEY,
  last_extracted_msg_id BIGINT NOT NULL DEFAULT 0,
  extracted_user_turns INT NOT NULL DEFAULT 0,
  updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
