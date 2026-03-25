-- Document versioning migration
-- Run this in the Supabase SQL Editor

-- Add versioning columns to the documents (chunks) table
ALTER TABLE documents ADD COLUMN IF NOT EXISTS version INTEGER DEFAULT 1;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS is_latest BOOLEAN DEFAULT true;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS original_filename TEXT;

-- Add versioning columns to the document_registry table
ALTER TABLE document_registry ADD COLUMN IF NOT EXISTS version INTEGER DEFAULT 1;
ALTER TABLE document_registry ADD COLUMN IF NOT EXISTS is_latest BOOLEAN DEFAULT true;
ALTER TABLE document_registry ADD COLUMN IF NOT EXISTS original_filename TEXT;

-- Backfill original_filename for existing rows
UPDATE documents SET original_filename = metadata->>'filename' WHERE original_filename IS NULL;
UPDATE document_registry SET original_filename = filename WHERE original_filename IS NULL;
