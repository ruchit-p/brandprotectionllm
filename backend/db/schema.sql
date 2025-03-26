-- Core Tables
CREATE TABLE brands (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    website_url VARCHAR(255) NOT NULL,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE brand_assets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    brand_id UUID REFERENCES brands(id) ON DELETE CASCADE,
    asset_type VARCHAR(50) NOT NULL, -- logo, product_image, etc.
    file_path VARCHAR(255) NOT NULL,
    mime_type VARCHAR(100),
    vector_id VARCHAR(255), -- Reference to Qdrant
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE brand_social_media (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    brand_id UUID REFERENCES brands(id) ON DELETE CASCADE,
    platform VARCHAR(50) NOT NULL,
    handle VARCHAR(255) NOT NULL,
    url VARCHAR(255) NOT NULL
);

CREATE TABLE brand_keywords (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    brand_id UUID REFERENCES brands(id) ON DELETE CASCADE,
    keyword VARCHAR(255) NOT NULL
);

-- Monitored Sites
CREATE TABLE websites (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    url VARCHAR(255) NOT NULL UNIQUE,
    domain VARCHAR(255) NOT NULL,
    title VARCHAR(255),
    first_discovered_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_checked_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_flagged BOOLEAN DEFAULT FALSE,
    analysis_status VARCHAR(50) DEFAULT 'PENDING',
    analysis_status_message TEXT,
    analysis_completed_at TIMESTAMP
);

CREATE TABLE website_snapshots (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    website_id UUID REFERENCES websites(id) ON DELETE CASCADE,
    html_path VARCHAR(255),
    text_content TEXT,
    screenshot_path VARCHAR(255),
    html_vector_id VARCHAR(255),
    text_vector_id VARCHAR(255),
    screenshot_vector_id VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE website_assets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    website_id UUID REFERENCES websites(id) ON DELETE CASCADE,
    snapshot_id UUID REFERENCES website_snapshots(id) ON DELETE CASCADE,
    asset_type VARCHAR(50) NOT NULL,
    url VARCHAR(255) NOT NULL,
    file_path VARCHAR(255),
    vector_id VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Detection Results
CREATE TABLE detections (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    brand_id UUID REFERENCES brands(id) ON DELETE CASCADE,
    website_id UUID REFERENCES websites(id) ON DELETE CASCADE,
    detection_type VARCHAR(50) NOT NULL, -- html_similarity, image_match, text_similarity
    confidence FLOAT NOT NULL, -- 0.0 to 1.0
    status VARCHAR(50) DEFAULT 'new', -- new, reviewed, dismissed, confirmed
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE detection_evidence (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    detection_id UUID REFERENCES detections(id) ON DELETE CASCADE,
    evidence_type VARCHAR(50) NOT NULL, -- text, image, html
    description TEXT,
    file_path VARCHAR(255),
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Custom Models for AWS Rekognition
CREATE TABLE brand_custom_models (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    brand_id UUID REFERENCES brands(id) ON DELETE CASCADE,
    project_arn VARCHAR(2048) NOT NULL,
    model_version_arn VARCHAR(2048),
    status VARCHAR(50) NOT NULL, -- TRAINING_IN_PROGRESS, TRAINING_COMPLETED, TRAINING_FAILED, RUNNING, STOPPED
    status_message TEXT,
    min_inference_units INTEGER DEFAULT 1,
    task_id VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
