//! Core entity types and structures for information extraction

/// Entity types for named entity recognition
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EntityType {
    /// Person names and personal identifiers
    Person,
    /// Organization names, companies, institutions
    Organization,
    /// Geographic locations, places, addresses
    Location,
    /// Date expressions and temporal references
    Date,
    /// Time expressions and temporal references
    Time,
    /// Monetary amounts and currency references
    Money,
    /// Percentage values and ratios
    Percentage,
    /// Email addresses
    Email,
    /// URL and web addresses
    Url,
    /// Phone numbers and contact information
    Phone,
    /// Custom entity type defined by user
    Custom(String),
    /// Other unspecified entity type
    Other,
}

/// Extracted entity with type and position information
#[derive(Debug, Clone)]
pub struct Entity {
    /// The extracted text content
    pub text: String,
    /// The type of entity detected
    pub entity_type: EntityType,
    /// Start position in the original text
    pub start: usize,
    /// End position in the original text
    pub end: usize,
    /// Confidence score for the extraction (0.0 to 1.0)
    pub confidence: f64,
}

/// Cluster of similar entities
#[derive(Debug, Clone)]
pub struct EntityCluster {
    /// Representative entity for the cluster
    pub representative: Entity,
    /// All entities in the cluster
    pub members: Vec<Entity>,
    /// Type of entities in the cluster
    pub entity_type: EntityType,
    /// Confidence score for the clustering
    pub confidence: f64,
}
