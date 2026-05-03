//! Domain-specific file format support
//!
//! This module provides specialized file format support for various scientific domains:
//! - Bioinformatics: FASTA, FASTQ, SAM/BAM, VCF
//! - Geospatial: GeoTIFF, Shapefile, GeoJSON, KML
//! - Astronomical: FITS, VOTable
//!
//! These formats are commonly used in their respective fields and require
//! specialized handling for efficient processing and metadata preservation.

#![allow(dead_code)]
#![allow(missing_docs)]

/// Bioinformatics file formats
///
/// Provides support for common bioinformatics file formats:
/// - FASTA: Nucleotide and protein sequences
/// - FASTQ: Sequences with quality scores
/// - SAM/BAM: Sequence alignment data
/// - VCF: Variant Call Format for genomic variations
pub mod astronomical;
pub mod avro;
pub mod bioinformatics;
pub mod bson;
pub mod cbor;
pub mod feather;
pub mod geospatial;
pub mod msgpack;
pub mod orc;
pub mod orc_lite;
pub mod protobuf;
