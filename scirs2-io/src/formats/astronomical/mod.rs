//! Astronomical file format support
//!
//! This module provides support for file formats commonly used in astronomy,
//! astrophysics, and space science research.
//!
//! ## Supported Formats
//!
//! - **FITS**: Flexible Image Transport System - standard format for astronomical data.
//!   Parsing and decoding are done by the pure-Rust [`fitsio_pure`] crate: images
//!   (including tile-compressed images and gzip-compressed `.fits.gz` files) are
//!   returned with BSCALE/BZERO applied, and table columns are read from the file.
//! - **VOTable**: Virtual Observatory Table format for tabular astronomical data
//!
//! ## Image axis order
//!
//! 2D images are returned with shape `(NAXIS1, NAXIS2)` and indexed `[x, y]`, so
//! `image[[x, y]]` is FITS pixel `(x + 1, y + 1)`. This is the transpose of the
//! `(NAXIS2, NAXIS1)` row-major order used by astropy and fitsio.
//! [`FitsWriter::write_image_2d`] uses the same order, so images round-trip.
//!
//! ## Examples
//!
//! ```rust,no_run
//! use scirs2_io::formats::astronomical::{FitsFile, FitsTableReader, VOTable};
//! use scirs2_core::ndarray::Array2;
//!
//! // Read FITS file
//! let fits = FitsFile::open("hubble_image.fits")?;
//! let header = fits.primaryheader();
//! let image: Array2<f32> = fits.read_image()?;
//!
//! // Access header values
//! let exposure_time = header.get_f64("EXPTIME")?;
//! let telescope = header.get_string("TELESCOP")?;
//!
//! // Read FITS table
//! let tablehdu = fits.gethdu(1)?;
//! let table_reader = FitsTableReader::new(tablehdu.clone())?;
//! let column_data = table_reader.read_column("FLUX")?;
//! # Ok::<(), scirs2_io::error::IoError>(())
//! ```

#![allow(dead_code)]
#![allow(missing_docs)]

use crate::error::{IoError, Result};
use byteorder::{BigEndian, ReadBytesExt, WriteBytesExt};
use fitsio_pure::bintable::{self, BinaryColumnData, BinaryColumnType};
use fitsio_pure::extension::{build_extension_header, ExtensionType};
use fitsio_pure::hdu::{Hdu as RawHdu, HduInfo};
use fitsio_pure::header::{serialize_header, Card};
use fitsio_pure::table::{self, AsciiColumnData};
use fitsio_pure::value::Value;
use scirs2_core::ndarray::{Array2, ShapeBuilder};
use std::borrow::Cow;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// FITS file structure
pub struct FitsFile {
    file_path: String,
    hdus: Vec<HDU>,
}

/// Header Data Unit
#[derive(Debug, Clone)]
pub struct HDU {
    /// HDU type
    pub hdu_type: HDUType,
    /// Header cards
    pub header: FitsHeader,
    /// Data offset in file (in the decompressed stream for `.fits.gz` files)
    pub data_offset: u64,
    /// Data size in bytes, including any binary table heap
    pub data_size: usize,
    /// Where the data unit is and how to decode it
    raw: Arc<RawData>,
}

/// The data source and parsed layout of one HDU.
#[derive(Debug)]
struct RawData {
    source: Arc<Source>,
    hdu: RawHdu,
}

/// The bytes an HDU's data unit is read from.
#[derive(Debug)]
enum Source {
    /// A plain FITS file, read on demand one data unit at a time.
    File(PathBuf),
    /// A decompressed `.fits.gz` file held in memory.
    Memory(Vec<u8>),
}

impl RawData {
    /// The data unit's bytes, with an HDU whose `data_start` indexes them.
    fn data(&self) -> Result<(Cow<'_, [u8]>, RawHdu)> {
        match self.source.as_ref() {
            Source::Memory(bytes) => Ok((Cow::Borrowed(bytes), self.hdu.clone())),
            Source::File(path) => {
                let mut file = File::open(path)
                    .map_err(|_e| IoError::FileNotFound(path.display().to_string()))?;
                file.seek(SeekFrom::Start(self.hdu.data_start as u64))
                    .map_err(|e| IoError::ParseError(format!("Failed to seek to data: {e}")))?;
                let mut bytes = vec![0u8; self.hdu.data_len];
                file.read_exact(&mut bytes)
                    .map_err(|e| IoError::ParseError(format!("Failed to read data: {e}")))?;
                let mut hdu = self.hdu.clone();
                hdu.data_start = 0;
                Ok((Cow::Owned(bytes), hdu))
            }
        }
    }
}

fn fits_error(context: &str, e: fitsio_pure::Error) -> IoError {
    IoError::ParseError(format!("{context}: {e}"))
}

/// HDU types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HDUType {
    /// Primary HDU (must be first)
    Primary,
    /// Image extension (including tile-compressed images)
    Image,
    /// ASCII table extension
    AsciiTable,
    /// Binary table extension
    BinaryTable,
}

/// FITS header
#[derive(Debug, Clone)]
pub struct FitsHeader {
    /// Header cards (key-value pairs)
    cards: Vec<HeaderCard>,
    /// Quick lookup map
    card_map: HashMap<String, usize>,
}

/// Header card (80 characters)
#[derive(Debug, Clone)]
pub struct HeaderCard {
    /// Keyword (up to 8 characters)
    pub keyword: String,
    /// Value
    pub value: CardValue,
    /// Comment
    pub comment: Option<String>,
}

/// FITS header card values
#[derive(Debug, Clone, PartialEq)]
pub enum CardValue {
    /// Boolean value
    Boolean(bool),
    /// Integer value
    Integer(i64),
    /// Floating point value
    Float(f64),
    /// String value
    String(String),
    /// Complex value
    Complex(f64, f64),
    /// No value (comment card)
    None,
}

impl FitsHeader {
    /// Create a new empty header
    pub fn new() -> Self {
        Self {
            cards: Vec::new(),
            card_map: HashMap::new(),
        }
    }

    /// Build a header from parsed cards, leaving out END.
    ///
    /// A `HIERARCH` card is keyed by its name without the prefix
    /// (`ESO DET NDIT`), with its value; a long string continued on `CONTINUE`
    /// cards is joined, and those cards are not kept separately.
    fn from_cards(cards: &[Card]) -> Self {
        let mut header = Self::new();
        let mut in_hierarch_string = false;
        for card in cards.iter().filter(|c| !c.is_end()) {
            if in_hierarch_string && card.keyword_str() == "CONTINUE" {
                continue;
            }
            let (keyword, value, comment) = match fitsio_pure::header::hierarch(card) {
                Some((name, value, comment)) => {
                    match fitsio_pure::header::find_hierarch(cards, name) {
                        Some((value, comment)) => (name.to_string(), Some(value), comment),
                        None => (name.to_string(), Some(value), comment),
                    }
                }
                None => (
                    card.keyword_str().to_string(),
                    card.value.clone(),
                    card.comment.clone(),
                ),
            };
            in_hierarch_string =
                card.keyword_str() == "HIERARCH" && matches!(value, Some(Value::String(_)));
            let value = match value {
                Some(Value::Logical(b)) => CardValue::Boolean(b),
                Some(Value::Integer(i)) => CardValue::Integer(i),
                Some(Value::Float(f)) => CardValue::Float(f),
                Some(Value::String(s)) => CardValue::String(s),
                Some(Value::ComplexInt(re, im)) => CardValue::Complex(re as f64, im as f64),
                Some(Value::ComplexFloat(re, im)) => CardValue::Complex(re, im),
                Some(Value::Undefined) | None => CardValue::None,
            };
            header.add_card(HeaderCard {
                keyword,
                value,
                comment,
            });
        }
        header
    }

    /// Add a card to the header
    pub fn add_card(&mut self, card: HeaderCard) {
        let index = self.cards.len();
        self.card_map.insert(card.keyword.clone(), index);
        self.cards.push(card);
    }

    /// Get a card by keyword
    pub fn get_card(&self, keyword: &str) -> Option<&HeaderCard> {
        self.card_map.get(keyword).map(|&idx| &self.cards[idx])
    }

    /// Get a boolean value
    pub fn get_bool(&self, keyword: &str) -> Result<bool> {
        match self.get_card(keyword) {
            Some(card) => match &card.value {
                CardValue::Boolean(b) => Ok(*b),
                _ => Err(IoError::ParseError(format!(
                    "Keyword {keyword} is not a boolean"
                ))),
            },
            None => Err(IoError::ParseError(format!("Keyword {keyword} not found"))),
        }
    }

    /// Get an integer value
    pub fn get_i64(&self, keyword: &str) -> Result<i64> {
        match self.get_card(keyword) {
            Some(card) => match &card.value {
                CardValue::Integer(i) => Ok(*i),
                _ => Err(IoError::ParseError(format!(
                    "Keyword {keyword} is not an integer"
                ))),
            },
            None => Err(IoError::ParseError(format!("Keyword {keyword} not found"))),
        }
    }

    /// Get a float value
    pub fn get_f64(&self, keyword: &str) -> Result<f64> {
        match self.get_card(keyword) {
            Some(card) => match &card.value {
                CardValue::Float(f) => Ok(*f),
                CardValue::Integer(i) => Ok(*i as f64),
                _ => Err(IoError::ParseError(format!(
                    "Keyword {keyword} is not a number"
                ))),
            },
            None => Err(IoError::ParseError(format!("Keyword {keyword} not found"))),
        }
    }

    /// Get a string value
    pub fn get_string(&self, keyword: &str) -> Result<String> {
        match self.get_card(keyword) {
            Some(card) => match &card.value {
                CardValue::String(s) => Ok(s.clone()),
                _ => Err(IoError::ParseError(format!(
                    "Keyword {keyword} is not a string"
                ))),
            },
            None => Err(IoError::ParseError(format!("Keyword {keyword} not found"))),
        }
    }

    /// Get all cards
    pub fn cards(&self) -> &[HeaderCard] {
        &self.cards
    }

    /// Check if a keyword exists
    pub fn has_keyword(&self, keyword: &str) -> bool {
        self.card_map.contains_key(keyword)
    }
}

impl Default for FitsHeader {
    fn default() -> Self {
        Self::new()
    }
}

/// FITS data types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FitsDataType {
    /// 8-bit unsigned integer
    UInt8,
    /// 16-bit signed integer
    Int16,
    /// 32-bit signed integer
    Int32,
    /// 64-bit signed integer
    Int64,
    /// 32-bit floating point
    Float32,
    /// 64-bit floating point
    Float64,
}

impl FitsDataType {
    /// Get size in bytes
    pub fn byte_size(&self) -> usize {
        match self {
            FitsDataType::UInt8 => 1,
            FitsDataType::Int16 => 2,
            FitsDataType::Int32 => 4,
            FitsDataType::Int64 => 8,
            FitsDataType::Float32 => 4,
            FitsDataType::Float64 => 8,
        }
    }

    /// Get BITPIX value
    pub fn bitpix(&self) -> i32 {
        match self {
            FitsDataType::UInt8 => 8,
            FitsDataType::Int16 => 16,
            FitsDataType::Int32 => 32,
            FitsDataType::Int64 => 64,
            FitsDataType::Float32 => -32,
            FitsDataType::Float64 => -64,
        }
    }

    /// From BITPIX value
    pub fn frombitpix(bitpix: i32) -> Result<Self> {
        match bitpix {
            8 => Ok(FitsDataType::UInt8),
            16 => Ok(FitsDataType::Int16),
            32 => Ok(FitsDataType::Int32),
            64 => Ok(FitsDataType::Int64),
            -32 => Ok(FitsDataType::Float32),
            -64 => Ok(FitsDataType::Float64),
            _ => Err(IoError::ParseError(format!(
                "Invalid BITPIX value: {bitpix}"
            ))),
        }
    }
}

impl FitsFile {
    /// Open a FITS file, plain or gzip-compressed.
    ///
    /// Only the headers are read here; a plain file's data units are read when
    /// they are asked for. A gzip-compressed file is decompressed into memory.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let file_path = path.to_string_lossy().to_string();
        let mut file = File::open(path).map_err(|_e| IoError::FileNotFound(file_path.clone()))?;

        let mut magic = [0u8; 2];
        let is_gzip = file.read_exact(&mut magic).is_ok() && fitsio_pure::gzip::is_gzip(&magic);

        let (source, parsed) = if is_gzip {
            let mut compressed = Vec::new();
            file.seek(SeekFrom::Start(0))
                .and_then(|_| file.read_to_end(&mut compressed))
                .map_err(|e| IoError::FileError(format!("Failed to read {file_path}: {e}")))?;
            let bytes = fitsio_pure::gzip::decompress(&compressed)
                .map_err(|e| IoError::DecompressionError(format!("{file_path}: {e}")))?;
            let parsed = fitsio_pure::hdu::parse_fits(&bytes)
                .map_err(|e| fits_error("Failed to parse FITS", e))?;
            (Source::Memory(bytes), parsed)
        } else {
            let len = file
                .metadata()
                .map_err(|e| IoError::FileError(format!("Failed to stat {file_path}: {e}")))?
                .len() as usize;
            let parsed = fitsio_pure::hdu::parse_fits_headers(len, |offset, buf| {
                file.seek(SeekFrom::Start(offset as u64))
                    .and_then(|_| file.read_exact(buf))
                    .map_err(fitsio_pure::Error::from)
            })
            .map_err(|e| fits_error("Failed to parse FITS", e))?;
            (Source::File(path.to_path_buf()), parsed)
        };

        let source = Arc::new(source);
        let hdus = parsed
            .hdus
            .into_iter()
            .enumerate()
            .map(|(index, hdu)| {
                let hdu_type = match hdu.info {
                    _ if index == 0 => HDUType::Primary,
                    HduInfo::AsciiTable { .. } => HDUType::AsciiTable,
                    HduInfo::BinaryTable { .. } => HDUType::BinaryTable,
                    _ => HDUType::Image,
                };
                // A tile-compressed image is presented with the header of the
                // image it holds, as cfitsio and astropy present it.
                let header = match hdu.info {
                    HduInfo::CompressedImage { .. } => FitsHeader::from_cards(
                        &fitsio_pure::tiled::uncompressed_header(&hdu)
                            .map_err(|e| fits_error("Failed to read compressed image header", e))?,
                    ),
                    _ => FitsHeader::from_cards(&hdu.cards),
                };
                Ok(HDU {
                    hdu_type,
                    header,
                    data_offset: hdu.data_start as u64,
                    data_size: hdu.data_len,
                    raw: Arc::new(RawData {
                        source: Arc::clone(&source),
                        hdu,
                    }),
                })
            })
            .collect::<Result<Vec<HDU>>>()?;

        Ok(Self { file_path, hdus })
    }

    /// Get primary header
    pub fn primaryheader(&self) -> &FitsHeader {
        &self.hdus[0].header
    }

    /// Get number of HDUs
    pub fn hdu_count(&self) -> usize {
        self.hdus.len()
    }

    /// Get HDU by index
    pub fn gethdu(&self, index: usize) -> Result<&HDU> {
        self.hdus
            .get(index)
            .ok_or_else(|| IoError::ParseError(format!("HDU index {index} out of range")))
    }

    /// Read primary image as 2D array
    pub fn read_image<T: FitsNumeric>(&self) -> Result<Array2<T>> {
        self.readhdu_image(0)
    }

    /// Read image from specific HDU, with BSCALE/BZERO applied and BLANK
    /// pixels as NaN.
    ///
    /// The result has shape `(NAXIS1, NAXIS2)`, indexed `[x, y]` (see the
    /// module documentation). Images with more than two axes are accepted
    /// when every axis past the second has length 1.
    pub fn readhdu_image<T: FitsNumeric>(&self, hduindex: usize) -> Result<Array2<T>> {
        let hdu = self.gethdu(hduindex)?;

        if hdu.hdu_type != HDUType::Primary && hdu.hdu_type != HDUType::Image {
            return Err(IoError::ParseError(format!(
                "HDU {hduindex} is not an image"
            )));
        }

        let dims = self.image_dimensions(hduindex)?;
        if dims.len() < 2 || dims[2..].iter().any(|&n| n != 1) {
            return Err(IoError::ParseError(format!(
                "Expected 2D image, got dimensions {dims:?}"
            )));
        }
        let (naxis1, naxis2) = (dims[0], dims[1]);

        let (bytes, raw) = hdu.raw.data()?;
        let physical = fitsio_pure::image::read_image_physical(&bytes, &raw)
            .map_err(|e| fits_error("Failed to read image", e))?;
        let values = physical
            .into_iter()
            .map(T::from_physical)
            .collect::<Result<Vec<T>>>()?;

        // FITS stores NAXIS1 fastest, which is column-major for shape (NAXIS1, NAXIS2).
        Array2::from_shape_vec((naxis1, naxis2).f(), values)
            .map_err(|e| IoError::ParseError(format!("Failed to create array: {e}")))
    }

    /// Get image dimensions (NAXIS1, NAXIS2, ...); for a tile-compressed image,
    /// the dimensions of the image rather than of the table holding it.
    pub fn image_dimensions(&self, hduindex: usize) -> Result<Vec<usize>> {
        let hdu = self.gethdu(hduindex)?;
        if let Ok(dims) = fitsio_pure::image::image_dimensions(&hdu.raw.hdu) {
            return Ok(dims);
        }
        let naxis = hdu.header.get_i64("NAXIS")? as usize;

        let mut dims = Vec::with_capacity(naxis);
        for i in 1..=naxis {
            let axis_key = format!("NAXIS{i}");
            let size = hdu.header.get_i64(&axis_key)? as usize;
            dims.push(size);
        }

        Ok(dims)
    }
}

/// Trait for numeric types supported by FITS
pub trait FitsNumeric: Default + Clone {
    fn read_fits<R: Read>(reader: &mut R, datatype: FitsDataType) -> Result<Self>;
    fn write_fits<W: Write>(&self, writer: &mut W, datatype: FitsDataType) -> Result<()>;

    /// Convert a physical pixel value (BSCALE/BZERO already applied).
    fn from_physical(value: f64) -> Result<Self> {
        Self::read_fits(&mut &value.to_be_bytes()[..], FitsDataType::Float64)
    }
}

impl FitsNumeric for f32 {
    fn read_fits<R: Read>(reader: &mut R, datatype: FitsDataType) -> Result<Self> {
        match datatype {
            FitsDataType::Float32 => reader
                .read_f32::<BigEndian>()
                .map_err(|e| IoError::ParseError(format!("Failed to read f32: {e}"))),
            FitsDataType::Float64 => reader
                .read_f64::<BigEndian>()
                .map(|v| v as f32)
                .map_err(|e| IoError::ParseError(format!("Failed to read f64: {e}"))),
            FitsDataType::Int16 => reader
                .read_i16::<BigEndian>()
                .map(|v| v as f32)
                .map_err(|e| IoError::ParseError(format!("Failed to read i16: {e}"))),
            FitsDataType::Int32 => reader
                .read_i32::<BigEndian>()
                .map(|v| v as f32)
                .map_err(|e| IoError::ParseError(format!("Failed to read i32: {e}"))),
            _ => Err(IoError::ParseError(format!(
                "Unsupported conversion from {datatype:?} to f32"
            ))),
        }
    }

    fn write_fits<W: Write>(&self, writer: &mut W, datatype: FitsDataType) -> Result<()> {
        match datatype {
            FitsDataType::Float32 => writer
                .write_f32::<BigEndian>(*self)
                .map_err(|e| IoError::FileError(format!("Failed to write f32: {e}"))),
            _ => Err(IoError::FileError(format!(
                "Unsupported conversion from f32 to {datatype:?}"
            ))),
        }
    }
}

impl FitsNumeric for f64 {
    fn read_fits<R: Read>(reader: &mut R, datatype: FitsDataType) -> Result<Self> {
        match datatype {
            FitsDataType::Float64 => reader
                .read_f64::<BigEndian>()
                .map_err(|e| IoError::ParseError(format!("Failed to read f64: {e}"))),
            FitsDataType::Float32 => reader
                .read_f32::<BigEndian>()
                .map(|v| v as f64)
                .map_err(|e| IoError::ParseError(format!("Failed to read f32: {e}"))),
            FitsDataType::Int32 => reader
                .read_i32::<BigEndian>()
                .map(|v| v as f64)
                .map_err(|e| IoError::ParseError(format!("Failed to read i32: {e}"))),
            _ => Err(IoError::ParseError(format!(
                "Unsupported conversion from {datatype:?} to f64"
            ))),
        }
    }

    fn write_fits<W: Write>(&self, writer: &mut W, datatype: FitsDataType) -> Result<()> {
        match datatype {
            FitsDataType::Float64 => writer
                .write_f64::<BigEndian>(*self)
                .map_err(|e| IoError::FileError(format!("Failed to write f64: {e}"))),
            _ => Err(IoError::FileError(format!(
                "Unsupported conversion from f64 to {datatype:?}"
            ))),
        }
    }
}

/// FITS file writer
pub struct FitsWriter {
    writer: BufWriter<File>,
    currenthdu: usize,
}

impl FitsWriter {
    /// Create a new FITS file
    pub fn create<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::create(path.as_ref())
            .map_err(|e| IoError::FileError(format!("Failed to create file: {e}")))?;

        Ok(Self {
            writer: BufWriter::new(file),
            currenthdu: 0,
        })
    }

    /// Write a 2D image HDU: the primary HDU first, then IMAGE extensions.
    ///
    /// `data` is indexed `[x, y]`: its shape `(n0, n1)` is written as
    /// `NAXIS1 = n0`, `NAXIS2 = n1`, the order [`FitsFile::read_image`] returns.
    pub fn write_image_2d<T: FitsNumeric>(
        &mut self,
        data: &Array2<T>,
        datatype: FitsDataType,
    ) -> Result<()> {
        let (naxis1, naxis2) = data.dim();
        let bitpix = datatype.bitpix() as i64;
        let naxes = [naxis1, naxis2];

        let cards = if self.currenthdu == 0 {
            let mut cards = fitsio_pure::primary::build_primary_header(bitpix, &naxes)
                .map_err(|e| IoError::FileError(format!("Failed to build header: {e}")))?;
            cards.push(
                Card::new("EXTEND", Value::Logical(true))
                    .map_err(|e| IoError::FileError(format!("Failed to build header: {e}")))?,
            );
            cards
        } else {
            build_extension_header(ExtensionType::Image, bitpix, &naxes, 0, 1)
                .map_err(|e| IoError::FileError(format!("Failed to build header: {e}")))?
        };
        let header = serialize_header(&cards)
            .map_err(|e| IoError::FileError(format!("Failed to build header: {e}")))?;
        self.writer
            .write_all(&header)
            .map_err(|e| IoError::FileError(format!("Failed to write header: {e}")))?;

        // FITS order: NAXIS1 (x) varies fastest.
        for y in 0..naxis2 {
            for x in 0..naxis1 {
                data[[x, y]].write_fits(&mut self.writer, datatype)?;
            }
        }

        // Pad to 2880-byte boundary
        let data_bytes = naxis1 * naxis2 * datatype.byte_size();
        let padding = (2880 - (data_bytes % 2880)) % 2880;
        if padding > 0 {
            let pad_bytes = vec![0u8; padding];
            self.writer
                .write_all(&pad_bytes)
                .map_err(|e| IoError::FileError(format!("Failed to write padding: {e}")))?;
        }

        self.currenthdu += 1;

        Ok(())
    }

    /// Flush the writer
    pub fn flush(&mut self) -> Result<()> {
        self.writer
            .flush()
            .map_err(|e| IoError::FileError(format!("Failed to flush: {e}")))
    }

    /// Close the file
    pub fn close(mut self) -> Result<()> {
        self.flush()
    }
}

/// VOTable (Virtual Observatory Table) support
pub struct VOTable {
    /// Table metadata
    pub metadata: HashMap<String, String>,
    /// Column definitions
    pub columns: Vec<VOTableColumn>,
    /// Table data
    pub data: Vec<Vec<VOTableValue>>,
}

/// VOTable column definition
#[derive(Debug, Clone)]
pub struct VOTableColumn {
    /// Column name
    pub name: String,
    /// Data type
    pub datatype: String,
    /// Array size (for array columns)
    pub arraysize: Option<String>,
    /// Unit
    pub unit: Option<String>,
    /// Description
    pub description: Option<String>,
    /// UCD (Unified Content Descriptor)
    pub ucd: Option<String>,
}

/// VOTable value types
#[derive(Debug, Clone, PartialEq)]
pub enum VOTableValue {
    /// Boolean value
    Boolean(bool),
    /// Integer value
    Integer(i64),
    /// Float value
    Float(f64),
    /// Double value
    Double(f64),
    /// String value
    String(String),
    /// Array of values
    Array(Vec<VOTableValue>),
    /// Null value
    Null,
}

impl VOTable {
    /// Create a new empty VOTable
    pub fn new() -> Self {
        Self {
            metadata: HashMap::new(),
            columns: Vec::new(),
            data: Vec::new(),
        }
    }

    /// Add a column definition
    pub fn add_column(&mut self, column: VOTableColumn) {
        self.columns.push(column);
    }

    /// Add a row of data
    pub fn add_row(&mut self, row: Vec<VOTableValue>) -> Result<()> {
        if row.len() != self.columns.len() {
            return Err(IoError::FileError(format!(
                "Row has {} values but table has {} columns",
                row.len(),
                self.columns.len()
            )));
        }
        self.data.push(row);
        Ok(())
    }

    /// Get column by name
    pub fn get_column(&self, name: &str) -> Option<usize> {
        self.columns.iter().position(|c| c.name == name)
    }

    /// Get column data
    pub fn get_column_data(&self, columnindex: usize) -> Result<Vec<&VOTableValue>> {
        if columnindex >= self.columns.len() {
            return Err(IoError::ParseError(format!(
                "Column _index {columnindex} out of range"
            )));
        }

        Ok(self.data.iter().map(|row| &row[columnindex]).collect())
    }

    /// Read VOTable from XML file (simplified)
    pub fn read<P: AsRef<Path>>(path: P) -> Result<Self> {
        // Simplified implementation
        // In reality, would use an XML parser
        Ok(Self::new())
    }

    /// Write VOTable to XML file (simplified)
    pub fn write<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let file = File::create(path.as_ref())
            .map_err(|e| IoError::FileError(format!("Failed to create file: {e}")))?;
        let mut writer = BufWriter::new(file);

        // Write XML header
        writeln!(writer, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>")
            .map_err(|e| IoError::FileError(format!("Failed to write XML header: {e}")))?;
        writeln!(
            writer,
            "<VOTABLE version=\"1.4\" xmlns=\"http://www.ivoa.net/xml/VOTable/v1.3\">"
        )
        .map_err(|e| IoError::FileError(format!("Failed to write VOTABLE tag: {e}")))?;

        // Write resource
        writeln!(writer, "  <RESOURCE>")
            .map_err(|e| IoError::FileError(format!("Failed to write RESOURCE tag: {e}")))?;
        writeln!(writer, "    <TABLE>")
            .map_err(|e| IoError::FileError(format!("Failed to write TABLE tag: {e}")))?;

        // Write fields
        for column in &self.columns {
            write!(
                writer,
                "      <FIELD name=\"{}\" datatype=\"{}\"",
                column.name, column.datatype
            )
            .map_err(|e| IoError::FileError(format!("Failed to write FIELD: {e}")))?;

            if let Some(unit) = &column.unit {
                write!(writer, " unit=\"{unit}\"")
                    .map_err(|e| IoError::FileError(format!("Failed to write unit: {e}")))?;
            }

            writeln!(writer, "/>")
                .map_err(|e| IoError::FileError(format!("Failed to write FIELD close: {e}")))?;
        }

        // Write data
        writeln!(writer, "      <DATA>")
            .map_err(|e| IoError::FileError(format!("Failed to write DATA tag: {e}")))?;
        writeln!(writer, "        <TABLEDATA>")
            .map_err(|e| IoError::FileError(format!("Failed to write TABLEDATA tag: {e}")))?;

        for row in &self.data {
            write!(writer, "          <TR>")
                .map_err(|e| IoError::FileError(format!("Failed to write TR: {e}")))?;

            for value in row {
                match value {
                    VOTableValue::String(s) => write!(writer, "<TD>{s}</TD>"),
                    VOTableValue::Integer(i) => write!(writer, "<TD>{i}</TD>"),
                    VOTableValue::Float(f) | VOTableValue::Double(f) => {
                        write!(writer, "<TD>{f}</TD>")
                    }
                    VOTableValue::Boolean(b) => {
                        write!(writer, "<TD>{}</TD>", if *b { "true" } else { "false" })
                    }
                    VOTableValue::Null => write!(writer, "<TD/>"),
                    VOTableValue::Array(_) => write!(writer, "<TD>[]</TD>"), // Simplified
                }
                .map_err(|e| IoError::FileError(format!("Failed to write TD: {e}")))?;
            }

            writeln!(writer, "</TR>")
                .map_err(|e| IoError::FileError(format!("Failed to write TR close: {e}")))?;
        }

        writeln!(writer, "        </TABLEDATA>")
            .map_err(|e| IoError::FileError(format!("Failed to close TABLEDATA: {e}")))?;
        writeln!(writer, "      </DATA>")
            .map_err(|e| IoError::FileError(format!("Failed to close DATA: {e}")))?;
        writeln!(writer, "    </TABLE>")
            .map_err(|e| IoError::FileError(format!("Failed to close TABLE: {e}")))?;
        writeln!(writer, "  </RESOURCE>")
            .map_err(|e| IoError::FileError(format!("Failed to close RESOURCE: {e}")))?;
        writeln!(writer, "</VOTABLE>")
            .map_err(|e| IoError::FileError(format!("Failed to close VOTABLE: {e}")))?;

        writer
            .flush()
            .map_err(|e| IoError::FileError(format!("Failed to flush: {e}")))
    }
}

impl Default for VOTable {
    fn default() -> Self {
        Self::new()
    }
}

/// Coordinate transformation utilities for astronomical data
#[derive(Debug, Clone)]
pub struct GeoTransform {
    /// Reference longitude (degrees)
    pub ref_lon: f64,
    /// Reference latitude (degrees)
    pub ref_lat: f64,
    /// Pixel scale in longitude direction (degrees per pixel)
    pub lon_scale: f64,
    /// Pixel scale in latitude direction (degrees per pixel)
    pub latscale: f64,
}

impl GeoTransform {
    /// Create a new coordinate transformation
    pub fn new(ref_lon: f64, ref_lat: f64, lon_scale: f64, latscale: f64) -> Self {
        Self {
            ref_lon,
            ref_lat,
            lon_scale,
            latscale,
        }
    }

    /// Convert pixel coordinates to celestial coordinates
    pub fn pixel_to_geo(&self, px: f64, py: f64) -> (f64, f64) {
        let lon = self.ref_lon + px * self.lon_scale;
        let lat = self.ref_lat + py * self.latscale;
        (lon, lat)
    }

    /// Convert celestial coordinates to pixel coordinates
    pub fn geo_to_pixel(&self, lon: f64, lat: f64) -> (f64, f64) {
        let px = (lon - self.ref_lon) / self.lon_scale;
        let py = (lat - self.ref_lat) / self.latscale;
        (px, py)
    }

    /// Apply World Coordinate System transformation
    pub fn apply_wcs(&self, wcs: &WCSTransform) -> GeoTransform {
        Self {
            ref_lon: wcs.crval1,
            ref_lat: wcs.crval2,
            lon_scale: wcs.cdelt1,
            latscale: wcs.cdelt2,
        }
    }
}

/// World Coordinate System parameters
///
/// Pixel coordinates are 1-based FITS pixel coordinates. For `RA---TAN` /
/// `DEC--TAN` axes the gnomonic (TAN) projection is applied with the default
/// `LONPOLE`; other axis types are mapped linearly. SIP and other distortion
/// terms are not applied.
#[derive(Debug, Clone)]
pub struct WCSTransform {
    /// Reference coordinate value at reference pixel (RA)
    pub crval1: f64,
    /// Reference coordinate value at reference pixel (Dec)
    pub crval2: f64,
    /// Reference pixel coordinate
    pub crpix1: f64,
    /// Reference pixel coordinate
    pub crpix2: f64,
    /// Coordinate increment at reference pixel
    pub cdelt1: f64,
    /// Coordinate increment at reference pixel
    pub cdelt2: f64,
    /// Linear transformation matrix (degrees per pixel): from CDi_j, or from
    /// PCi_j or CROTA2 scaled by CDELTi. When `None`, `diag(cdelt1, cdelt2)`.
    pub cd_matrix: Option<[[f64; 2]; 2]>,
    /// Coordinate type (e.g., "RA---TAN", "DEC--TAN")
    pub ctype1: String,
    /// Coordinate type
    pub ctype2: String,
}

impl WCSTransform {
    /// Create WCS transform from FITS header
    pub fn from_fitsheader(header: &FitsHeader) -> Result<Self> {
        let cdelt1 = header.get_f64("CDELT1").unwrap_or(1.0);
        let cdelt2 = header.get_f64("CDELT2").unwrap_or(1.0);
        let matrix = |prefix: &str, identity: bool| {
            let keys = [["1_1", "1_2"], ["2_1", "2_2"]];
            if !keys
                .iter()
                .flatten()
                .any(|k| header.has_keyword(&format!("{prefix}{k}")))
            {
                return None;
            }
            let mut m = [[0.0; 2]; 2];
            for (i, row) in keys.iter().enumerate() {
                for (j, k) in row.iter().enumerate() {
                    let default = if identity && i == j { 1.0 } else { 0.0 };
                    m[i][j] = header.get_f64(&format!("{prefix}{k}")).unwrap_or(default);
                }
            }
            Some(m)
        };
        let cd_matrix = matrix("CD", false)
            .or_else(|| {
                matrix("PC", true).map(|pc| {
                    [
                        [cdelt1 * pc[0][0], cdelt1 * pc[0][1]],
                        [cdelt2 * pc[1][0], cdelt2 * pc[1][1]],
                    ]
                })
            })
            .or_else(|| {
                let rho = header.get_f64("CROTA2").ok()?.to_radians();
                Some([
                    [cdelt1 * rho.cos(), -cdelt2 * rho.sin()],
                    [cdelt1 * rho.sin(), cdelt2 * rho.cos()],
                ])
            });
        Ok(Self {
            crval1: header.get_f64("CRVAL1").unwrap_or(0.0),
            crval2: header.get_f64("CRVAL2").unwrap_or(0.0),
            crpix1: header.get_f64("CRPIX1").unwrap_or(1.0),
            crpix2: header.get_f64("CRPIX2").unwrap_or(1.0),
            cdelt1,
            cdelt2,
            cd_matrix,
            ctype1: header.get_string("CTYPE1").unwrap_or_default(),
            ctype2: header.get_string("CTYPE2").unwrap_or_default(),
        })
    }

    fn matrix(&self) -> [[f64; 2]; 2] {
        self.cd_matrix
            .unwrap_or([[self.cdelt1, 0.0], [0.0, self.cdelt2]])
    }

    fn is_tan(&self) -> bool {
        self.ctype1.trim_end().ends_with("-TAN") && self.ctype2.trim_end().ends_with("-TAN")
    }

    /// Native longitude of the celestial pole: the default LONPOLE for a
    /// zenithal projection.
    fn lonpole(&self) -> f64 {
        if self.crval2 >= 90.0 {
            0.0
        } else {
            180.0
        }
    }

    /// Convert pixel coordinates to world coordinates
    pub fn pixel_to_world(&self, px: f64, py: f64) -> (f64, f64) {
        let m = self.matrix();
        let dx = px - self.crpix1;
        let dy = py - self.crpix2;
        let x = m[0][0] * dx + m[0][1] * dy;
        let y = m[1][0] * dx + m[1][1] * dy;

        if !self.is_tan() {
            return (self.crval1 + x, self.crval2 + y);
        }

        // Gnomonic projection (Calabretta & Greisen 2002, eqs. 54, 2).
        let phi = x.atan2(-y);
        let theta = (180.0 / std::f64::consts::PI).atan2(x.hypot(y));
        let (dec0, ra0) = (self.crval2.to_radians(), self.crval1);
        let dphi = phi - self.lonpole().to_radians();
        let ra = ra0
            + (-theta.cos() * dphi.sin())
                .atan2(theta.sin() * dec0.cos() - theta.cos() * dec0.sin() * dphi.cos())
                .to_degrees();
        let dec = (theta.sin() * dec0.sin() + theta.cos() * dec0.cos() * dphi.cos())
            .clamp(-1.0, 1.0)
            .asin()
            .to_degrees();
        (ra.rem_euclid(360.0), dec)
    }

    /// Convert world coordinates to pixel coordinates. For a TAN projection,
    /// points 90 degrees or more from the reference point give NaN.
    pub fn world_to_pixel(&self, ra: f64, dec: f64) -> (f64, f64) {
        let (x, y) = if self.is_tan() {
            let (ra, dec) = (ra.to_radians(), dec.to_radians());
            let (ra0, dec0) = (self.crval1.to_radians(), self.crval2.to_radians());
            let dra = ra - ra0;
            // cos(theta) sin(phi - lonpole) and cos(theta) cos(phi - lonpole).
            let a = -dec.cos() * dra.sin();
            let b = dec.sin() * dec0.cos() - dec.cos() * dec0.sin() * dra.cos();
            let phi = self.lonpole().to_radians() + a.atan2(b);
            let sin_theta = dec.sin() * dec0.sin() + dec.cos() * dec0.cos() * dra.cos();
            if sin_theta <= 0.0 {
                return (f64::NAN, f64::NAN);
            }
            // hypot(a, b) keeps cos(theta) precise near the reference point.
            let r = a.hypot(b) / sin_theta * (180.0 / std::f64::consts::PI);
            (r * phi.sin(), -r * phi.cos())
        } else {
            (ra - self.crval1, dec - self.crval2)
        };

        let m = self.matrix();
        let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
        let dx = (m[1][1] * x - m[0][1] * y) / det;
        let dy = (-m[1][0] * x + m[0][0] * y) / det;

        (self.crpix1 + dx, self.crpix2 + dy)
    }
}

/// FITS table column reader
pub struct FitsTableReader {
    hdu: HDU,
}

impl FitsTableReader {
    /// Create a new table reader
    pub fn new(hdu: HDU) -> Result<Self> {
        match hdu.hdu_type {
            HDUType::AsciiTable | HDUType::BinaryTable => Ok(Self { hdu }),
            _ => Err(IoError::ParseError("HDU is not a table".to_string())),
        }
    }

    /// Read a column by name (matched exactly, then ignoring case).
    ///
    /// Each row gives one value, or a [`VOTableValue::Array`] for columns with a
    /// repeat count above 1 and variable-length arrays. TSCALn/TZEROn are
    /// applied, and integer values equal to TNULLn are [`VOTableValue::Null`].
    /// Integer columns stay integers when the scaling keeps them integral
    /// (e.g. unsigned columns stored with TZERO); otherwise scaled values are
    /// [`VOTableValue::Double`]. Complex values are two-element arrays.
    pub fn read_column(&self, columnname: &str) -> Result<Vec<VOTableValue>> {
        let names = self.get_columnnames()?;
        let index = names
            .iter()
            .position(|n| n == columnname)
            .or_else(|| {
                names
                    .iter()
                    .position(|n| n.eq_ignore_ascii_case(columnname))
            })
            .ok_or_else(|| IoError::ParseError(format!("Column '{columnname}' not found")))?;

        let (bytes, raw) = self.hdu.raw.data()?;
        let cards = &raw.cards;
        let number = index + 1;
        let (tscal, tzero) = bintable::extract_column_scaling(cards, number);
        let scaling = Scaling::new(tscal, tzero, column_tnull(cards, number));
        let error =
            |e: fitsio_pure::Error| fits_error(&format!("Failed to read column '{columnname}'"), e);

        if self.hdu.hdu_type == HDUType::AsciiTable {
            let data = table::read_ascii_column(&bytes, &raw, index).map_err(error)?;
            return Ok(match data {
                AsciiColumnData::Character(v) => v.into_iter().map(VOTableValue::String).collect(),
                AsciiColumnData::Integer(v) => v.into_iter().map(|i| scaling.int(i)).collect(),
                AsciiColumnData::Float(v) => {
                    v.into_iter().map(|f| scaling.float(f, true)).collect()
                }
            });
        }

        let (nrows, tfields) = match raw.info {
            HduInfo::BinaryTable {
                naxis2, tfields, ..
            } => (naxis2, tfields),
            _ => return Err(IoError::ParseError("HDU is not a binary table".to_string())),
        };
        let columns = bintable::parse_binary_table_columns(cards, tfields).map_err(error)?;
        let column = &columns[index];
        let repeat = column.repeat;
        let data = match column.col_type {
            BinaryColumnType::VarArrayP(_) | BinaryColumnType::VarArrayQ(_) => {
                bintable::read_binary_column_vla(&bytes, &raw, index)
            }
            _ => bintable::read_binary_column(&bytes, &raw, index),
        }
        .map_err(error)?;

        Ok(match data {
            BinaryColumnData::Logical(v) => rows(v, repeat, nrows, VOTableValue::Boolean),
            BinaryColumnData::Byte(v) => rows(v, repeat, nrows, |x| scaling.int(x as i64)),
            BinaryColumnData::Short(v) => rows(v, repeat, nrows, |x| scaling.int(x as i64)),
            BinaryColumnData::Int(v) => rows(v, repeat, nrows, |x| scaling.int(x as i64)),
            BinaryColumnData::Long(v) => rows(v, repeat, nrows, |x| scaling.int(x)),
            BinaryColumnData::Float(v) => {
                rows(v, repeat, nrows, |x| scaling.float(x as f64, false))
            }
            BinaryColumnData::Double(v) => rows(v, repeat, nrows, |x| scaling.float(x, true)),
            BinaryColumnData::ComplexFloat(v) => rows(v, repeat, nrows, |(re, im)| {
                VOTableValue::Array(vec![
                    VOTableValue::Float(re as f64),
                    VOTableValue::Float(im as f64),
                ])
            }),
            BinaryColumnData::ComplexDouble(v) => rows(v, repeat, nrows, |(re, im)| {
                VOTableValue::Array(vec![VOTableValue::Double(re), VOTableValue::Double(im)])
            }),
            BinaryColumnData::Ascii(v) => v.into_iter().map(VOTableValue::String).collect(),
            BinaryColumnData::Bit(v) => v
                .into_iter()
                .map(|bytes| {
                    let bits: Vec<VOTableValue> = (0..repeat)
                        .map(|b| VOTableValue::Boolean(bytes[b / 8] & (0x80 >> (b % 8)) != 0))
                        .collect();
                    single_or_array(bits, repeat)
                })
                .collect(),
            BinaryColumnData::VarByte(v) => var_rows(v, |x| scaling.int(x as i64)),
            BinaryColumnData::VarShort(v) => var_rows(v, |x| scaling.int(x as i64)),
            BinaryColumnData::VarInt(v) => var_rows(v, |x| scaling.int(x as i64)),
            BinaryColumnData::VarLong(v) => var_rows(v, |x| scaling.int(x)),
            BinaryColumnData::VarFloat(v) => var_rows(v, |x| scaling.float(x as f64, false)),
            BinaryColumnData::VarDouble(v) => var_rows(v, |x| scaling.float(x, true)),
        })
    }

    /// Get column names (TTYPEn; empty for an unnamed column)
    pub fn get_columnnames(&self) -> Result<Vec<String>> {
        let tfields = self.hdu.header.get_i64("TFIELDS")?;
        Ok((1..=tfields)
            .map(|n| {
                self.hdu
                    .header
                    .get_string(&format!("TTYPE{n}"))
                    .map(|s| s.trim_end().to_string())
                    .unwrap_or_default()
            })
            .collect())
    }

    /// Get number of rows
    pub fn get_row_count(&self) -> Result<usize> {
        self.hdu.header.get_i64("NAXIS2").map(|n| n as usize)
    }
}

/// TSCALn/TZEROn/TNULLn of one column.
struct Scaling {
    tscal: f64,
    tzero: f64,
    tnull: Option<i64>,
}

impl Scaling {
    fn new(tscal: f64, tzero: f64, tnull: Option<i64>) -> Self {
        Self {
            tscal,
            tzero,
            tnull,
        }
    }

    /// An integer cell: kept as an integer when TSCAL is 1 and TZERO is a
    /// whole number (the unsigned-integer convention) and the result fits.
    fn int(&self, raw: i64) -> VOTableValue {
        if self.tnull == Some(raw) {
            return VOTableValue::Null;
        }
        if self.tscal == 1.0 && self.tzero.fract() == 0.0 && self.tzero.abs() <= 2f64.powi(64) {
            if let Ok(v) = i64::try_from(raw as i128 + self.tzero as i128) {
                return VOTableValue::Integer(v);
            }
        }
        VOTableValue::Double(self.tzero + self.tscal * raw as f64)
    }

    /// A floating-point cell; single precision stays `Float` when unscaled.
    fn float(&self, raw: f64, double: bool) -> VOTableValue {
        if self.tscal == 1.0 && self.tzero == 0.0 {
            if double {
                VOTableValue::Double(raw)
            } else {
                VOTableValue::Float(raw)
            }
        } else {
            VOTableValue::Double(self.tzero + self.tscal * raw)
        }
    }
}

fn column_tnull(cards: &[Card], number: usize) -> Option<i64> {
    let keyword = format!("TNULL{number}");
    cards
        .iter()
        .find(|c| c.keyword_str() == keyword)
        .and_then(|c| match c.value {
            Some(Value::Integer(n)) => Some(n),
            _ => None,
        })
}

/// One value per row, or an array of `repeat` values per row.
fn rows<T, F: Fn(T) -> VOTableValue>(
    values: Vec<T>,
    repeat: usize,
    nrows: usize,
    f: F,
) -> Vec<VOTableValue> {
    let mut iter = values.into_iter();
    (0..nrows)
        .map(|_| single_or_array(iter.by_ref().take(repeat).map(&f).collect(), repeat))
        .collect()
}

fn var_rows<T, F: Fn(T) -> VOTableValue>(values: Vec<Vec<T>>, f: F) -> Vec<VOTableValue> {
    values
        .into_iter()
        .map(|row| VOTableValue::Array(row.into_iter().map(&f).collect()))
        .collect()
}

fn single_or_array(mut row: Vec<VOTableValue>, repeat: usize) -> VOTableValue {
    if repeat == 1 && row.len() == 1 {
        row.remove(0)
    } else {
        VOTableValue::Array(row)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_fitsheader() {
        let mut header = FitsHeader::new();

        header.add_card(HeaderCard {
            keyword: "SIMPLE".to_string(),
            value: CardValue::Boolean(true),
            comment: Some("Standard FITS".to_string()),
        });

        header.add_card(HeaderCard {
            keyword: "NAXIS".to_string(),
            value: CardValue::Integer(2),
            comment: Some("Number of axes".to_string()),
        });

        header.add_card(HeaderCard {
            keyword: "EXPTIME".to_string(),
            value: CardValue::Float(300.0),
            comment: Some("Exposure time in seconds".to_string()),
        });

        assert!(header.get_bool("SIMPLE").expect("Operation failed"));
        assert_eq!(header.get_i64("NAXIS").expect("Operation failed"), 2);
        assert_eq!(header.get_f64("EXPTIME").expect("Operation failed"), 300.0);
    }

    #[test]
    fn test_geo_transform() {
        let transform = GeoTransform::new(0.0, 90.0, 0.25, -0.25);

        let (lon, lat) = transform.pixel_to_geo(100.0, 100.0);
        assert_eq!(lon, 25.0);
        assert_eq!(lat, 65.0);

        let (px, py) = transform.geo_to_pixel(25.0, 65.0);
        assert!((px - 100.0).abs() < 1e-10);
        assert!((py - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_votable() {
        let mut votable = VOTable::new();

        votable.add_column(VOTableColumn {
            name: "RA".to_string(),
            datatype: "double".to_string(),
            arraysize: None,
            unit: Some("deg".to_string()),
            description: Some("Right Ascension".to_string()),
            ucd: Some("pos.eq.ra".to_string()),
        });

        votable.add_column(VOTableColumn {
            name: "DEC".to_string(),
            datatype: "double".to_string(),
            arraysize: None,
            unit: Some("deg".to_string()),
            description: Some("Declination".to_string()),
            ucd: Some("pos.eq.dec".to_string()),
        });

        votable
            .add_row(vec![
                VOTableValue::Double(180.0),
                VOTableValue::Double(45.0),
            ])
            .expect("Operation failed");

        assert_eq!(votable.columns.len(), 2);
        assert_eq!(votable.data.len(), 1);
        assert_eq!(votable.get_column("RA"), Some(0));
    }

    #[test]
    fn test_fits_write_read() -> Result<()> {
        let temp_file = NamedTempFile::new().expect("Operation failed");
        let path = temp_file.path();

        // Create test data
        let data = Array2::from_shape_fn((10, 20), |(i, j)| (i * 20 + j) as f32);

        // Write FITS
        {
            let mut writer = FitsWriter::create(path)?;
            writer.write_image_2d(&data, FitsDataType::Float32)?;
            writer.close()?;
        }

        // Note: Reading would require a proper FITS implementation
        // This is just a test of the writing interface

        Ok(())
    }
}
