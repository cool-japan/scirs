//! HDF5 file reader implementation
//!
//! Parses HDF5 binary structures: object headers, B-tree nodes, symbol tables,
//! and data storage (contiguous/chunked) to provide read access to groups and datasets.

use crate::error::{IoError, Result};
use crate::hdf5_lite::superblock::{
    parse_superblock, read_length, read_offset, undefined_address, SuperblockInfo,
};
use crate::hdf5_lite::types::*;
use byteorder::{LittleEndian, ReadBytesExt};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Cursor, Read, Seek, SeekFrom};
use std::path::Path;

/// HDF5 object header message types
const MSG_NIL: u16 = 0x0000;
const MSG_DATASPACE: u16 = 0x0001;
const MSG_LINK_INFO: u16 = 0x0002;
const MSG_DATATYPE: u16 = 0x0003;
const MSG_FILL_VALUE_OLD: u16 = 0x0004;
const MSG_FILL_VALUE: u16 = 0x0005;
const MSG_LINK: u16 = 0x0006;
const MSG_DATA_LAYOUT: u16 = 0x0008;
const MSG_FILTER_PIPELINE: u16 = 0x000B;
const MSG_ATTRIBUTE: u16 = 0x000C;
const MSG_HEADER_CONTINUATION: u16 = 0x0010;
const MSG_SYMBOL_TABLE: u16 = 0x0011;
const MSG_ATTRIBUTE_INFO: u16 = 0x0015;

/// Pure Rust HDF5 file reader
pub struct Hdf5Reader {
    data: Vec<u8>,
    sb: SuperblockInfo,
}

/// Internal object header message
#[derive(Debug)]
struct ObjHeaderMsg {
    msg_type: u16,
    data: Vec<u8>,
}

/// Internal: parsed object header
#[derive(Debug)]
struct ObjectHeader {
    messages: Vec<ObjHeaderMsg>,
}

/// Internal: storage layout
#[derive(Debug)]
enum DataLayout {
    Contiguous {
        address: u64,
        size: u64,
    },
    Chunked {
        address: u64,
        ndims: u8,
        chunk_dims: Vec<u32>,
    },
    Compact {
        data: Vec<u8>,
    },
}

/// Symbol table entry cache info
#[derive(Debug)]
struct SymbolTableEntry {
    name_offset: u64,
    object_header_address: u64,
    cache_type: u32,
    #[allow(dead_code)]
    btree_address: u64,
    #[allow(dead_code)]
    name_heap_address: u64,
}

impl Hdf5Reader {
    /// Open an HDF5 file for reading
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        if !path.exists() {
            return Err(IoError::FileNotFound(path.display().to_string()));
        }
        let file = File::open(path).map_err(|e| {
            IoError::FileError(format!("Cannot open HDF5 file '{}': {e}", path.display()))
        })?;
        let mut buf = BufReader::new(file);
        let mut data = Vec::new();
        buf.read_to_end(&mut data).map_err(|e| {
            IoError::FileError(format!("Cannot read HDF5 file '{}': {e}", path.display()))
        })?;

        let mut cursor = Cursor::new(&data);
        let sb = parse_superblock(&mut cursor)?;

        Ok(Self { data, sb })
    }

    /// Create a reader from in-memory bytes
    pub fn from_bytes(data: Vec<u8>) -> Result<Self> {
        let mut cursor = Cursor::new(&data);
        let sb = parse_superblock(&mut cursor)?;
        Ok(Self { data, sb })
    }

    /// Get the superblock info
    pub fn superblock(&self) -> &SuperblockInfo {
        &self.sb
    }

    /// Read the root group
    pub fn root_group(&self) -> Result<Hdf5Group> {
        self.read_group_at(self.sb.root_group_address, "/")
    }

    /// Read a group at a given path
    pub fn read_group(&self, path: &str) -> Result<Hdf5Group> {
        let parts: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();

        if parts.is_empty() {
            return self.root_group();
        }

        let root = self.root_group()?;
        self.navigate_to_group(&root, &parts, path)
    }

    /// Read a dataset at a given path
    pub fn read_dataset(&self, path: &str) -> Result<Hdf5Dataset> {
        let parts: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();

        if parts.is_empty() {
            return Err(IoError::FormatError(
                "Cannot read dataset at root path".to_string(),
            ));
        }

        let dataset_name = parts
            .last()
            .ok_or_else(|| IoError::FormatError("Empty path".to_string()))?
            .to_string();
        let group_parts = &parts[..parts.len() - 1];

        // Navigate to the parent group
        let parent = if group_parts.is_empty() {
            self.root_group()?
        } else {
            let root = self.root_group()?;
            let group_path = format!("/{}", group_parts.join("/"));
            self.navigate_to_group(&root, group_parts, &group_path)?
        };

        // Find the dataset node address
        let obj_addr = self.find_child_address(self.sb.root_group_address, &parts)?;

        self.read_dataset_at(obj_addr, &dataset_name, path)
    }

    /// List all paths in the file recursively
    pub fn list_all(&self) -> Result<Vec<Hdf5Node>> {
        let mut nodes = Vec::new();
        self.list_recursive(self.sb.root_group_address, "/", &mut nodes)?;
        Ok(nodes)
    }

    /// Recursively list nodes starting from a given address
    fn list_recursive(&self, address: u64, prefix: &str, nodes: &mut Vec<Hdf5Node>) -> Result<()> {
        let header = self.read_object_header(address)?;

        // Look for symbol table message => this is a group
        let sym_msg = header
            .messages
            .iter()
            .find(|m| m.msg_type == MSG_SYMBOL_TABLE);

        let link_msgs: Vec<&ObjHeaderMsg> = header
            .messages
            .iter()
            .filter(|m| m.msg_type == MSG_LINK)
            .collect();

        if let Some(msg) = sym_msg {
            // V1 group with symbol table
            let entries = self.parse_symbol_table_message(&msg.data)?;
            for (name, addr) in &entries {
                let child_path = if prefix == "/" {
                    format!("/{name}")
                } else {
                    format!("{prefix}/{name}")
                };

                // Determine type by trying to read the object header
                let child_header = self.read_object_header(*addr);
                let node_type = match &child_header {
                    Ok(h) => {
                        if h.messages
                            .iter()
                            .any(|m| m.msg_type == MSG_SYMBOL_TABLE || m.msg_type == MSG_LINK)
                        {
                            Hdf5NodeType::Group
                        } else if h.messages.iter().any(|m| m.msg_type == MSG_DATASPACE) {
                            Hdf5NodeType::Dataset
                        } else {
                            Hdf5NodeType::Group
                        }
                    }
                    Err(_) => Hdf5NodeType::Dataset,
                };

                nodes.push(Hdf5Node {
                    name: name.clone(),
                    path: child_path.clone(),
                    node_type: node_type.clone(),
                });

                if node_type == Hdf5NodeType::Group {
                    // Silently ignore errors in recursive traversal
                    let _ = self.list_recursive(*addr, &child_path, nodes);
                }
            }
        } else if !link_msgs.is_empty() {
            // V2 links
            for msg in &link_msgs {
                if let Ok((name, addr)) = self.parse_link_message(&msg.data) {
                    let child_path = if prefix == "/" {
                        format!("/{name}")
                    } else {
                        format!("{prefix}/{name}")
                    };

                    let child_header = self.read_object_header(addr);
                    let node_type = match &child_header {
                        Ok(h) => {
                            if h.messages
                                .iter()
                                .any(|m| m.msg_type == MSG_SYMBOL_TABLE || m.msg_type == MSG_LINK)
                            {
                                Hdf5NodeType::Group
                            } else if h.messages.iter().any(|m| m.msg_type == MSG_DATASPACE) {
                                Hdf5NodeType::Dataset
                            } else {
                                Hdf5NodeType::Group
                            }
                        }
                        Err(_) => Hdf5NodeType::Dataset,
                    };

                    nodes.push(Hdf5Node {
                        name: name.clone(),
                        path: child_path.clone(),
                        node_type: node_type.clone(),
                    });

                    if node_type == Hdf5NodeType::Group {
                        let _ = self.list_recursive(addr, &child_path, nodes);
                    }
                }
            }
        }

        Ok(())
    }

    // =========================================================================
    // Internal helpers
    // =========================================================================

    /// Navigate from root down to a group at the specified path parts
    fn navigate_to_group(
        &self,
        _root: &Hdf5Group,
        parts: &[&str],
        full_path: &str,
    ) -> Result<Hdf5Group> {
        let mut current_addr = self.sb.root_group_address;

        for (i, &part) in parts.iter().enumerate() {
            let header = self.read_object_header(current_addr)?;
            let child_addr = self.find_child_in_header(&header, part)?;
            current_addr = child_addr;

            if i == parts.len() - 1 {
                let group_name = parts.last().copied().unwrap_or("/");
                return self.read_group_at(current_addr, group_name);
            }
        }

        Err(IoError::NotFound(format!("Group not found: {full_path}")))
    }

    /// Find a child address by navigating path parts from root
    fn find_child_address(&self, root_addr: u64, parts: &[&str]) -> Result<u64> {
        let mut current_addr = root_addr;

        for &part in parts {
            let header = self.read_object_header(current_addr)?;
            current_addr = self.find_child_in_header(&header, part)?;
        }

        Ok(current_addr)
    }

    /// Find a named child in an object header (group)
    fn find_child_in_header(&self, header: &ObjectHeader, name: &str) -> Result<u64> {
        // Check symbol table entries
        if let Some(msg) = header
            .messages
            .iter()
            .find(|m| m.msg_type == MSG_SYMBOL_TABLE)
        {
            let entries = self.parse_symbol_table_message(&msg.data)?;
            for (child_name, addr) in &entries {
                if child_name == name {
                    return Ok(*addr);
                }
            }
        }

        // Check link messages
        for msg in header.messages.iter().filter(|m| m.msg_type == MSG_LINK) {
            if let Ok((link_name, addr)) = self.parse_link_message(&msg.data) {
                if link_name == name {
                    return Ok(addr);
                }
            }
        }

        Err(IoError::NotFound(format!(
            "Child '{name}' not found in group"
        )))
    }

    /// Read a group at the given object header address
    fn read_group_at(&self, address: u64, name: &str) -> Result<Hdf5Group> {
        let header = self.read_object_header(address)?;

        let mut children = Vec::new();
        let mut child_nodes = Vec::new();
        let mut attributes = HashMap::new();

        // Collect children from symbol table
        if let Some(msg) = header
            .messages
            .iter()
            .find(|m| m.msg_type == MSG_SYMBOL_TABLE)
        {
            let entries = self.parse_symbol_table_message(&msg.data)?;
            for (child_name, child_addr) in &entries {
                children.push(child_name.clone());

                let child_header = self.read_object_header(*child_addr);
                let node_type = match &child_header {
                    Ok(h) => {
                        if h.messages
                            .iter()
                            .any(|m| m.msg_type == MSG_SYMBOL_TABLE || m.msg_type == MSG_LINK)
                        {
                            Hdf5NodeType::Group
                        } else if h.messages.iter().any(|m| m.msg_type == MSG_DATASPACE) {
                            Hdf5NodeType::Dataset
                        } else {
                            Hdf5NodeType::Group
                        }
                    }
                    Err(_) => Hdf5NodeType::Dataset,
                };

                let child_path = if name == "/" {
                    format!("/{child_name}")
                } else {
                    format!("{name}/{child_name}")
                };

                child_nodes.push(Hdf5Node {
                    name: child_name.clone(),
                    path: child_path,
                    node_type,
                });
            }
        }

        // Collect children from link messages
        for msg in header.messages.iter().filter(|m| m.msg_type == MSG_LINK) {
            if let Ok((link_name, link_addr)) = self.parse_link_message(&msg.data) {
                children.push(link_name.clone());

                let child_header = self.read_object_header(link_addr);
                let node_type = match &child_header {
                    Ok(h) => {
                        if h.messages
                            .iter()
                            .any(|m| m.msg_type == MSG_SYMBOL_TABLE || m.msg_type == MSG_LINK)
                        {
                            Hdf5NodeType::Group
                        } else if h.messages.iter().any(|m| m.msg_type == MSG_DATASPACE) {
                            Hdf5NodeType::Dataset
                        } else {
                            Hdf5NodeType::Group
                        }
                    }
                    Err(_) => Hdf5NodeType::Dataset,
                };

                let child_path = if name == "/" {
                    format!("/{link_name}")
                } else {
                    format!("{name}/{link_name}")
                };

                child_nodes.push(Hdf5Node {
                    name: link_name,
                    path: child_path,
                    node_type,
                });
            }
        }

        // Parse attributes
        for msg in header
            .messages
            .iter()
            .filter(|m| m.msg_type == MSG_ATTRIBUTE)
        {
            if let Ok(attr) = self.parse_attribute_message(&msg.data) {
                attributes.insert(attr.name.clone(), attr);
            }
        }

        let path = if name == "/" {
            "/".to_string()
        } else {
            name.to_string()
        };

        Ok(Hdf5Group {
            name: name.to_string(),
            path,
            children,
            nodes: child_nodes,
            attributes,
        })
    }

    /// Read a dataset at the given object header address
    fn read_dataset_at(&self, address: u64, name: &str, full_path: &str) -> Result<Hdf5Dataset> {
        let header = self.read_object_header(address)?;

        // Parse dataspace (shape)
        let shape = header
            .messages
            .iter()
            .find(|m| m.msg_type == MSG_DATASPACE)
            .map(|m| self.parse_dataspace_message(&m.data))
            .transpose()?
            .unwrap_or_default();

        // Parse datatype
        let dtype = header
            .messages
            .iter()
            .find(|m| m.msg_type == MSG_DATATYPE)
            .map(|m| self.parse_datatype_message(&m.data))
            .transpose()?
            .unwrap_or(Hdf5DataType::Unknown(0, 0));

        // Parse data layout
        let layout = header
            .messages
            .iter()
            .find(|m| m.msg_type == MSG_DATA_LAYOUT)
            .map(|m| self.parse_layout_message(&m.data))
            .transpose()?;

        // Parse filter pipeline if present
        let _has_filters = header
            .messages
            .iter()
            .any(|m| m.msg_type == MSG_FILTER_PIPELINE);

        // Read the data
        let total_elements: usize = if shape.is_empty() {
            1
        } else {
            shape.iter().product()
        };

        let data = if let Some(layout) = layout {
            self.read_data_from_layout(&layout, &dtype, total_elements)?
        } else {
            Hdf5Value::Raw(Vec::new())
        };

        // Parse attributes
        let mut attributes = HashMap::new();
        for msg in header
            .messages
            .iter()
            .filter(|m| m.msg_type == MSG_ATTRIBUTE)
        {
            if let Ok(attr) = self.parse_attribute_message(&msg.data) {
                attributes.insert(attr.name.clone(), attr);
            }
        }

        Ok(Hdf5Dataset {
            name: name.to_string(),
            path: full_path.to_string(),
            dtype,
            shape,
            data,
            attributes,
        })
    }

    // =========================================================================
    // Object header parsing
    // =========================================================================

    /// Read and parse an object header at the given address
    fn read_object_header(&self, address: u64) -> Result<ObjectHeader> {
        let addr = address as usize;
        if addr >= self.data.len() {
            return Err(IoError::FormatError(format!(
                "Object header address {address:#x} beyond file end"
            )));
        }

        let mut cursor = Cursor::new(&self.data[addr..]);

        // Check for v2 object header signature "OHDR"
        let mut peek = [0u8; 4];
        cursor
            .read_exact(&mut peek)
            .map_err(|e| IoError::FormatError(format!("Failed to read object header: {e}")))?;

        if &peek == b"OHDR" {
            self.read_object_header_v2(&mut cursor, addr)
        } else {
            // V1 header: first byte is version
            cursor
                .seek(SeekFrom::Start(0))
                .map_err(|e| IoError::FormatError(format!("Seek error: {e}")))?;
            self.read_object_header_v1(&mut cursor, addr)
        }
    }

    /// Parse a v1 object header
    fn read_object_header_v1(
        &self,
        cursor: &mut Cursor<&[u8]>,
        base_addr: usize,
    ) -> Result<ObjectHeader> {
        let version = cursor
            .read_u8()
            .map_err(|e| IoError::FormatError(format!("Failed to read OH v1 version: {e}")))?;

        if version != 1 {
            return Err(IoError::FormatError(format!(
                "Expected object header version 1, got {version}"
            )));
        }

        // Reserved byte
        let _reserved = cursor
            .read_u8()
            .map_err(|e| IoError::FormatError(format!("Failed to read reserved: {e}")))?;

        // Number of header messages
        let num_messages = cursor
            .read_u16::<LittleEndian>()
            .map_err(|e| IoError::FormatError(format!("Failed to read num messages: {e}")))?;

        // Object reference count
        let _ref_count = cursor
            .read_u32::<LittleEndian>()
            .map_err(|e| IoError::FormatError(format!("Failed to read ref count: {e}")))?;

        // Header data size
        let header_size = cursor
            .read_u32::<LittleEndian>()
            .map_err(|e| IoError::FormatError(format!("Failed to read header size: {e}")))?
            as usize;

        // Align to 8 bytes (padding after header preamble)
        let pos = cursor.position() as usize;
        let aligned = (pos + 7) & !7;
        if aligned > pos {
            cursor
                .seek(SeekFrom::Start(aligned as u64))
                .map_err(|e| IoError::FormatError(format!("Seek error: {e}")))?;
        }

        let mut messages = Vec::new();
        let msg_area_start = cursor.position() as usize;
        let msg_area_end = msg_area_start + header_size;

        let mut count = 0u16;
        while (cursor.position() as usize) < msg_area_end && count < num_messages {
            let msg_type = cursor
                .read_u16::<LittleEndian>()
                .map_err(|e| IoError::FormatError(format!("Failed to read msg type: {e}")))?;
            let msg_size = cursor
                .read_u16::<LittleEndian>()
                .map_err(|e| IoError::FormatError(format!("Failed to read msg size: {e}")))?
                as usize;
            let _msg_flags = cursor
                .read_u8()
                .map_err(|e| IoError::FormatError(format!("Failed to read msg flags: {e}")))?;
            // 3 bytes reserved
            let mut reserved3 = [0u8; 3];
            cursor
                .read_exact(&mut reserved3)
                .map_err(|e| IoError::FormatError(format!("Failed to read reserved: {e}")))?;

            let mut msg_data = vec![0u8; msg_size];
            cursor
                .read_exact(&mut msg_data)
                .map_err(|e| IoError::FormatError(format!("Failed to read msg data: {e}")))?;

            // Handle continuation messages
            if msg_type == MSG_HEADER_CONTINUATION {
                if let Ok(cont_msgs) = self.read_continuation_block(&msg_data) {
                    messages.extend(cont_msgs);
                }
            } else if msg_type != MSG_NIL {
                messages.push(ObjHeaderMsg {
                    msg_type,
                    data: msg_data,
                });
            }

            // Align to 8 bytes
            let pos = cursor.position() as usize;
            let aligned = (pos + 7) & !7;
            if aligned > pos && aligned <= msg_area_end {
                let _ = cursor.seek(SeekFrom::Start(aligned as u64));
            }

            count += 1;
        }

        let _ = base_addr; // used for context only

        Ok(ObjectHeader { messages })
    }

    /// Parse a v2 object header
    fn read_object_header_v2(
        &self,
        cursor: &mut Cursor<&[u8]>,
        _base_addr: usize,
    ) -> Result<ObjectHeader> {
        // Already read "OHDR" signature
        let version = cursor
            .read_u8()
            .map_err(|e| IoError::FormatError(format!("Failed to read OH v2 version: {e}")))?;

        if version != 2 {
            return Err(IoError::FormatError(format!(
                "Expected object header version 2, got {version}"
            )));
        }

        let flags = cursor
            .read_u8()
            .map_err(|e| IoError::FormatError(format!("Failed to read OH v2 flags: {e}")))?;

        // Optional timestamps
        if flags & 0x04 != 0 {
            let _access_time = cursor
                .read_u32::<LittleEndian>()
                .map_err(|e| IoError::FormatError(format!("Failed to read access time: {e}")))?;
            let _modification_time = cursor
                .read_u32::<LittleEndian>()
                .map_err(|e| IoError::FormatError(format!("Failed to read mod time: {e}")))?;
            let _change_time = cursor
                .read_u32::<LittleEndian>()
                .map_err(|e| IoError::FormatError(format!("Failed to read change time: {e}")))?;
            let _birth_time = cursor
                .read_u32::<LittleEndian>()
                .map_err(|e| IoError::FormatError(format!("Failed to read birth time: {e}")))?;
        }

        // Optional maximum # of compact/dense attributes
        if flags & 0x10 != 0 {
            let _max_compact = cursor
                .read_u16::<LittleEndian>()
                .map_err(|e| IoError::FormatError(format!("Failed to read max compact: {e}")))?;
            let _min_dense = cursor
                .read_u16::<LittleEndian>()
                .map_err(|e| IoError::FormatError(format!("Failed to read min dense: {e}")))?;
        }

        // Chunk#0 size (1, 2, 4, or 8 bytes depending on flags bits 0-1)
        let size_field_size = 1 << (flags & 0x03);
        let chunk_size = match size_field_size {
            1 => cursor
                .read_u8()
                .map(u64::from)
                .map_err(|e| IoError::FormatError(format!("Failed to read chunk size: {e}")))?,
            2 => cursor
                .read_u16::<LittleEndian>()
                .map(u64::from)
                .map_err(|e| IoError::FormatError(format!("Failed to read chunk size: {e}")))?,
            4 => cursor
                .read_u32::<LittleEndian>()
                .map(u64::from)
                .map_err(|e| IoError::FormatError(format!("Failed to read chunk size: {e}")))?,
            8 => cursor
                .read_u64::<LittleEndian>()
                .map_err(|e| IoError::FormatError(format!("Failed to read chunk size: {e}")))?,
            _ => {
                return Err(IoError::FormatError(
                    "Invalid chunk size field size".to_string(),
                ))
            }
        };

        let mut messages = Vec::new();
        let chunk_start = cursor.position() as usize;
        let chunk_end = chunk_start + chunk_size as usize;

        while (cursor.position() as usize) + 4 <= chunk_end {
            let msg_type = cursor
                .read_u8()
                .map_err(|e| IoError::FormatError(format!("Failed to read v2 msg type: {e}")))?;

            if msg_type == 0 {
                // Padding / end of messages
                break;
            }

            let msg_size = cursor
                .read_u16::<LittleEndian>()
                .map_err(|e| IoError::FormatError(format!("Failed to read v2 msg size: {e}")))?
                as usize;

            let _msg_flags = cursor
                .read_u8()
                .map_err(|e| IoError::FormatError(format!("Failed to read v2 msg flags: {e}")))?;

            // Optional creation order
            if flags & 0x04 != 0 {
                let _creation_order = cursor.read_u16::<LittleEndian>().map_err(|e| {
                    IoError::FormatError(format!("Failed to read creation order: {e}"))
                })?;
            }

            let mut msg_data = vec![0u8; msg_size];
            cursor
                .read_exact(&mut msg_data)
                .map_err(|e| IoError::FormatError(format!("Failed to read v2 msg data: {e}")))?;

            let msg_type_u16 = msg_type as u16;

            if msg_type_u16 == MSG_HEADER_CONTINUATION {
                if let Ok(cont_msgs) = self.read_continuation_block(&msg_data) {
                    messages.extend(cont_msgs);
                }
            } else {
                messages.push(ObjHeaderMsg {
                    msg_type: msg_type_u16,
                    data: msg_data,
                });
            }
        }

        Ok(ObjectHeader { messages })
    }

    /// Read a header continuation block
    fn read_continuation_block(&self, msg_data: &[u8]) -> Result<Vec<ObjHeaderMsg>> {
        let mut cur = Cursor::new(msg_data);
        let cont_offset = read_offset(&mut cur, self.sb.offset_size)?;
        let cont_length = read_length(&mut cur, self.sb.length_size)?;

        let start = cont_offset as usize;
        let end = start + cont_length as usize;
        if end > self.data.len() {
            return Err(IoError::FormatError(
                "Continuation block beyond file end".to_string(),
            ));
        }

        let mut messages = Vec::new();
        let mut cursor = Cursor::new(&self.data[start..end]);
        let block_len = cont_length as usize;

        while (cursor.position() as usize) + 8 <= block_len {
            let msg_type = cursor
                .read_u16::<LittleEndian>()
                .map_err(|e| IoError::FormatError(format!("Failed to read cont msg type: {e}")))?;
            let msg_size = cursor
                .read_u16::<LittleEndian>()
                .map_err(|e| IoError::FormatError(format!("Failed to read cont msg size: {e}")))?
                as usize;
            let _msg_flags = cursor
                .read_u8()
                .map_err(|e| IoError::FormatError(format!("Failed to read cont msg flags: {e}")))?;
            let mut reserved3 = [0u8; 3];
            cursor
                .read_exact(&mut reserved3)
                .map_err(|e| IoError::FormatError(format!("Failed to read reserved: {e}")))?;

            let mut data = vec![0u8; msg_size];
            cursor
                .read_exact(&mut data)
                .map_err(|e| IoError::FormatError(format!("Failed to read cont msg data: {e}")))?;

            if msg_type != MSG_NIL {
                messages.push(ObjHeaderMsg { msg_type, data });
            }

            // Align to 8 bytes
            let pos = cursor.position() as usize;
            let aligned = (pos + 7) & !7;
            if aligned > pos && aligned <= block_len {
                let _ = cursor.seek(SeekFrom::Start(aligned as u64));
            }
        }

        Ok(messages)
    }

    // =========================================================================
    // Message parsing
    // =========================================================================

    /// Parse a dataspace message to get dimensions
    fn parse_dataspace_message(&self, data: &[u8]) -> Result<Vec<usize>> {
        if data.len() < 4 {
            return Ok(Vec::new());
        }
        let mut cur = Cursor::new(data);
        let version = cur
            .read_u8()
            .map_err(|e| IoError::FormatError(format!("Failed to read dataspace version: {e}")))?;
        let ndims = cur
            .read_u8()
            .map_err(|e| IoError::FormatError(format!("Failed to read ndims: {e}")))?
            as usize;
        let flags = cur
            .read_u8()
            .map_err(|e| IoError::FormatError(format!("Failed to read dataspace flags: {e}")))?;

        if version == 1 {
            // 5 reserved bytes
            let mut reserved = [0u8; 5];
            cur.read_exact(&mut reserved)
                .map_err(|e| IoError::FormatError(format!("Failed to read reserved: {e}")))?;
        } else if version == 2 {
            // 1 reserved byte already consumed (flags)
            // No additional reserved
        }

        // Read dimension sizes
        let mut dims = Vec::with_capacity(ndims);
        for _ in 0..ndims {
            let dim = read_length(&mut cur, self.sb.length_size)?;
            dims.push(dim as usize);
        }

        // Skip max dimension sizes if present
        let has_max = flags & 0x01 != 0;
        if has_max {
            for _ in 0..ndims {
                let _ = read_length(&mut cur, self.sb.length_size);
            }
        }

        Ok(dims)
    }

    /// Parse a datatype message
    fn parse_datatype_message(&self, data: &[u8]) -> Result<Hdf5DataType> {
        if data.len() < 8 {
            return Err(IoError::FormatError(
                "Datatype message too short".to_string(),
            ));
        }

        let class_and_version = data[0];
        let type_class = class_and_version & 0x0F;
        let _version = (class_and_version >> 4) & 0x0F;
        let class_bits = [data[1], data[2], data[3]];
        let type_size = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;

        match type_class {
            // Fixed-point (integer)
            0 => {
                let signed = class_bits[0] & 0x08 != 0;
                match (type_size, signed) {
                    (1, false) => Ok(Hdf5DataType::UInt8),
                    (1, true) => Ok(Hdf5DataType::Int8),
                    (2, false) => Ok(Hdf5DataType::UInt16),
                    (2, true) => Ok(Hdf5DataType::Int16),
                    (4, false) => Ok(Hdf5DataType::UInt32),
                    (4, true) => Ok(Hdf5DataType::Int32),
                    (8, false) => Ok(Hdf5DataType::UInt64),
                    (8, true) => Ok(Hdf5DataType::Int64),
                    _ => Ok(Hdf5DataType::Unknown(type_class, type_size)),
                }
            }
            // Floating-point
            1 => match type_size {
                4 => Ok(Hdf5DataType::Float32),
                8 => Ok(Hdf5DataType::Float64),
                _ => Ok(Hdf5DataType::Unknown(type_class, type_size)),
            },
            // String
            3 => {
                let padding = class_bits[0] & 0x0F;
                // padding 0 = null-terminated, 1 = null-padded, 2 = space-padded
                if type_size == 0 {
                    // Variable-length string
                    Ok(Hdf5DataType::VarString)
                } else {
                    Ok(Hdf5DataType::FixedString(type_size))
                }
            }
            // Compound
            6 => {
                let num_members = u16::from_le_bytes([class_bits[0], class_bits[1]]) as usize;
                let mut cur = Cursor::new(&data[8..]);
                let mut fields = Vec::with_capacity(num_members);

                for _ in 0..num_members {
                    // Read name (null-terminated)
                    let mut name_bytes = Vec::new();
                    loop {
                        let b = cur.read_u8().map_err(|e| {
                            IoError::FormatError(format!("Failed to read compound name: {e}"))
                        })?;
                        if b == 0 {
                            break;
                        }
                        name_bytes.push(b);
                    }
                    let name = String::from_utf8_lossy(&name_bytes).to_string();

                    // Align to 8 bytes
                    let pos = cur.position();
                    let aligned = (pos + 7) & !7;
                    if aligned > pos {
                        let _ = cur.seek(SeekFrom::Start(aligned));
                    }

                    let byte_offset = cur.read_u32::<LittleEndian>().map_err(|e| {
                        IoError::FormatError(format!("Failed to read byte offset: {e}"))
                    })? as usize;

                    // Read nested datatype
                    let remaining = &data[8 + cur.position() as usize..];
                    let field_dt = if remaining.len() >= 8 {
                        self.parse_datatype_message(remaining)
                            .unwrap_or(Hdf5DataType::Unknown(0, 0))
                    } else {
                        Hdf5DataType::Unknown(0, 0)
                    };

                    let dt_size = field_dt.element_size().max(8);
                    let _ = cur.seek(SeekFrom::Current(dt_size as i64));

                    fields.push((name, field_dt, byte_offset));
                }

                Ok(Hdf5DataType::Compound(fields))
            }
            // Opaque
            5 => Ok(Hdf5DataType::Opaque(type_size)),
            // Variable-length
            9 => {
                // Variable-length type: the inner type follows
                // Simplified: just treat as VarString for string sub-type
                Ok(Hdf5DataType::VarString)
            }
            _ => Ok(Hdf5DataType::Unknown(type_class, type_size)),
        }
    }

    /// Parse a data layout message
    fn parse_layout_message(&self, data: &[u8]) -> Result<DataLayout> {
        if data.is_empty() {
            return Err(IoError::FormatError("Empty layout message".to_string()));
        }

        let version = data[0];

        match version {
            3 | 4 => {
                if data.len() < 2 {
                    return Err(IoError::FormatError("Layout message too short".to_string()));
                }
                let layout_class = data[1];

                match layout_class {
                    0 => {
                        // Compact storage
                        if data.len() < 4 {
                            return Ok(DataLayout::Compact { data: Vec::new() });
                        }
                        let compact_size = u16::from_le_bytes([data[2], data[3]]) as usize;
                        let compact_data = if data.len() >= 4 + compact_size {
                            data[4..4 + compact_size].to_vec()
                        } else {
                            Vec::new()
                        };
                        Ok(DataLayout::Compact { data: compact_data })
                    }
                    1 => {
                        // Contiguous storage
                        let mut cur = Cursor::new(&data[2..]);
                        let address = read_offset(&mut cur, self.sb.offset_size)?;
                        let size = read_length(&mut cur, self.sb.length_size)?;
                        Ok(DataLayout::Contiguous { address, size })
                    }
                    2 => {
                        // Chunked storage
                        let mut cur = Cursor::new(&data[2..]);
                        let ndims = cur.read_u8().map_err(|e| {
                            IoError::FormatError(format!("Failed to read chunk ndims: {e}"))
                        })?;
                        let address = read_offset(&mut cur, self.sb.offset_size)?;

                        let mut chunk_dims = Vec::with_capacity(ndims as usize);
                        for _ in 0..ndims {
                            let dim = cur.read_u32::<LittleEndian>().map_err(|e| {
                                IoError::FormatError(format!("Failed to read chunk dim: {e}"))
                            })?;
                            chunk_dims.push(dim);
                        }

                        Ok(DataLayout::Chunked {
                            address,
                            ndims,
                            chunk_dims,
                        })
                    }
                    _ => Err(IoError::UnsupportedFormat(format!(
                        "Unsupported layout class: {layout_class}"
                    ))),
                }
            }
            _ => Err(IoError::UnsupportedFormat(format!(
                "Unsupported layout version: {version}"
            ))),
        }
    }

    /// Parse a symbol table message to get child names and addresses
    fn parse_symbol_table_message(&self, data: &[u8]) -> Result<Vec<(String, u64)>> {
        let mut cur = Cursor::new(data);
        let btree_address = read_offset(&mut cur, self.sb.offset_size)?;
        let heap_address = read_offset(&mut cur, self.sb.offset_size)?;

        // Read the local heap to get names
        let heap_data = self.read_local_heap(heap_address)?;

        // Read B-tree entries
        self.read_btree_group_entries(btree_address, &heap_data)
    }

    /// Read local heap data
    fn read_local_heap(&self, address: u64) -> Result<Vec<u8>> {
        let addr = address as usize;
        if addr + 8 > self.data.len() {
            return Err(IoError::FormatError(
                "Local heap address beyond file".to_string(),
            ));
        }

        let mut cur = Cursor::new(&self.data[addr..]);

        // Signature "HEAP"
        let mut sig = [0u8; 4];
        cur.read_exact(&mut sig)
            .map_err(|e| IoError::FormatError(format!("Failed to read heap signature: {e}")))?;
        if &sig != b"HEAP" {
            return Err(IoError::FormatError(
                "Invalid local heap signature".to_string(),
            ));
        }

        let _version = cur
            .read_u8()
            .map_err(|e| IoError::FormatError(format!("Failed to read heap version: {e}")))?;

        // 3 reserved bytes
        let mut reserved = [0u8; 3];
        cur.read_exact(&mut reserved)
            .map_err(|e| IoError::FormatError(format!("Failed to read heap reserved: {e}")))?;

        // Data segment size
        let data_size = read_length(&mut cur, self.sb.length_size)?;

        // Offset to head of free-list (skip)
        let _free_list_offset = read_length(&mut cur, self.sb.length_size)?;

        // Address of data segment
        let data_address = read_offset(&mut cur, self.sb.offset_size)?;

        let data_start = data_address as usize;
        let data_end = data_start + data_size as usize;
        if data_end > self.data.len() {
            return Err(IoError::FormatError(
                "Local heap data beyond file end".to_string(),
            ));
        }

        Ok(self.data[data_start..data_end].to_vec())
    }

    /// Read group entries from a v1 B-tree
    fn read_btree_group_entries(
        &self,
        address: u64,
        heap_data: &[u8],
    ) -> Result<Vec<(String, u64)>> {
        let undef = undefined_address(self.sb.offset_size);
        if address == undef || address as usize >= self.data.len() {
            return Ok(Vec::new());
        }

        let addr = address as usize;
        let mut cur = Cursor::new(&self.data[addr..]);

        // B-tree signature "TREE"
        let mut sig = [0u8; 4];
        cur.read_exact(&mut sig)
            .map_err(|e| IoError::FormatError(format!("Failed to read B-tree signature: {e}")))?;
        if &sig != b"TREE" {
            return Err(IoError::FormatError("Invalid B-tree signature".to_string()));
        }

        let node_type = cur
            .read_u8()
            .map_err(|e| IoError::FormatError(format!("Failed to read node type: {e}")))?;
        let node_level = cur
            .read_u8()
            .map_err(|e| IoError::FormatError(format!("Failed to read node level: {e}")))?;
        let entries_used = cur
            .read_u16::<LittleEndian>()
            .map_err(|e| IoError::FormatError(format!("Failed to read entries used: {e}")))?
            as usize;

        // Left sibling (skip)
        let _left = read_offset(&mut cur, self.sb.offset_size)?;
        // Right sibling (skip)
        let _right = read_offset(&mut cur, self.sb.offset_size)?;

        let mut entries = Vec::new();

        if node_type == 0 {
            // Group node
            if node_level == 0 {
                // Leaf node: entries are symbol table nodes
                for _ in 0..entries_used {
                    // Key (heap offset)
                    let _key = read_length(&mut cur, self.sb.length_size)?;
                    // Child address (symbol table node)
                    let child_addr = read_offset(&mut cur, self.sb.offset_size)?;

                    if let Ok(stn_entries) = self.read_symbol_table_node(child_addr, heap_data) {
                        entries.extend(stn_entries);
                    }
                }
                // Last key
                let _last_key = read_length(&mut cur, self.sb.length_size);
            } else {
                // Internal node: recurse
                for _ in 0..entries_used {
                    let _key = read_length(&mut cur, self.sb.length_size)?;
                    let child_addr = read_offset(&mut cur, self.sb.offset_size)?;

                    if let Ok(child_entries) = self.read_btree_group_entries(child_addr, heap_data)
                    {
                        entries.extend(child_entries);
                    }
                }
                let _last_key = read_length(&mut cur, self.sb.length_size);
            }
        }

        Ok(entries)
    }

    /// Read a symbol table node to get (name, address) pairs
    fn read_symbol_table_node(&self, address: u64, heap_data: &[u8]) -> Result<Vec<(String, u64)>> {
        let addr = address as usize;
        if addr + 8 > self.data.len() {
            return Err(IoError::FormatError(
                "Symbol table node beyond file".to_string(),
            ));
        }

        let mut cur = Cursor::new(&self.data[addr..]);

        // Signature "SNOD"
        let mut sig = [0u8; 4];
        cur.read_exact(&mut sig)
            .map_err(|e| IoError::FormatError(format!("Failed to read SNOD signature: {e}")))?;
        if &sig != b"SNOD" {
            return Err(IoError::FormatError(
                "Invalid symbol table node signature".to_string(),
            ));
        }

        let _version = cur
            .read_u8()
            .map_err(|e| IoError::FormatError(format!("Failed to read SNOD version: {e}")))?;
        let _reserved = cur
            .read_u8()
            .map_err(|e| IoError::FormatError(format!("Failed to read SNOD reserved: {e}")))?;
        let num_symbols = cur
            .read_u16::<LittleEndian>()
            .map_err(|e| IoError::FormatError(format!("Failed to read SNOD symbols: {e}")))?
            as usize;

        let mut entries = Vec::new();

        for _ in 0..num_symbols {
            let name_offset = read_offset(&mut cur, self.sb.offset_size)?;
            let obj_header_addr = read_offset(&mut cur, self.sb.offset_size)?;
            let _cache_type = cur
                .read_u32::<LittleEndian>()
                .map_err(|e| IoError::FormatError(format!("Failed to read cache type: {e}")))?;
            let _reserved2 = cur
                .read_u32::<LittleEndian>()
                .map_err(|e| IoError::FormatError(format!("Failed to read reserved: {e}")))?;
            // Scratch-pad space (16 bytes)
            let mut scratch = [0u8; 16];
            cur.read_exact(&mut scratch)
                .map_err(|e| IoError::FormatError(format!("Failed to read scratch pad: {e}")))?;

            // Read name from heap
            let name = self.read_string_from_heap(heap_data, name_offset as usize);

            if !name.is_empty() {
                entries.push((name, obj_header_addr));
            }
        }

        Ok(entries)
    }

    /// Read a null-terminated string from heap data
    fn read_string_from_heap(&self, heap_data: &[u8], offset: usize) -> String {
        if offset >= heap_data.len() {
            return String::new();
        }
        let end = heap_data[offset..]
            .iter()
            .position(|&b| b == 0)
            .map(|p| offset + p)
            .unwrap_or(heap_data.len());
        String::from_utf8_lossy(&heap_data[offset..end]).to_string()
    }

    /// Parse a link message (v2 groups)
    fn parse_link_message(&self, data: &[u8]) -> Result<(String, u64)> {
        if data.len() < 4 {
            return Err(IoError::FormatError("Link message too short".to_string()));
        }

        let mut cur = Cursor::new(data);
        let version = cur
            .read_u8()
            .map_err(|e| IoError::FormatError(format!("Failed to read link version: {e}")))?;

        if version != 1 {
            return Err(IoError::FormatError(format!(
                "Unsupported link message version: {version}"
            )));
        }

        let flags = cur
            .read_u8()
            .map_err(|e| IoError::FormatError(format!("Failed to read link flags: {e}")))?;

        let link_type = if flags & 0x08 != 0 {
            cur.read_u8()
                .map_err(|e| IoError::FormatError(format!("Failed to read link type: {e}")))?
        } else {
            0 // hard link
        };

        // Optional creation order
        if flags & 0x04 != 0 {
            let _creation_order = cur
                .read_u64::<LittleEndian>()
                .map_err(|e| IoError::FormatError(format!("Failed to read creation order: {e}")))?;
        }

        // Optional link name character set
        if flags & 0x10 != 0 {
            let _charset = cur
                .read_u8()
                .map_err(|e| IoError::FormatError(format!("Failed to read charset: {e}")))?;
        }

        // Name length (size depends on flags bits 0-1)
        let name_size_field = (flags & 0x03) as usize;
        let name_len = match name_size_field {
            0 => cur
                .read_u8()
                .map(|v| v as usize)
                .map_err(|e| IoError::FormatError(format!("Failed to read name len: {e}")))?,
            1 => cur
                .read_u16::<LittleEndian>()
                .map(|v| v as usize)
                .map_err(|e| IoError::FormatError(format!("Failed to read name len: {e}")))?,
            2 => cur
                .read_u32::<LittleEndian>()
                .map(|v| v as usize)
                .map_err(|e| IoError::FormatError(format!("Failed to read name len: {e}")))?,
            3 => cur
                .read_u64::<LittleEndian>()
                .map(|v| v as usize)
                .map_err(|e| IoError::FormatError(format!("Failed to read name len: {e}")))?,
            _ => return Err(IoError::FormatError("Invalid name size field".to_string())),
        };

        let mut name_bytes = vec![0u8; name_len];
        cur.read_exact(&mut name_bytes)
            .map_err(|e| IoError::FormatError(format!("Failed to read link name: {e}")))?;
        let name = String::from_utf8_lossy(&name_bytes).to_string();

        // Link value
        if link_type == 0 {
            // Hard link: address follows
            let address = read_offset(&mut cur, self.sb.offset_size)?;
            Ok((name, address))
        } else {
            Err(IoError::UnsupportedFormat(format!(
                "Only hard links are supported, got type {link_type}"
            )))
        }
    }

    /// Parse an attribute message
    fn parse_attribute_message(&self, data: &[u8]) -> Result<Hdf5Attribute> {
        if data.len() < 6 {
            return Err(IoError::FormatError(
                "Attribute message too short".to_string(),
            ));
        }

        let mut cur = Cursor::new(data);
        let version = cur
            .read_u8()
            .map_err(|e| IoError::FormatError(format!("Failed to read attr version: {e}")))?;

        let _flags = if version >= 2 {
            cur.read_u8()
                .map_err(|e| IoError::FormatError(format!("Failed to read attr flags: {e}")))?
        } else {
            let _ = cur.read_u8(); // reserved
            0u8
        };

        let name_size = cur
            .read_u16::<LittleEndian>()
            .map_err(|e| IoError::FormatError(format!("Failed to read attr name size: {e}")))?
            as usize;

        let datatype_size = cur
            .read_u16::<LittleEndian>()
            .map_err(|e| IoError::FormatError(format!("Failed to read attr dt size: {e}")))?
            as usize;

        let dataspace_size = cur
            .read_u16::<LittleEndian>()
            .map_err(|e| IoError::FormatError(format!("Failed to read attr ds size: {e}")))?
            as usize;

        // Read name
        let pos = cur.position() as usize;
        if pos + name_size > data.len() {
            return Err(IoError::FormatError(
                "Attribute name beyond message".to_string(),
            ));
        }
        let name_raw = &data[pos..pos + name_size];
        let name = String::from_utf8_lossy(name_raw)
            .trim_end_matches('\0')
            .to_string();
        let pos = pos + name_size;

        // Align to 8 for v1
        let pos = if version == 1 { (pos + 7) & !7 } else { pos };

        // Read datatype
        if pos + datatype_size > data.len() {
            return Err(IoError::FormatError(
                "Attribute datatype beyond message".to_string(),
            ));
        }
        let dt_data = &data[pos..pos + datatype_size];
        let dtype = self
            .parse_datatype_message(dt_data)
            .unwrap_or(Hdf5DataType::Unknown(0, 0));
        let pos = pos + datatype_size;

        // Align to 8 for v1
        let pos = if version == 1 { (pos + 7) & !7 } else { pos };

        // Read dataspace
        if pos + dataspace_size > data.len() {
            return Err(IoError::FormatError(
                "Attribute dataspace beyond message".to_string(),
            ));
        }
        let ds_data = &data[pos..pos + dataspace_size];
        let shape = self.parse_dataspace_message(ds_data).unwrap_or_default();
        let pos = pos + dataspace_size;

        // Remaining data is the attribute value
        let value_data = if pos < data.len() { &data[pos..] } else { &[] };

        let total_elements: usize = if shape.is_empty() {
            1
        } else {
            shape.iter().product()
        };

        let value = self.decode_typed_data(value_data, &dtype, total_elements);

        Ok(Hdf5Attribute { name, dtype, value })
    }

    // =========================================================================
    // Data reading
    // =========================================================================

    /// Read data from a layout
    fn read_data_from_layout(
        &self,
        layout: &DataLayout,
        dtype: &Hdf5DataType,
        total_elements: usize,
    ) -> Result<Hdf5Value> {
        match layout {
            DataLayout::Contiguous { address, size } => {
                let undef = undefined_address(self.sb.offset_size);
                if *address == undef || *size == 0 {
                    return Ok(self.empty_value(dtype));
                }
                let start = *address as usize;
                let end = start + *size as usize;
                if end > self.data.len() {
                    return Err(IoError::FormatError(
                        "Contiguous data beyond file end".to_string(),
                    ));
                }
                let raw = &self.data[start..end];
                Ok(self.decode_typed_data(raw, dtype, total_elements))
            }
            DataLayout::Chunked {
                address,
                ndims: _,
                chunk_dims: _,
            } => {
                // For chunked data, we try reading contiguous data at the B-tree address
                // This is a simplification; full chunk assembly would require B-tree traversal
                let undef = undefined_address(self.sb.offset_size);
                if *address == undef {
                    return Ok(self.empty_value(dtype));
                }

                // Try reading chunk B-tree
                let raw = self.read_chunked_data(*address, dtype, total_elements)?;
                Ok(self.decode_typed_data(&raw, dtype, total_elements))
            }
            DataLayout::Compact { data } => Ok(self.decode_typed_data(data, dtype, total_elements)),
        }
    }

    /// Read chunked data by traversing the chunk B-tree
    fn read_chunked_data(
        &self,
        btree_address: u64,
        dtype: &Hdf5DataType,
        total_elements: usize,
    ) -> Result<Vec<u8>> {
        let elem_size = dtype.element_size();
        let total_bytes = total_elements * elem_size;

        let addr = btree_address as usize;
        if addr + 4 > self.data.len() {
            return Ok(vec![0u8; total_bytes]);
        }

        // Check if it's a B-tree
        if &self.data[addr..addr + 4] == b"TREE" {
            // V1 B-tree for chunked data
            let mut cur = Cursor::new(&self.data[addr..]);
            let mut sig = [0u8; 4];
            cur.read_exact(&mut sig)
                .map_err(|e| IoError::FormatError(format!("Failed to read chunk tree sig: {e}")))?;
            let _node_type = cur.read_u8().map_err(|e| {
                IoError::FormatError(format!("Failed to read chunk tree type: {e}"))
            })?;
            let _node_level = cur.read_u8().map_err(|e| {
                IoError::FormatError(format!("Failed to read chunk tree level: {e}"))
            })?;
            let entries_used = cur
                .read_u16::<LittleEndian>()
                .map_err(|e| IoError::FormatError(format!("Failed to read chunk entries: {e}")))?
                as usize;

            // Skip left/right siblings
            let _left = read_offset(&mut cur, self.sb.offset_size);
            let _right = read_offset(&mut cur, self.sb.offset_size);

            // Collect all chunk data
            let mut result = vec![0u8; total_bytes];
            let mut offset = 0usize;

            for _ in 0..entries_used {
                // Chunk size + filter mask
                let chunk_size = cur
                    .read_u32::<LittleEndian>()
                    .map_err(|e| IoError::FormatError(format!("Failed to read chunk size: {e}")))?
                    as usize;
                let _filter_mask = cur.read_u32::<LittleEndian>().map_err(|e| {
                    IoError::FormatError(format!("Failed to read filter mask: {e}"))
                })?;

                // Chunk offset coordinates (skip for now, just read sequentially)
                // For proper handling, we'd need to map coordinates to output positions

                let chunk_addr = read_offset(&mut cur, self.sb.offset_size)?;
                let chunk_start = chunk_addr as usize;
                let chunk_end = chunk_start + chunk_size;

                if chunk_end <= self.data.len() && offset + chunk_size <= result.len() {
                    result[offset..offset + chunk_size]
                        .copy_from_slice(&self.data[chunk_start..chunk_end]);
                    offset += chunk_size;
                }
            }

            Ok(result)
        } else {
            // Fallback: try reading raw data at the address
            let end = (addr + total_bytes).min(self.data.len());
            if addr < self.data.len() {
                Ok(self.data[addr..end].to_vec())
            } else {
                Ok(vec![0u8; total_bytes])
            }
        }
    }

    /// Decode raw bytes into typed data
    fn decode_typed_data(&self, raw: &[u8], dtype: &Hdf5DataType, count: usize) -> Hdf5Value {
        match dtype {
            Hdf5DataType::Int8 => {
                let values: Vec<i8> = raw.iter().take(count).map(|&b| b as i8).collect();
                Hdf5Value::Int8(values)
            }
            Hdf5DataType::Int16 => {
                let values: Vec<i16> = raw
                    .chunks_exact(2)
                    .take(count)
                    .map(|c| i16::from_le_bytes([c[0], c[1]]))
                    .collect();
                Hdf5Value::Int16(values)
            }
            Hdf5DataType::Int32 => {
                let values: Vec<i32> = raw
                    .chunks_exact(4)
                    .take(count)
                    .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                Hdf5Value::Int32(values)
            }
            Hdf5DataType::Int64 => {
                let values: Vec<i64> = raw
                    .chunks_exact(8)
                    .take(count)
                    .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                    .collect();
                Hdf5Value::Int64(values)
            }
            Hdf5DataType::UInt8 => {
                let values: Vec<u8> = raw.iter().take(count).copied().collect();
                Hdf5Value::UInt8(values)
            }
            Hdf5DataType::UInt16 => {
                let values: Vec<u16> = raw
                    .chunks_exact(2)
                    .take(count)
                    .map(|c| u16::from_le_bytes([c[0], c[1]]))
                    .collect();
                Hdf5Value::UInt16(values)
            }
            Hdf5DataType::UInt32 => {
                let values: Vec<u32> = raw
                    .chunks_exact(4)
                    .take(count)
                    .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                Hdf5Value::UInt32(values)
            }
            Hdf5DataType::UInt64 => {
                let values: Vec<u64> = raw
                    .chunks_exact(8)
                    .take(count)
                    .map(|c| u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                    .collect();
                Hdf5Value::UInt64(values)
            }
            Hdf5DataType::Float32 => {
                let values: Vec<f32> = raw
                    .chunks_exact(4)
                    .take(count)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                Hdf5Value::Float32(values)
            }
            Hdf5DataType::Float64 => {
                let values: Vec<f64> = raw
                    .chunks_exact(8)
                    .take(count)
                    .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                    .collect();
                Hdf5Value::Float64(values)
            }
            Hdf5DataType::FixedString(len) => {
                let values: Vec<String> = raw
                    .chunks(*len)
                    .take(count)
                    .map(|chunk| {
                        let s = String::from_utf8_lossy(chunk);
                        s.trim_end_matches('\0').to_string()
                    })
                    .collect();
                Hdf5Value::Strings(values)
            }
            Hdf5DataType::VarString => {
                // Variable-length strings are stored via global heap
                // Simplified: try to read null-terminated strings
                let mut strings = Vec::new();
                let mut pos = 0;
                for _ in 0..count {
                    if pos >= raw.len() {
                        strings.push(String::new());
                        continue;
                    }
                    let end = raw[pos..]
                        .iter()
                        .position(|&b| b == 0)
                        .map(|p| pos + p)
                        .unwrap_or(raw.len());
                    strings.push(String::from_utf8_lossy(&raw[pos..end]).to_string());
                    pos = end + 1;
                }
                Hdf5Value::Strings(strings)
            }
            _ => Hdf5Value::Raw(raw.to_vec()),
        }
    }

    /// Create an empty value for a given data type
    fn empty_value(&self, dtype: &Hdf5DataType) -> Hdf5Value {
        match dtype {
            Hdf5DataType::Int8 => Hdf5Value::Int8(Vec::new()),
            Hdf5DataType::Int16 => Hdf5Value::Int16(Vec::new()),
            Hdf5DataType::Int32 => Hdf5Value::Int32(Vec::new()),
            Hdf5DataType::Int64 => Hdf5Value::Int64(Vec::new()),
            Hdf5DataType::UInt8 => Hdf5Value::UInt8(Vec::new()),
            Hdf5DataType::UInt16 => Hdf5Value::UInt16(Vec::new()),
            Hdf5DataType::UInt32 => Hdf5Value::UInt32(Vec::new()),
            Hdf5DataType::UInt64 => Hdf5Value::UInt64(Vec::new()),
            Hdf5DataType::Float32 => Hdf5Value::Float32(Vec::new()),
            Hdf5DataType::Float64 => Hdf5Value::Float64(Vec::new()),
            Hdf5DataType::FixedString(_) | Hdf5DataType::VarString => {
                Hdf5Value::Strings(Vec::new())
            }
            _ => Hdf5Value::Raw(Vec::new()),
        }
    }
}
