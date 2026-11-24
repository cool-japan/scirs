//! FITS regression tests against real files, with astropy as the reference.
//!
//! Fixtures in `tests/data/fits/`:
//!
//! - `u16_none.fits`, `u16_rice.fits`: from OrbitalCommons/fits-test-cases
//!   `refimage/` (written by the refimage 1.0.0-pre6 crate, MIT OR Apache-2.0).
//!   A 57 x 23 (NAXIS1 x NAXIS2) `uint16` ramp `(i * 977) mod 65536`, stored as
//!   BITPIX 16 with BZERO = 32768; plain and RICE_1 tile-compressed.
//! - `u16_none.fits.gz`: `gzip -9n u16_none.fits`.
//! - `fpack_rice_u16_theap_gap.fits`: from fits-test-cases `tile-compression/`
//!   (Apache-2.0): a 64 x 64 RICE_1 `uint16` image written by fpack (CFITSIO
//!   4.6.4), with 100 bytes inserted before the compressed table's heap
//!   (THEAP = 612).
//! - `tb.fits`: astropy's test file `astropy/io/fits/tests/data/tb.fits`
//!   (BSD-3-Clause), a 2-row binary table with J, A, E (TSCAL/TZERO) and L
//!   columns.
//! - `cfitsio_keys_and_unsigned.fits`: from fits-test-cases `cfitsio-writer/`,
//!   written by cfitsio 4.3.1 (synthetic, no restrictions): CONTINUE long
//!   strings, doubled quotes, and unsigned columns stored with TZERO.
//! - `layout.fits`, `card36.fits`: written by astropy 8.0.1 for these tests.
//!   `layout.fits` is a BINTABLE with a 3000-byte heap followed by an IMAGE
//!   extension; `card36.fits` is a 5 x 3 BITPIX 16 image with BSCALE = 0.5,
//!   BZERO = 100 and exactly 36 header cards before END.
//! - `theap_gap.fits`: written byte by byte for these tests; a one-row BINTABLE
//!   with a `1PJ` column whose heap starts 100 bytes after the table
//!   (THEAP = 108, PCOUNT = 112). astropy reads the cell as `[7, 8, 9]`.
//!
//! Expected values were read with astropy 8.0.1 (`astropy.io.fits`,
//! `astropy.wcs`). Images are compared in this module's `[x, y]` index order,
//! i.e. `image[[x, y]] == astropy_data[y, x]`.

use scirs2_core::ndarray::Array2;
use scirs2_io::formats::astronomical::{
    CardValue, FitsDataType, FitsFile, FitsHeader, FitsTableReader, FitsWriter, HDUType,
    HeaderCard, VOTableValue, WCSTransform,
};
use std::path::PathBuf;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("data")
        .join("fits")
        .join(name)
}

fn open(name: &str) -> FitsFile {
    FitsFile::open(fixture(name)).unwrap_or_else(|e| panic!("open {name}: {e}"))
}

/// The refimage ramp at FITS pixel (x, y), 0-based, as astropy decodes it.
fn ramp(x: usize, y: usize) -> f64 {
    (((y * 57 + x) * 977) % 65536) as f64
}

fn assert_ramp_f64(image: &Array2<f64>) {
    assert_eq!(image.dim(), (57, 23), "shape is (NAXIS1, NAXIS2)");
    for ((x, y), &v) in image.indexed_iter() {
        assert_eq!(v, ramp(x, y), "pixel x={x} y={y}");
    }
}

#[test]
fn u16_bzero_image_reads_physical_values() {
    let fits = open("u16_none.fits");
    assert_ramp_f64(&fits.read_image::<f64>().expect("read f64"));

    let image: Array2<f32> = fits.read_image().expect("read f32");
    assert_eq!(image[[0, 1]], 55689.0, "astropy data[1, 0]");
    assert_eq!(image[[56, 22]], ramp(56, 22) as f32);
}

#[test]
fn image_dimensions_are_naxis_order() {
    let fits = open("u16_none.fits");
    assert_eq!(fits.image_dimensions(0).expect("dims"), vec![57, 23]);
}

#[test]
fn gzip_file_reads() {
    let fits = open("u16_none.fits.gz");
    assert_ramp_f64(&fits.read_image::<f64>().expect("read gz"));
}

#[test]
fn tile_compressed_image_reads() {
    let fits = open("u16_rice.fits");
    assert_eq!(fits.hdu_count(), 2);
    let hdu = fits.gethdu(1).expect("hdu 1");
    assert_eq!(hdu.hdu_type, HDUType::Image);
    // The header of the image, not of the table holding it.
    assert_eq!(hdu.header.get_i64("BITPIX").expect("BITPIX"), 16);
    assert_eq!(hdu.header.get_i64("NAXIS1").expect("NAXIS1"), 57);
    assert_eq!(hdu.header.get_i64("NAXIS2").expect("NAXIS2"), 23);
    assert_eq!(hdu.header.get_f64("BZERO").expect("BZERO"), 32768.0);
    assert!(!hdu.header.has_keyword("ZIMAGE"));
    assert!(!hdu.header.has_keyword("TFORM1"));
    assert_eq!(fits.image_dimensions(1).expect("dims"), vec![57, 23]);
    assert_ramp_f64(&fits.readhdu_image::<f64>(1).expect("read rice"));
}

#[test]
fn tile_compressed_heap_after_a_theap_gap() {
    let fits = open("fpack_rice_u16_theap_gap.fits");
    let image: Array2<f64> = fits.readhdu_image(1).expect("read");
    // astropy: data.sum(), data[5, 17], data[17, 5], data[63, 63]
    assert_eq!(image.sum(), 50532870.0);
    assert_eq!(image[[17, 5]], 1015.0);
    assert_eq!(image[[5, 17]], 1557.0);
    assert_eq!(image[[63, 63]], 1246.0);
    assert_eq!(
        image.slice(scirs2_core::ndarray::s![0..4, 0]).to_vec(),
        vec![999.0, 1029.0, 1007.0, 977.0]
    );
}

#[test]
fn bscale_bzero_and_36_card_header() {
    let fits = open("card36.fits");
    let image: Array2<f64> = fits.read_image().expect("read");
    // astropy: data[y, x] = 0.5 * (3 * (5 * y + x) - 7) + 100
    assert_eq!(image.dim(), (5, 3));
    for ((x, y), &v) in image.indexed_iter() {
        let expected = 0.5 * (3.0 * (5 * y + x) as f64 - 7.0) + 100.0;
        assert_eq!(v, expected, "pixel x={x} y={y}");
    }
}

#[test]
fn quoted_values_keep_slashes_and_quotes() {
    let fits = open("card36.fits");
    assert_eq!(
        fits.primaryheader()
            .get_string("OBSERVER")
            .expect("OBSERVER"),
        "O'Brien / night 2"
    );
    let fits = open("tb.fits");
    assert_eq!(
        fits.primaryheader().get_string("ORIGIN").expect("ORIGIN"),
        "STScI-STSDAS/TABLES"
    );
}

#[test]
fn continue_long_strings_are_joined() {
    let fits = open("cfitsio_keys_and_unsigned.fits");
    let header = fits.primaryheader();
    let alphabet = "abcdefghijklmnopqrstuvwxyz0123456789 ";
    let str200: String = format!("200:{}", alphabet.repeat(6))
        .chars()
        .take(200)
        .collect();
    assert_eq!(header.get_string("STR200").expect("STR200"), str200);
    assert_eq!(
        header.get_string("QUOTES").expect("QUOTES"),
        "it's a value with 'quotes' placed so that one doubled quote falls \
         near the split point of the card's text"
    );
}

#[test]
fn heap_after_a_theap_gap() {
    let fits = open("theap_gap.fits");
    let table = FitsTableReader::new(fits.gethdu(1).expect("hdu 1").clone()).expect("table");
    assert_eq!(
        table.read_column("V").expect("V"),
        vec![VOTableValue::Array(
            [7, 8, 9].into_iter().map(VOTableValue::Integer).collect()
        )]
    );
}

#[test]
fn hierarch_cards_keep_their_values() {
    let fits = open("cfitsio_keys_and_unsigned.fits");
    let header = fits.primaryheader();
    assert_eq!(header.get_i64("STARTTIME").expect("STARTTIME"), 1458492896);
    assert_eq!(header.get_i64("ESO DET NDIT").expect("NDIT"), 42);
    assert_eq!(
        header.get_string("ESO OBS NAME").expect("NAME"),
        "short hierarch string"
    );
    let alphabet = "abcdefghijklmnopqrstuvwxyz0123456789 ";
    let str200: String = format!("200:{}", alphabet.repeat(6))
        .chars()
        .take(200)
        .collect();
    assert_eq!(
        header.get_string("ESO OBS LONGSTR").expect("LONGSTR"),
        str200
    );
    assert!(!header.has_keyword("CONTINUE"));
}

#[test]
fn hdu_after_a_heap_is_found() {
    let fits = open("layout.fits");
    assert_eq!(fits.hdu_count(), 3);
    let table = FitsTableReader::new(fits.gethdu(1).expect("hdu 1").clone()).expect("table");
    assert_eq!(
        table.read_column("SAMPLES").expect("SAMPLES"),
        vec![VOTableValue::Array(
            (0..750).map(VOTableValue::Integer).collect()
        )]
    );
    let hdu = fits.gethdu(2).expect("hdu 2");
    assert_eq!(hdu.hdu_type, HDUType::Image);
    assert_eq!(hdu.header.get_string("EXTNAME").expect("EXTNAME"), "AFTER");
    let image: Array2<f64> = fits.readhdu_image(2).expect("read");
    assert_eq!(image.dim(), (4, 3));
    for ((x, y), &v) in image.indexed_iter() {
        assert_eq!(v, ((4 * y + x) * 7) as f64 - 20.0, "pixel x={x} y={y}");
    }
}

#[test]
fn table_columns_come_from_the_file() {
    let fits = open("tb.fits");
    let reader = FitsTableReader::new(fits.gethdu(1).expect("hdu 1").clone()).expect("table");
    assert_eq!(reader.get_row_count().expect("rows"), 2);
    assert_eq!(
        reader.get_columnnames().expect("names"),
        vec!["c1", "c2", "c3", "c4"]
    );
    assert_eq!(
        reader.read_column("c1").expect("c1"),
        vec![VOTableValue::Integer(1), VOTableValue::Integer(2)]
    );
    assert_eq!(
        reader.read_column("c2").expect("c2"),
        vec![
            VOTableValue::String("abc".into()),
            VOTableValue::String("xy".into())
        ]
    );
    // TSCAL3 = 3, TZERO3 = 0.4 applied to the stored f32, as astropy does.
    assert_eq!(
        reader.read_column("c3").expect("c3"),
        vec![
            VOTableValue::Double(3.7000000715255736),
            VOTableValue::Double(6.699999713897705)
        ]
    );
    assert_eq!(
        reader.read_column("c4").expect("c4"),
        vec![VOTableValue::Boolean(false), VOTableValue::Boolean(true)]
    );
    assert!(reader.read_column("FLUX").is_err());
}

#[test]
fn table_tzero_unsigned_columns() {
    let fits = open("cfitsio_keys_and_unsigned.fits");
    let reader = FitsTableReader::new(fits.gethdu(1).expect("hdu 1").clone()).expect("table");
    let ints = |v: &[i64]| {
        v.iter()
            .map(|&i| VOTableValue::Integer(i))
            .collect::<Vec<_>>()
    };
    assert_eq!(
        reader.read_column("U16").expect("U16"),
        ints(&[0, 1, 32767, 32768, 65535])
    );
    assert_eq!(
        reader.read_column("U32").expect("U32"),
        ints(&[0, 1, 2147483647, 2147483648, 4294967295])
    );
    assert_eq!(
        reader.read_column("I8").expect("I8"),
        ints(&[-128, -1, 0, 1, 127])
    );
    let vec_col = reader.read_column("U16VEC").expect("U16VEC");
    assert_eq!(
        vec_col[2],
        VOTableValue::Array(ints(&[65534, 32767, 4])),
        "row 3"
    );
}

fn header(cards: &[(&str, CardValue)]) -> FitsHeader {
    let mut header = FitsHeader::new();
    for (keyword, value) in cards {
        header.add_card(HeaderCard {
            keyword: keyword.to_string(),
            value: value.clone(),
            comment: None,
        });
    }
    header
}

fn check_wcs(wcs: &WCSTransform, cases: &[([f64; 2], [f64; 2])]) {
    for (pix, world) in cases {
        let (ra, dec) = wcs.pixel_to_world(pix[0], pix[1]);
        assert!(
            (ra - world[0]).abs() < 1e-9 && (dec - world[1]).abs() < 1e-9,
            "pixel {pix:?}: got ({ra}, {dec}), astropy {world:?}"
        );
        let (px, py) = wcs.world_to_pixel(world[0], world[1]);
        assert!(
            (px - pix[0]).abs() < 1e-8 && (py - pix[1]).abs() < 1e-8,
            "world {world:?}: got ({px}, {py}), astropy {pix:?}"
        );
    }
}

/// HDU 2 of astropy's `test0.fits` (HST WFPC2): RA---TAN with a rotated CD.
#[test]
fn wcs_tan_cd_matrix_hst() {
    use CardValue::{Float, String as S};
    let wcs = WCSTransform::from_fitsheader(&header(&[
        ("CRVAL1", Float(215.584896839)),
        ("CRVAL2", Float(-12.7488294839)),
        ("CRPIX1", Float(212.0)),
        ("CRPIX2", Float(207.25)),
        ("CD1_1", Float(2.19546E-05)),
        ("CD1_2", Float(-5.0801E-05)),
        ("CD2_1", Float(-5.07551E-05)),
        ("CD2_2", Float(-2.19745E-05)),
        ("CTYPE1", S("RA---TAN".into())),
        ("CTYPE2", S("DEC--TAN".into())),
    ]))
    .expect("wcs");
    check_wcs(
        &wcs,
        &[
            ([1.0, 1.0], [215.59088951120967, -12.73358785015611]),
            ([40.0, 40.0], [215.5897361891466, -12.736424327820327]),
            ([20.5, 7.25], [215.5910029357816, -12.734714912576617]),
            ([-300.0, 450.0], [215.56073028248693, -12.728176088159564]),
            ([1500.0, -2000.0], [215.72886315572381, -12.765659858871578]),
        ],
    );
}

/// A wide field (tens of degrees from the reference point) where the
/// gnomonic projection departs strongly from a linear mapping.
#[test]
fn wcs_tan_cd_matrix_wide_field() {
    use CardValue::{Float, String as S};
    let wcs = WCSTransform::from_fitsheader(&header(&[
        ("CTYPE1", S("RA---TAN".into())),
        ("CTYPE2", S("DEC--TAN".into())),
        ("CRVAL1", Float(150.0)),
        ("CRVAL2", Float(60.0)),
        ("CRPIX1", Float(512.5)),
        ("CRPIX2", Float(384.25)),
        ("CD1_1", Float(-0.01)),
        ("CD1_2", Float(0.002)),
        ("CD2_1", Float(0.0015)),
        ("CD2_2", Float(0.01)),
    ]))
    .expect("wcs");
    check_wcs(
        &wcs,
        &[
            ([512.5, 384.25], [150.0, 60.0]),
            ([1.0, 1.0], [157.59058568517975, 55.174051648622275]),
            ([1024.0, 768.0], [140.00165351971228, 64.2534332555543]),
            ([2000.0, -900.0], [125.24978427762557, 46.76486493507083]),
            ([-1500.0, 1600.0], [197.41137729398514, 60.51671435885501]),
        ],
    );
}

/// PCi_j with CDELTi, near the south pole so RA wraps through 0.
#[test]
fn wcs_tan_pc_cdelt() {
    use CardValue::{Float, String as S};
    let wcs = WCSTransform::from_fitsheader(&header(&[
        ("CTYPE1", S("RA---TAN".into())),
        ("CTYPE2", S("DEC--TAN".into())),
        ("CRVAL1", Float(10.0)),
        ("CRVAL2", Float(-75.0)),
        ("CRPIX1", Float(100.0)),
        ("CRPIX2", Float(50.0)),
        ("CDELT1", Float(-0.02)),
        ("CDELT2", Float(0.03)),
        ("PC1_1", Float(0.8)),
        ("PC1_2", Float(-0.6)),
        ("PC2_1", Float(0.6)),
        ("PC2_2", Float(0.8)),
    ]))
    .expect("wcs");
    check_wcs(
        &wcs,
        &[
            ([1.0, 1.0], [14.75570825027522, -77.91498743784457]),
            ([400.0, -300.0], [332.9724477056699, -75.0877327875104]),
            ([-250.0, 700.0], [39.36790347282512, -62.698957608004704]),
        ],
    );
}

#[test]
fn writer_round_trips_non_square_image() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("written.fits");
    // 5 x 3 in this module's [x, y] order: NAXIS1 = 5, NAXIS2 = 3.
    let data = Array2::from_shape_fn((5, 3), |(x, y)| (10 * y + x) as f32 + 0.25);
    let mut writer = FitsWriter::create(&path).expect("create");
    writer
        .write_image_2d(&data, FitsDataType::Float32)
        .expect("write");
    writer.close().expect("close");

    let fits = FitsFile::open(&path).expect("open written");
    assert_eq!(fits.image_dimensions(0).expect("dims"), vec![5, 3]);
    assert_eq!(fits.read_image::<f32>().expect("read"), data);
}
