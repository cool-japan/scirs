//! Discrete Wavelet Transform (DWT) Implementation
//!
//! Provides 1D, 2D, and N-D discrete wavelet transforms with multiple wavelet families.
//! Implements efficient decomposition and reconstruction with proper boundary handling.

use crate::error::{Result, TransformError};
use rayon::prelude::*;
use scirs2_core::ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, Axis};
/// Wavelet types supported by the DWT implementation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WaveletType {
    /// Haar wavelet (Daubechies-1)
    Haar,
    /// Daubechies wavelets (N = 2, 4, 6, 8, 10, 12, 14, 16, 18, 20)
    Daubechies(usize),
    /// Symlet wavelets
    Symlet(usize),
    /// Coiflet wavelets
    Coiflet(usize),
    /// Biorthogonal wavelets
    Biorthogonal(usize, usize),
}
/// Boundary extension modes for DWT
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BoundaryMode {
    /// Zero padding
    Zero,
    /// Constant padding (edge values)
    Constant,
    /// Symmetric padding
    Symmetric,
    /// Periodic padding
    Periodic,
    /// Reflect padding
    Reflect,
}
/// Wavelet filter coefficients
#[derive(Debug, Clone)]
pub struct WaveletFilters {
    /// Low-pass decomposition filter
    pub dec_lo: Vec<f64>,
    /// High-pass decomposition filter
    pub dec_hi: Vec<f64>,
    /// Low-pass reconstruction filter
    pub rec_lo: Vec<f64>,
    /// High-pass reconstruction filter
    pub rec_hi: Vec<f64>,
}
impl WaveletFilters {
    /// Get filter coefficients for a specific wavelet type
    pub fn from_wavelet(wavelet: WaveletType) -> Result<Self> {
        match wavelet {
            WaveletType::Haar => Self::haar(),
            WaveletType::Daubechies(n) => Self::daubechies(n),
            WaveletType::Symlet(n) => Self::symlet(n),
            WaveletType::Coiflet(n) => Self::coiflet(n),
            WaveletType::Biorthogonal(p, q) => Self::biorthogonal(p, q),
        }
    }
    /// Haar wavelet filters
    fn haar() -> Result<Self> {
        let norm = 1.0 / 2.0_f64.sqrt();
        Ok(WaveletFilters {
            dec_lo: vec![norm, norm],
            dec_hi: vec![norm, -norm],
            rec_lo: vec![norm, norm],
            rec_hi: vec![-norm, norm],
        })
    }
    /// Daubechies wavelet filters
    fn daubechies(n: usize) -> Result<Self> {
        match n {
            2 => {
                let sqrt3 = 3.0_f64.sqrt();
                let denom = 4.0 * 2.0_f64.sqrt();
                let dec_lo = vec![
                    (1.0 + sqrt3) / denom,
                    (3.0 + sqrt3) / denom,
                    (3.0 - sqrt3) / denom,
                    (1.0 - sqrt3) / denom,
                ];
                let mut dec_hi = Vec::with_capacity(dec_lo.len());
                for (i, &val) in dec_lo.iter().enumerate().rev() {
                    dec_hi.push(if i % 2 == 0 { val } else { -val });
                }
                let mut rec_lo = dec_lo.clone();
                rec_lo.reverse();
                let mut rec_hi = dec_hi.clone();
                rec_hi.reverse();
                Ok(WaveletFilters {
                    dec_lo,
                    dec_hi,
                    rec_lo,
                    rec_hi,
                })
            }
            4 => {
                let dec_lo = vec![
                    -0.010597401784997,
                    0.032883011666983,
                    0.030841381835987,
                    -0.187034811718881,
                    -0.027983769416984,
                    0.630880767929590,
                    0.714846570552542,
                    0.230377813308855,
                ];
                let mut dec_hi = Vec::with_capacity(dec_lo.len());
                for (i, &val) in dec_lo.iter().enumerate().rev() {
                    dec_hi.push(if i % 2 == 0 { val } else { -val });
                }
                let mut rec_lo = dec_lo.clone();
                rec_lo.reverse();
                let mut rec_hi = dec_hi.clone();
                rec_hi.reverse();
                Ok(WaveletFilters {
                    dec_lo,
                    dec_hi,
                    rec_lo,
                    rec_hi,
                })
            }
            1 => Self::haar(),
            3 => {
                let dec_lo: Vec<f64> = vec![
                    0.035226291882100656,
                    -0.08544127388224149,
                    -0.13501102001039084,
                    0.4598775021193313,
                    0.8068915093133388,
                    0.3326705529509569,
                ];
                let mut dec_hi: Vec<f64> = Vec::with_capacity(dec_lo.len());
                for (i, &val) in dec_lo.iter().enumerate().rev() {
                    dec_hi.push(if i % 2 == 0 { val } else { -val });
                }
                let mut rec_lo = dec_lo.clone();
                rec_lo.reverse();
                let mut rec_hi = dec_hi.clone();
                rec_hi.reverse();
                Ok(WaveletFilters {
                    dec_lo,
                    dec_hi,
                    rec_lo,
                    rec_hi,
                })
            }
            5 => {
                let dec_lo: Vec<f64> = vec![
                    0.003335725285001549,
                    -0.012580751999015526,
                    -0.006241490213011705,
                    0.07757149384006515,
                    -0.03224486958502952,
                    -0.24229488706619015,
                    0.13842814590110342,
                    0.7243085284385744,
                    0.6038292697974898,
                    0.16010239797412501,
                ];
                let mut dec_hi: Vec<f64> = Vec::with_capacity(dec_lo.len());
                for (i, &val) in dec_lo.iter().enumerate().rev() {
                    dec_hi.push(if i % 2 == 0 { val } else { -val });
                }
                let mut rec_lo = dec_lo.clone();
                rec_lo.reverse();
                let mut rec_hi = dec_hi.clone();
                rec_hi.reverse();
                Ok(WaveletFilters {
                    dec_lo,
                    dec_hi,
                    rec_lo,
                    rec_hi,
                })
            }
            6 => {
                let dec_lo: Vec<f64> = vec![
                    -0.0010773010853084796,
                    0.004777257510945511,
                    0.0005538422011614961,
                    -0.03158203931748603,
                    0.027522865530305727,
                    0.09750160558732304,
                    -0.12976686756726194,
                    -0.22626469396543983,
                    0.31525035170919763,
                    0.7511339080210954,
                    0.49462389039845306,
                    0.11154074335010947,
                ];
                let mut dec_hi: Vec<f64> = Vec::with_capacity(dec_lo.len());
                for (i, &val) in dec_lo.iter().enumerate().rev() {
                    dec_hi.push(if i % 2 == 0 { val } else { -val });
                }
                let mut rec_lo = dec_lo.clone();
                rec_lo.reverse();
                let mut rec_hi = dec_hi.clone();
                rec_hi.reverse();
                Ok(WaveletFilters {
                    dec_lo,
                    dec_hi,
                    rec_lo,
                    rec_hi,
                })
            }
            7 => {
                let dec_lo: Vec<f64> = vec![
                    0.00035371379997452024,
                    -0.0018016407040474908,
                    0.0004295779729213665,
                    0.01255099855609984,
                    -0.01657454163066688,
                    -0.03802993693501441,
                    0.08061260915108308,
                    0.07130921926683026,
                    -0.22403618499387498,
                    -0.14390600392856498,
                    0.4697822874051931,
                    0.7291320908462351,
                    0.3965393194819173,
                    0.07785205408500918,
                ];
                let mut dec_hi: Vec<f64> = Vec::with_capacity(dec_lo.len());
                for (i, &val) in dec_lo.iter().enumerate().rev() {
                    dec_hi.push(if i % 2 == 0 { val } else { -val });
                }
                let mut rec_lo = dec_lo.clone();
                rec_lo.reverse();
                let mut rec_hi = dec_hi.clone();
                rec_hi.reverse();
                Ok(WaveletFilters {
                    dec_lo,
                    dec_hi,
                    rec_lo,
                    rec_hi,
                })
            }
            8 => {
                let dec_lo: Vec<f64> = vec![
                    -0.00011747678412476953,
                    0.0006754494064505693,
                    -0.00039174037337694705,
                    -0.004870352993451574,
                    0.008746094047405777,
                    0.013981027917398282,
                    -0.044088253930794755,
                    -0.017369301001807547,
                    0.12874742662047847,
                    0.0004724845739132828,
                    -0.2840155429615469,
                    -0.015829105256349306,
                    0.5853546836542067,
                    0.6756307362972898,
                    0.31287159091429995,
                    0.05441584224310401,
                ];
                let mut dec_hi: Vec<f64> = Vec::with_capacity(dec_lo.len());
                for (i, &val) in dec_lo.iter().enumerate().rev() {
                    dec_hi.push(if i % 2 == 0 { val } else { -val });
                }
                let mut rec_lo = dec_lo.clone();
                rec_lo.reverse();
                let mut rec_hi = dec_hi.clone();
                rec_hi.reverse();
                Ok(WaveletFilters {
                    dec_lo,
                    dec_hi,
                    rec_lo,
                    rec_hi,
                })
            }
            9 => {
                let dec_lo: Vec<f64> = vec![
                    3.93473203162716e-05,
                    -0.0002519631889427101,
                    0.00023038576352319597,
                    0.0018476468830562265,
                    -0.00428150368246343,
                    -0.004723204757751397,
                    0.022361662123679096,
                    0.00025094711483145197,
                    -0.06763282906132997,
                    0.03072568147933338,
                    0.14854074933810638,
                    -0.09684078322297646,
                    -0.2932737832791749,
                    0.13319738582500756,
                    0.6572880780513005,
                    0.6048231236901112,
                    0.24383467461259034,
                    0.038077947363878345,
                ];
                let mut dec_hi: Vec<f64> = Vec::with_capacity(dec_lo.len());
                for (i, &val) in dec_lo.iter().enumerate().rev() {
                    dec_hi.push(if i % 2 == 0 { val } else { -val });
                }
                let mut rec_lo = dec_lo.clone();
                rec_lo.reverse();
                let mut rec_hi = dec_hi.clone();
                rec_hi.reverse();
                Ok(WaveletFilters {
                    dec_lo,
                    dec_hi,
                    rec_lo,
                    rec_hi,
                })
            }
            10 => {
                let dec_lo: Vec<f64> = vec![
                    -1.3264202894521244e-05,
                    9.358867032006959e-05,
                    -0.00011646685512928545,
                    -0.0006858566949597116,
                    0.001992405295185056,
                    0.001395351747052901,
                    -0.010733175483330575,
                    0.0036065535669561697,
                    0.033212674059341,
                    -0.029457536821875813,
                    -0.07139414716639708,
                    0.09305736460357235,
                    0.12736934033579325,
                    -0.19594627437737705,
                    -0.24984642432731538,
                    0.2811723436605775,
                    0.6884590394536035,
                    0.5272011889317256,
                    0.1881768000776915,
                    0.026670057900555554,
                ];
                let mut dec_hi: Vec<f64> = Vec::with_capacity(dec_lo.len());
                for (i, &val) in dec_lo.iter().enumerate().rev() {
                    dec_hi.push(if i % 2 == 0 { val } else { -val });
                }
                let mut rec_lo = dec_lo.clone();
                rec_lo.reverse();
                let mut rec_hi = dec_hi.clone();
                rec_hi.reverse();
                Ok(WaveletFilters {
                    dec_lo,
                    dec_hi,
                    rec_lo,
                    rec_hi,
                })
            }
            _ => Err(TransformError::InvalidInput(format!(
                "Daubechies-{} not implemented (supported: 1-10)",
                n
            ))),
        }
    }
    /// Symlet wavelet filters
    ///
    /// Symlets are the "least asymmetric" variant of the Daubechies wavelets
    /// (same number of vanishing moments and filter length as `db{n}`, but a
    /// different, more nearly-symmetric phase choice among the valid
    /// spectral factorizations). Coefficients below are the standard
    /// tabulated `sym2`-`sym8` values from PyWavelets
    /// (`pywt/_extensions/c/wavelets_coeffs.template.h`, arrays
    /// `sym2_`..`sym8_`), the canonical, widely cross-validated reference
    /// implementation (matches MATLAB's Wavelet Toolbox to machine
    /// precision). They are combined into the four decomposition/
    /// reconstruction filters using the exact same orthogonal-QMF
    /// construction already used by [`Self::daubechies`] in this file.
    fn symlet(n: usize) -> Result<Self> {
        let dec_lo: Vec<f64> = match n {
            2 => {
                vec![
                    0.48296291314469025,
                    0.83651630373746899,
                    0.22414386804185735,
                    -0.12940952255092145,
                ]
            }
            3 => {
                vec![
                    0.33267055295095688,
                    0.80689150931333875,
                    0.45987750211933132,
                    -0.13501102001039084,
                    -0.085441273882241486,
                    0.035226291882100656,
                ]
            }
            4 => {
                vec![
                    0.032223100604042702,
                    -0.012603967262037833,
                    -0.099219543576847216,
                    0.29785779560527736,
                    0.80373875180591614,
                    0.49761866763201545,
                    -0.02963552764599851,
                    -0.075765714789273325,
                ]
            }
            5 => {
                vec![
                    0.019538882735286728,
                    -0.021101834024758855,
                    -0.17532808990845047,
                    0.016602105764522319,
                    0.63397896345821192,
                    0.72340769040242059,
                    0.1993975339773936,
                    -0.039134249302383094,
                    0.029519490925774643,
                    0.027333068345077982,
                ]
            }
            6 => {
                vec![
                    -0.007800708325034148,
                    0.0017677118642428036,
                    0.044724901770665779,
                    -0.021060292512300564,
                    -0.072637522786462516,
                    0.3379294217276218,
                    0.787641141030194,
                    0.49105594192674662,
                    -0.048311742585632998,
                    -0.11799011114819057,
                    0.0034907120842174702,
                    0.015404109327027373,
                ]
            }
            7 => {
                vec![
                    0.010268176708511255,
                    0.0040102448715336634,
                    -0.10780823770381774,
                    -0.14004724044296152,
                    0.28862963175151463,
                    0.76776431700316405,
                    0.5361019170917628,
                    0.017441255086855827,
                    -0.049552834937127255,
                    0.067892693501372697,
                    0.03051551316596357,
                    -0.01263630340325193,
                    -0.0010473848886829163,
                    0.0026818145682578781,
                ]
            }
            8 => {
                vec![
                    0.0018899503327594609,
                    -0.0003029205147213668,
                    -0.014952258337048231,
                    0.0038087520138906151,
                    0.049137179673607506,
                    -0.027219029917056003,
                    -0.051945838107709037,
                    0.3644418948353314,
                    0.77718575170052351,
                    0.48135965125837221,
                    -0.061273359067658524,
                    -0.14329423835080971,
                    0.0076074873249176054,
                    0.031695087811492981,
                    -0.00054213233179114812,
                    -0.0033824159510061256,
                ]
            }
            _ => {
                return Err(TransformError::InvalidInput(format!(
                    "Symlet-{} not implemented (supported: 2-8)",
                    n
                )));
            }
        };
        let mut dec_hi = Vec::with_capacity(dec_lo.len());
        for (i, &val) in dec_lo.iter().enumerate().rev() {
            dec_hi.push(if i % 2 == 0 { val } else { -val });
        }
        let mut rec_lo = dec_lo.clone();
        rec_lo.reverse();
        let mut rec_hi = dec_hi.clone();
        rec_hi.reverse();
        Ok(WaveletFilters {
            dec_lo,
            dec_hi,
            rec_lo,
            rec_hi,
        })
    }
    /// Coiflet wavelet filters
    fn coiflet(n: usize) -> Result<Self> {
        match n {
            1 => {
                let dec_lo: Vec<f64> = vec![
                    -0.015655728135791993,
                    -0.07273261951252645,
                    0.3848648468648578,
                    0.8525720202116004,
                    0.3378976624574818,
                    -0.07273261951252645,
                ];
                let mut dec_hi: Vec<f64> = Vec::with_capacity(dec_lo.len());
                for (i, &val) in dec_lo.iter().enumerate().rev() {
                    dec_hi.push(if i % 2 == 0 { val } else { -val });
                }
                let mut rec_lo = dec_lo.clone();
                rec_lo.reverse();
                let mut rec_hi = dec_hi.clone();
                rec_hi.reverse();
                Ok(WaveletFilters {
                    dec_lo,
                    dec_hi,
                    rec_lo,
                    rec_hi,
                })
            }
            2 => {
                let dec_lo: Vec<f64> = vec![
                    -0.000720549445520347,
                    -0.0018232088709110323,
                    0.005611434819368834,
                    0.02368017194684777,
                    -0.05943441864643109,
                    -0.07648859907828076,
                    0.4170051844232391,
                    0.8127236354494135,
                    0.3861100668227629,
                    -0.0673725547237256,
                    -0.04146493678687178,
                    0.01638733646320364,
                ];
                let mut dec_hi: Vec<f64> = Vec::with_capacity(dec_lo.len());
                for (i, &val) in dec_lo.iter().enumerate().rev() {
                    dec_hi.push(if i % 2 == 0 { val } else { -val });
                }
                let mut rec_lo = dec_lo.clone();
                rec_lo.reverse();
                let mut rec_hi = dec_hi.clone();
                rec_hi.reverse();
                Ok(WaveletFilters {
                    dec_lo,
                    dec_hi,
                    rec_lo,
                    rec_hi,
                })
            }
            3 => {
                let dec_lo: Vec<f64> = vec![
                    -3.459977319727278e-05,
                    -7.0983302506379e-05,
                    0.0004662169598204029,
                    0.0011175187708306303,
                    -0.0025745176881367972,
                    -0.009007976136730624,
                    0.015880544863669452,
                    0.03455502757329774,
                    -0.08230192710629983,
                    -0.07179982161915484,
                    0.42848347637737,
                    0.7937772226260872,
                    0.40517690240911824,
                    -0.06112339000297255,
                    -0.06577191128146936,
                    0.023452696142077168,
                    0.007782596425672746,
                    -0.003793512864380802,
                ];
                let mut dec_hi: Vec<f64> = Vec::with_capacity(dec_lo.len());
                for (i, &val) in dec_lo.iter().enumerate().rev() {
                    dec_hi.push(if i % 2 == 0 { val } else { -val });
                }
                let mut rec_lo = dec_lo.clone();
                rec_lo.reverse();
                let mut rec_hi = dec_hi.clone();
                rec_hi.reverse();
                Ok(WaveletFilters {
                    dec_lo,
                    dec_hi,
                    rec_lo,
                    rec_hi,
                })
            }
            4 => {
                let dec_lo: Vec<f64> = vec![
                    -1.7849909144933469e-06,
                    -3.259647940030751e-06,
                    3.1229861599195265e-05,
                    6.233885431278719e-05,
                    -0.0002599743371222568,
                    -0.0005890202246332165,
                    0.0012665610789256603,
                    0.0037514346971460866,
                    -0.0056582838001308835,
                    -0.015211728187697211,
                    0.02508225333794961,
                    0.03933442260558915,
                    -0.09622042453595264,
                    -0.06662747236681717,
                    0.43438603311435653,
                    0.7822389344242826,
                    0.41530842700068227,
                    -0.05607731960356926,
                    -0.08126671024919373,
                    0.02668230466960483,
                    0.01606894713157503,
                    -0.007346167936268051,
                    -0.001629492425226786,
                    0.000892313902537003,
                ];
                let mut dec_hi: Vec<f64> = Vec::with_capacity(dec_lo.len());
                for (i, &val) in dec_lo.iter().enumerate().rev() {
                    dec_hi.push(if i % 2 == 0 { val } else { -val });
                }
                let mut rec_lo = dec_lo.clone();
                rec_lo.reverse();
                let mut rec_hi = dec_hi.clone();
                rec_hi.reverse();
                Ok(WaveletFilters {
                    dec_lo,
                    dec_hi,
                    rec_lo,
                    rec_hi,
                })
            }
            5 => {
                let dec_lo: Vec<f64> = vec![
                    -9.604010112767894e-08,
                    -1.6237995172048338e-07,
                    2.0612203985788783e-06,
                    3.7007277113394796e-06,
                    -2.1270221672515614e-05,
                    -4.12198619242655e-05,
                    0.00014035632812373243,
                    0.0003018579416682448,
                    -0.0006375589261258812,
                    -0.0016616273039298788,
                    0.0024315754425382886,
                    0.006761520220620417,
                    -0.009159507338676163,
                    -0.019758391600965465,
                    0.032674799467057355,
                    0.041287530472117834,
                    -0.10556315130733723,
                    -0.06203775157498196,
                    0.4379823066591634,
                    0.7742936228603274,
                    0.42157126673075435,
                    -0.052046670253554764,
                    -0.09192158806008609,
                    0.028169744270532353,
                    0.023408322118927783,
                    -0.010131584846900276,
                    -0.00415931262757864,
                    0.0021782943778456947,
                    0.0003585777411617577,
                    -0.000212081862067494,
                ];
                let mut dec_hi: Vec<f64> = Vec::with_capacity(dec_lo.len());
                for (i, &val) in dec_lo.iter().enumerate().rev() {
                    dec_hi.push(if i % 2 == 0 { val } else { -val });
                }
                let mut rec_lo = dec_lo.clone();
                rec_lo.reverse();
                let mut rec_hi = dec_hi.clone();
                rec_hi.reverse();
                Ok(WaveletFilters {
                    dec_lo,
                    dec_hi,
                    rec_lo,
                    rec_hi,
                })
            }
            _ => Err(TransformError::InvalidInput(format!(
                "Coiflet-{} not implemented (supported: 1-5)",
                n
            ))),
        }
    }
    /// Biorthogonal (CDF/spline) wavelet filter banks: `bior{p}.{q}` in the
    /// standard PyWavelets/MATLAB naming (`p` = vanishing moments on the
    /// decomposition side, `q` = vanishing moments on the reconstruction
    /// side). Unlike orthogonal wavelets, decomposition and reconstruction
    /// use genuinely different filters (linear-phase / symmetric, at the
    /// cost of orthogonality).
    ///
    /// The two raw tables per family below (`table0`, the shared
    /// reconstruction-side lowpass, zero-padded to the family's longest
    /// member; and `table_q`, the `q`-specific decomposition-side lowpass)
    /// are the standard tabulated coefficients from PyWavelets
    /// (`pywt/_extensions/c/wavelets_coeffs.template.h`, arrays
    /// `bior{p}_0_` and `bior{p}_{q}_`). They are combined into
    /// `dec_lo`/`dec_hi`/`rec_lo`/`rec_hi` using the exact windowing +
    /// alternating-sign construction from PyWavelets' own C extension
    /// (`pywt/_extensions/c/wavelets.c`, `case BIOR`):
    ///   `dec_lo[i] = table_q[len-1-i]`
    ///   `rec_lo[i] = table0[i + (max_q - q)]`
    ///   `dec_hi[i] = (-1)^(len-1-i) * rec_lo[i]`
    ///   `rec_hi[i] = (-1)^i * dec_lo[i]`
    /// This exact assignment (not swapped) was verified against this
    /// crate's own decompose/reconstruct convention via a perfect-
    /// reconstruction round-trip test for each of the four supported
    /// wavelets (see the `bior*_perfect_reconstruction` tests below).
    fn biorthogonal(p: usize, q: usize) -> Result<Self> {
        let (table0, table_q, max_q): (Vec<f64>, Vec<f64>, usize) = match (p, q) {
            (1, 3) => (
                vec![
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    std::f64::consts::FRAC_1_SQRT_2,
                    std::f64::consts::FRAC_1_SQRT_2,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                vec![
                    -0.0883883476483184405501055452631,
                    0.0883883476483184405501055452631,
                    0.70710678118654752440084436210,
                    0.70710678118654752440084436210,
                    0.0883883476483184405501055452631,
                    -0.0883883476483184405501055452631,
                ],
                5,
            ),
            (2, 2) => (
                vec![
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.3535533905932737622004221810524,
                    0.7071067811865475244008443621048,
                    0.3535533905932737622004221810524,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                vec![
                    -0.1767766952966368811002110905262,
                    0.3535533905932737622004221810524,
                    1.0606601717798212866012665431573,
                    0.3535533905932737622004221810524,
                    -0.1767766952966368811002110905262,
                    0.0,
                ],
                8,
            ),
            (3, 1) => (
                vec![
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.1767766952966368811002110905262,
                    0.5303300858899106433006332715786,
                    0.5303300858899106433006332715786,
                    0.1767766952966368811002110905262,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                vec![
                    -0.3535533905932737622004221810524,
                    1.0606601717798212866012665431573,
                    1.0606601717798212866012665431573,
                    -0.3535533905932737622004221810524,
                ],
                9,
            ),
            (4, 4) => (
                vec![
                    0.0,
                    -0.064538882628697058,
                    -0.040689417609164058,
                    0.41809227322161724,
                    0.7884856164055829,
                    0.41809227322161724,
                    -0.040689417609164058,
                    -0.064538882628697058,
                    0.0,
                    0.0,
                ],
                vec![
                    0.03782845550726404,
                    -0.023849465019556843,
                    -0.11062440441843718,
                    0.37740285561283066,
                    0.85269867900889385,
                    0.37740285561283066,
                    -0.11062440441843718,
                    -0.023849465019556843,
                    0.03782845550726404,
                    0.0,
                ],
                4,
            ),
            _ => {
                return Err(TransformError::InvalidInput(format!(
                    "Biorthogonal {}.{} not implemented (supported: 1.3, 2.2, 3.1, 4.4)",
                    p, q
                )));
            }
        };
        let len = table_q.len();
        let window_start = max_q - q;
        let rec_lo: Vec<f64> = table0[window_start..window_start + len].to_vec();
        let mut dec_lo = table_q;
        dec_lo.reverse();
        let dec_hi: Vec<f64> = (0..len)
            .map(|i| {
                let sign = if (len - 1 - i) % 2 == 0 { 1.0 } else { -1.0 };
                sign * rec_lo[i]
            })
            .collect();
        let rec_hi: Vec<f64> = (0..len)
            .map(|i| {
                let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
                sign * dec_lo[i]
            })
            .collect();
        Ok(WaveletFilters {
            dec_lo,
            dec_hi,
            rec_lo,
            rec_hi,
        })
    }
}
/// 1D Discrete Wavelet Transform
#[derive(Debug, Clone)]
pub struct DWT {
    wavelet: WaveletType,
    filters: WaveletFilters,
    boundary: BoundaryMode,
    level: Option<usize>,
}
impl DWT {
    /// Create a new DWT instance
    pub fn new(wavelet: WaveletType) -> Result<Self> {
        let filters = WaveletFilters::from_wavelet(wavelet)?;
        Ok(DWT {
            wavelet,
            filters,
            boundary: BoundaryMode::Symmetric,
            level: None,
        })
    }
    /// Set the boundary mode
    pub fn with_boundary(mut self, boundary: BoundaryMode) -> Self {
        self.boundary = boundary;
        self
    }
    /// Set the decomposition level
    pub fn with_level(mut self, level: usize) -> Self {
        self.level = Some(level);
        self
    }
    /// Perform single-level decomposition
    pub fn decompose(&self, signal: &ArrayView1<f64>) -> Result<(Array1<f64>, Array1<f64>)> {
        let n = signal.len();
        if n < 2 {
            return Err(TransformError::InvalidInput(
                "Signal too short for DWT".to_string(),
            ));
        }
        let extended = self.extend_signal(signal)?;
        let approx = self.convolve_downsample(&extended, &self.filters.dec_lo)?;
        let detail = self.convolve_downsample(&extended, &self.filters.dec_hi)?;
        Ok((approx, detail))
    }
    /// Perform multi-level decomposition
    pub fn wavedec(&self, signal: &ArrayView1<f64>) -> Result<Vec<Array1<f64>>> {
        let max_level = self.max_decomposition_level(signal.len());
        let level = self.level.unwrap_or(max_level).min(max_level);
        let mut coeffs = Vec::with_capacity(level + 1);
        let mut current = signal.to_owned();
        for _ in 0..level {
            let (approx, detail) = self.decompose(&current.view())?;
            coeffs.push(detail);
            current = approx;
        }
        coeffs.push(current);
        coeffs.reverse();
        Ok(coeffs)
    }
    /// Perform single-level reconstruction
    pub fn reconstruct(
        &self,
        approx: &ArrayView1<f64>,
        detail: &ArrayView1<f64>,
    ) -> Result<Array1<f64>> {
        let approx_up = self.upsample_convolve(approx, &self.filters.rec_lo)?;
        let detail_up = self.upsample_convolve(detail, &self.filters.rec_hi)?;
        let min_len = approx_up.len().min(detail_up.len());
        let mut reconstructed = Array1::zeros(min_len);
        for i in 0..min_len {
            reconstructed[i] = approx_up[i] + detail_up[i];
        }
        Ok(reconstructed)
    }
    /// Perform multi-level reconstruction
    pub fn waverec(&self, coeffs: &[Array1<f64>]) -> Result<Array1<f64>> {
        if coeffs.is_empty() {
            return Err(TransformError::InvalidInput(
                "No coefficients provided for reconstruction".to_string(),
            ));
        }
        let mut current = coeffs[0].clone();
        for detail in &coeffs[1..] {
            current = self.reconstruct(&current.view(), &detail.view())?;
        }
        Ok(current)
    }
    fn extend_signal(&self, signal: &ArrayView1<f64>) -> Result<Array1<f64>> {
        let filter_len = self.filters.dec_lo.len();
        let n = signal.len();
        let pad_len = filter_len - 1;
        let mut extended = Array1::zeros(n + 2 * pad_len);
        match self.boundary {
            BoundaryMode::Zero => {
                for i in 0..n {
                    extended[i + pad_len] = signal[i];
                }
            }
            BoundaryMode::Constant => {
                let first = signal[0];
                let last = signal[n - 1];
                for i in 0..pad_len {
                    extended[i] = first;
                    extended[n + pad_len + i] = last;
                }
                for i in 0..n {
                    extended[i + pad_len] = signal[i];
                }
            }
            BoundaryMode::Symmetric => {
                for i in 0..pad_len {
                    extended[pad_len - 1 - i] = signal[i.min(n - 1)];
                    extended[n + pad_len + i] = signal[(n - 1 - i).max(0)];
                }
                for i in 0..n {
                    extended[i + pad_len] = signal[i];
                }
            }
            BoundaryMode::Periodic => {
                for i in 0..pad_len {
                    extended[i] = signal[(n - pad_len + i) % n];
                    extended[n + pad_len + i] = signal[i % n];
                }
                for i in 0..n {
                    extended[i + pad_len] = signal[i];
                }
            }
            BoundaryMode::Reflect => {
                for i in 0..pad_len {
                    let idx1 = if i < n { i } else { n - 1 };
                    let idx2 = if n > i + 1 { n - 1 - i } else { 0 };
                    extended[pad_len - 1 - i] = signal[idx1];
                    extended[n + pad_len + i] = signal[idx2];
                }
                for i in 0..n {
                    extended[i + pad_len] = signal[i];
                }
            }
        }
        Ok(extended)
    }
    fn convolve_downsample(&self, signal: &Array1<f64>, filter: &[f64]) -> Result<Array1<f64>> {
        let n = signal.len();
        let filter_len = filter.len();
        let output_len = (n + 1) / 2;
        let mut output = Array1::zeros(output_len);
        for i in 0..output_len {
            let pos = i * 2;
            let mut sum = 0.0;
            for (j, &coeff) in filter.iter().enumerate() {
                let idx = pos + j;
                if idx < n {
                    sum += signal[idx] * coeff;
                }
            }
            output[i] = sum;
        }
        Ok(output)
    }
    fn upsample_convolve(&self, signal: &ArrayView1<f64>, filter: &[f64]) -> Result<Array1<f64>> {
        let n = signal.len();
        let filter_len = filter.len();
        let output_len = n * 2;
        let mut output = Array1::zeros(output_len);
        let mut upsampled = Array1::zeros(output_len);
        for i in 0..n {
            upsampled[i * 2] = signal[i];
        }
        for i in 0..output_len {
            let mut sum = 0.0;
            for (j, &coeff) in filter.iter().enumerate() {
                if i >= j && i - j < output_len {
                    sum += upsampled[i - j] * coeff;
                }
            }
            output[i] = sum;
        }
        Ok(output)
    }
    fn max_decomposition_level(&self, signal_len: usize) -> usize {
        let filter_len = self.filters.dec_lo.len();
        let mut level: usize = 0;
        let mut current_len = signal_len;
        while current_len >= filter_len {
            current_len = (current_len + 1) / 2;
            level += 1;
        }
        level.saturating_sub(1)
    }
}
/// 2D Discrete Wavelet Transform
#[derive(Debug, Clone)]
pub struct DWT2D {
    wavelet: WaveletType,
    filters: WaveletFilters,
    boundary: BoundaryMode,
    level: Option<usize>,
}
impl DWT2D {
    /// Create a new DWT2D instance
    pub fn new(wavelet: WaveletType) -> Result<Self> {
        let filters = WaveletFilters::from_wavelet(wavelet)?;
        Ok(DWT2D {
            wavelet,
            filters,
            boundary: BoundaryMode::Symmetric,
            level: None,
        })
    }
    /// Set the boundary mode
    pub fn with_boundary(mut self, boundary: BoundaryMode) -> Self {
        self.boundary = boundary;
        self
    }
    /// Set the decomposition level
    pub fn with_level(mut self, level: usize) -> Self {
        self.level = Some(level);
        self
    }
    /// Perform single-level 2D decomposition
    pub fn decompose2(&self, image: &ArrayView2<f64>) -> Result<Dwt2dCoeffs> {
        let (rows, cols) = image.dim();
        if rows < 2 || cols < 2 {
            return Err(TransformError::InvalidInput(
                "Image too small for 2D DWT".to_string(),
            ));
        }
        let dwt1d = DWT {
            wavelet: self.wavelet,
            filters: self.filters.clone(),
            boundary: self.boundary,
            level: None,
        };
        let mut row_results_approx = Vec::with_capacity(rows);
        let mut row_results_detail = Vec::with_capacity(rows);
        for row_idx in 0..rows {
            let row = image.row(row_idx);
            let (approx, detail) = dwt1d.decompose(&row)?;
            row_results_approx.push(approx);
            row_results_detail.push(detail);
        }
        let approx_rows = row_results_approx[0].len();
        let detail_rows = row_results_detail[0].len();
        let mut approx_mat = Array2::zeros((rows, approx_rows));
        let mut detail_mat = Array2::zeros((rows, detail_rows));
        for (i, (app, det)) in row_results_approx
            .iter()
            .zip(row_results_detail.iter())
            .enumerate()
        {
            for (j, &val) in app.iter().enumerate() {
                approx_mat[[i, j]] = val;
            }
            for (j, &val) in det.iter().enumerate() {
                detail_mat[[i, j]] = val;
            }
        }
        let (ll, lh) = self.decompose_columns(&approx_mat.view(), &dwt1d)?;
        let (hl, hh) = self.decompose_columns(&detail_mat.view(), &dwt1d)?;
        Ok(Dwt2dCoeffs { ll, lh, hl, hh })
    }
    fn decompose_columns(
        &self,
        mat: &ArrayView2<f64>,
        dwt1d: &DWT,
    ) -> Result<(Array2<f64>, Array2<f64>)> {
        let (rows, cols) = mat.dim();
        let mut col_results_approx = Vec::with_capacity(cols);
        let mut col_results_detail = Vec::with_capacity(cols);
        for col_idx in 0..cols {
            let col = mat.column(col_idx);
            let (approx, detail) = dwt1d.decompose(&col)?;
            col_results_approx.push(approx);
            col_results_detail.push(detail);
        }
        let approx_cols = col_results_approx[0].len();
        let detail_cols = col_results_detail[0].len();
        let mut approx_result = Array2::zeros((approx_cols, cols));
        let mut detail_result = Array2::zeros((detail_cols, cols));
        for (j, (app, det)) in col_results_approx
            .iter()
            .zip(col_results_detail.iter())
            .enumerate()
        {
            for (i, &val) in app.iter().enumerate() {
                approx_result[[i, j]] = val;
            }
            for (i, &val) in det.iter().enumerate() {
                detail_result[[i, j]] = val;
            }
        }
        Ok((approx_result, detail_result))
    }
    /// Perform multi-level 2D decomposition
    pub fn wavedec2(&self, image: &ArrayView2<f64>) -> Result<Vec<Dwt2dCoeffs>> {
        let max_level = self.max_decomposition_level_2d(image.dim());
        let level = self.level.unwrap_or(max_level).min(max_level);
        let mut coeffs = Vec::with_capacity(level);
        let mut current = image.to_owned();
        for _ in 0..level {
            let dwt2d_coeffs = self.decompose2(&current.view())?;
            coeffs.push(dwt2d_coeffs.clone());
            current = dwt2d_coeffs.ll;
        }
        Ok(coeffs)
    }
    fn max_decomposition_level_2d(&self, shape: (usize, usize)) -> usize {
        let filter_len = self.filters.dec_lo.len();
        let min_dim = shape.0.min(shape.1);
        let mut level: usize = 0;
        let mut current_dim = min_dim;
        while current_dim >= filter_len {
            current_dim = (current_dim + 1) / 2;
            level += 1;
        }
        level.saturating_sub(1)
    }
}
/// 2D DWT coefficients (LL, LH, HL, HH)
#[derive(Debug, Clone)]
pub struct Dwt2dCoeffs {
    /// Approximation coefficients (low-low)
    pub ll: Array2<f64>,
    /// Horizontal detail coefficients (low-high)
    pub lh: Array2<f64>,
    /// Vertical detail coefficients (high-low)
    pub hl: Array2<f64>,
    /// Diagonal detail coefficients (high-high)
    pub hh: Array2<f64>,
}
/// 3D DWT coefficients — the 8 subbands produced by one level of separable 3D decomposition.
///
/// Naming convention: each letter indicates the filter applied along axis 0, 1, 2
/// respectively.  `L` = low-pass (approximation), `H` = high-pass (detail).
#[derive(Debug, Clone)]
pub struct Dwt3dCoeffs {
    /// LLL — approximation subband (low along all 3 axes)
    pub lll: Array3<f64>,
    /// LLH — detail along axis 2
    pub llh: Array3<f64>,
    /// LHL — detail along axis 1
    pub lhl: Array3<f64>,
    /// LHH — detail along axes 1 and 2
    pub lhh: Array3<f64>,
    /// HLL — detail along axis 0
    pub hll: Array3<f64>,
    /// HLH — detail along axes 0 and 2
    pub hlh: Array3<f64>,
    /// HHL — detail along axes 0 and 1
    pub hhl: Array3<f64>,
    /// HHH — detail along all 3 axes
    pub hhh: Array3<f64>,
}
/// N-D Discrete Wavelet Transform (supports 3D decomposition)
#[derive(Debug, Clone)]
pub struct DWTN {
    wavelet: WaveletType,
    boundary: BoundaryMode,
    level: Option<usize>,
}
impl DWTN {
    /// Create a new DWTN instance
    pub fn new(wavelet: WaveletType) -> Self {
        DWTN {
            wavelet,
            boundary: BoundaryMode::Symmetric,
            level: None,
        }
    }
    /// Set the boundary mode
    pub fn with_boundary(mut self, boundary: BoundaryMode) -> Self {
        self.boundary = boundary;
        self
    }
    /// Set the decomposition level
    pub fn with_level(mut self, level: usize) -> Self {
        self.level = Some(level);
        self
    }
    /// Perform single-level 3D DWT decomposition using separable filtering.
    ///
    /// The separable 3D DWT applies 1-D DWT independently along each of the three
    /// axes in sequence, producing 8 subbands (LLL through HHH).
    ///
    /// # Steps
    /// 1. Apply 1-D DWT along axis 0 → `Lo_x` and `Hi_x` halves.
    /// 2. For each of the 2 halves, apply 1-D DWT along axis 1 → 4 quarters.
    /// 3. For each of the 4 quarters, apply 1-D DWT along axis 2 → 8 subbands.
    pub fn decompose3(&self, volume: &Array3<f64>) -> Result<Dwt3dCoeffs> {
        let (d0, d1, d2) = volume.dim();
        if d0 < 2 || d1 < 2 || d2 < 2 {
            return Err(TransformError::InvalidInput(
                "Volume too small for 3D DWT: all dimensions must be >= 2".to_string(),
            ));
        }
        let dwt1d = DWT::new(self.wavelet)?.with_boundary(self.boundary);
        let half0 = (d0 + 1) / 2;
        let mut lo_x = Array3::<f64>::zeros((half0, d1, d2));
        let mut hi_x = Array3::<f64>::zeros((half0, d1, d2));
        for j in 0..d1 {
            for k in 0..d2 {
                let col: Vec<f64> = (0..d0).map(|i| volume[[i, j, k]]).collect();
                let col_arr = Array1::from(col);
                let (approx, detail) = dwt1d.decompose(&col_arr.view())?;
                for i in 0..approx.len().min(half0) {
                    lo_x[[i, j, k]] = approx[i];
                }
                for i in 0..detail.len().min(half0) {
                    hi_x[[i, j, k]] = detail[i];
                }
            }
        }
        let half1 = (d1 + 1) / 2;
        let (lo_x_lo_y, lo_x_hi_y) =
            self.apply_dwt_axis1_3d(&lo_x, &dwt1d, half0, d1, d2, half1)?;
        let (hi_x_lo_y, hi_x_hi_y) =
            self.apply_dwt_axis1_3d(&hi_x, &dwt1d, half0, d1, d2, half1)?;
        let half2 = (d2 + 1) / 2;
        let (lll, llh) = self.apply_dwt_axis2_3d(&lo_x_lo_y, &dwt1d, half0, half1, d2, half2)?;
        let (lhl, lhh) = self.apply_dwt_axis2_3d(&lo_x_hi_y, &dwt1d, half0, half1, d2, half2)?;
        let (hll, hlh) = self.apply_dwt_axis2_3d(&hi_x_lo_y, &dwt1d, half0, half1, d2, half2)?;
        let (hhl, hhh) = self.apply_dwt_axis2_3d(&hi_x_hi_y, &dwt1d, half0, half1, d2, half2)?;
        Ok(Dwt3dCoeffs {
            lll,
            llh,
            lhl,
            lhh,
            hll,
            hlh,
            hhl,
            hhh,
        })
    }
    fn apply_dwt_axis1_3d(
        &self,
        arr: &Array3<f64>,
        dwt1d: &DWT,
        size0: usize,
        size1: usize,
        size2: usize,
        out1: usize,
    ) -> Result<(Array3<f64>, Array3<f64>)> {
        let mut lo = Array3::<f64>::zeros((size0, out1, size2));
        let mut hi = Array3::<f64>::zeros((size0, out1, size2));
        for i in 0..size0 {
            for k in 0..size2 {
                let col: Vec<f64> = (0..size1).map(|j| arr[[i, j, k]]).collect();
                let col_arr = Array1::from(col);
                let (approx, detail) = dwt1d.decompose(&col_arr.view())?;
                for j in 0..approx.len().min(out1) {
                    lo[[i, j, k]] = approx[j];
                }
                for j in 0..detail.len().min(out1) {
                    hi[[i, j, k]] = detail[j];
                }
            }
        }
        Ok((lo, hi))
    }
    fn apply_dwt_axis2_3d(
        &self,
        arr: &Array3<f64>,
        dwt1d: &DWT,
        size0: usize,
        size1: usize,
        size2: usize,
        out2: usize,
    ) -> Result<(Array3<f64>, Array3<f64>)> {
        let mut lo = Array3::<f64>::zeros((size0, size1, out2));
        let mut hi = Array3::<f64>::zeros((size0, size1, out2));
        for i in 0..size0 {
            for j in 0..size1 {
                let row: Vec<f64> = (0..size2).map(|k| arr[[i, j, k]]).collect();
                let row_arr = Array1::from(row);
                let (approx, detail) = dwt1d.decompose(&row_arr.view())?;
                for k in 0..approx.len().min(out2) {
                    lo[[i, j, k]] = approx[k];
                }
                for k in 0..detail.len().min(out2) {
                    hi[[i, j, k]] = detail[k];
                }
            }
        }
        Ok((lo, hi))
    }
}

#[cfg(test)]
mod tests;
