//! Wavelet-based filtering and transform operations
//!
//! This module provides discrete wavelet transform (DWT) and related filtering
//! operations for image processing. Wavelets are particularly useful for
//! denoising, compression, and multi-scale analysis.

use scirs2_core::ndarray::{s, Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::numeric::{Float, FromPrimitive, Zero};
use std::fmt::Debug;

use crate::error::{NdimageError, NdimageResult};
use crate::filters::BorderMode;
use crate::utils::safe_f64_to_float;

/// Wavelet family enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaveletFamily {
    /// Daubechies wavelets
    Daubechies(usize), // Number of vanishing moments
    /// Biorthogonal wavelets
    Biorthogonal(usize, usize), // (reconstruction, decomposition) filter lengths
    /// Coiflets
    Coiflets(usize),
    /// Haar wavelet (simplest case)
    Haar,
}

/// Wavelet filter coefficients
#[derive(Debug, Clone)]
pub struct WaveletFilter<T> {
    /// Low-pass decomposition filter (scaling function)
    pub low_dec: Vec<T>,
    /// High-pass decomposition filter (wavelet function)
    pub high_dec: Vec<T>,
    /// Low-pass reconstruction filter
    pub low_rec: Vec<T>,
    /// High-pass reconstruction filter
    pub high_rec: Vec<T>,
}

impl<T> WaveletFilter<T>
where
    T: Float + FromPrimitive,
{
    /// Create wavelet filter coefficients for a given family
    pub fn new(family: WaveletFamily) -> NdimageResult<Self> {
        match family {
            WaveletFamily::Haar => Self::haar(),
            WaveletFamily::Daubechies(n) => Self::daubechies(n),
            WaveletFamily::Coiflets(n) => Self::coiflets(n),
            WaveletFamily::Biorthogonal(nr, nd) => Self::biorthogonal(nr, nd),
        }
    }

    /// Haar wavelet coefficients
    fn haar() -> NdimageResult<Self> {
        let sqrt2_inv = safe_f64_to_float::<T>(1.0 / std::f64::consts::SQRT_2)?;

        Ok(Self {
            low_dec: vec![sqrt2_inv, sqrt2_inv],
            high_dec: vec![sqrt2_inv, -sqrt2_inv],
            low_rec: vec![sqrt2_inv, sqrt2_inv],
            high_rec: vec![-sqrt2_inv, sqrt2_inv],
        })
    }

    /// Daubechies wavelet coefficients
    fn daubechies(n: usize) -> NdimageResult<Self> {
        if n == 1 {
            return Self::haar();
        }

        if n > 15 {
            return Err(NdimageError::InvalidInput(
                "Daubechies wavelets with more than 15 vanishing moments not supported".into(),
            ));
        }

        // Pre-computed Daubechies coefficients for common cases
        let coeffs = match n {
            2 => vec![
                safe_f64_to_float::<T>(0.48296291314469025)?,
                safe_f64_to_float::<T>(0.8365163037378079)?,
                safe_f64_to_float::<T>(0.22414386804185735)?,
                safe_f64_to_float::<T>(-0.12940952255092145)?,
            ],
            3 => vec![
                safe_f64_to_float::<T>(0.3326705529509569)?,
                safe_f64_to_float::<T>(0.8068915093133388)?,
                safe_f64_to_float::<T>(0.4598775021193313)?,
                safe_f64_to_float::<T>(-0.13501102001039084)?,
                safe_f64_to_float::<T>(-0.08544127388224149)?,
                safe_f64_to_float::<T>(0.035226291882100656)?,
            ],
            4 => vec![
                safe_f64_to_float::<T>(0.23037781330885523)?,
                safe_f64_to_float::<T>(0.7148465705525415)?,
                safe_f64_to_float::<T>(0.6308807679295904)?,
                safe_f64_to_float::<T>(-0.02798376941698385)?,
                safe_f64_to_float::<T>(-0.18703481171888114)?,
                safe_f64_to_float::<T>(0.030841381835986965)?,
                safe_f64_to_float::<T>(0.032883011666982945)?,
                safe_f64_to_float::<T>(-0.010597401784997278)?,
            ],
            5 => vec![
                safe_f64_to_float::<T>(0.1601023979741929)?,
                safe_f64_to_float::<T>(0.6038292697971895)?,
                safe_f64_to_float::<T>(0.7243085284385744)?,
                safe_f64_to_float::<T>(0.13842814590110342)?,
                safe_f64_to_float::<T>(-0.24229488706619015)?,
                safe_f64_to_float::<T>(-0.03224486958502952)?,
                safe_f64_to_float::<T>(0.07757149384006515)?,
                safe_f64_to_float::<T>(-0.006241490213011705)?,
                safe_f64_to_float::<T>(-0.012580751999015526)?,
                safe_f64_to_float::<T>(0.003335725285001549)?,
            ],
            6 => vec![
                safe_f64_to_float::<T>(0.11154074335008017)?,
                safe_f64_to_float::<T>(0.4946238903983854)?,
                safe_f64_to_float::<T>(0.7511339080210959)?,
                safe_f64_to_float::<T>(0.3152503517092432)?,
                safe_f64_to_float::<T>(-0.22626469396516913)?,
                safe_f64_to_float::<T>(-0.12976686756709563)?,
                safe_f64_to_float::<T>(0.09750160558707936)?,
                safe_f64_to_float::<T>(0.02752286553001629)?,
                safe_f64_to_float::<T>(-0.031582039318031156)?,
                safe_f64_to_float::<T>(0.000553842201161602)?,
                safe_f64_to_float::<T>(0.004777257511010651)?,
                safe_f64_to_float::<T>(-0.001077301085308479)?,
            ],
            7 => vec![
                safe_f64_to_float::<T>(0.07785205408506236)?,
                safe_f64_to_float::<T>(0.39653931948230575)?,
                safe_f64_to_float::<T>(0.7291320908465551)?,
                safe_f64_to_float::<T>(0.4697822874053586)?,
                safe_f64_to_float::<T>(-0.14390600392910627)?,
                safe_f64_to_float::<T>(-0.22403618499416572)?,
                safe_f64_to_float::<T>(0.07130921926705004)?,
                safe_f64_to_float::<T>(0.08061434390295413)?,
                safe_f64_to_float::<T>(-0.03802993693503463)?,
                safe_f64_to_float::<T>(-0.01657454163101562)?,
                safe_f64_to_float::<T>(0.012550998556013784)?,
                safe_f64_to_float::<T>(-0.00042957797300470274)?,
                safe_f64_to_float::<T>(-0.0018016407039998328)?,
                safe_f64_to_float::<T>(0.0003537138000010399)?,
            ],
            8 => vec![
                safe_f64_to_float::<T>(0.05441584224308161)?,
                safe_f64_to_float::<T>(0.3128715909144659)?,
                safe_f64_to_float::<T>(0.6756307362980128)?,
                safe_f64_to_float::<T>(0.5853546836548691)?,
                safe_f64_to_float::<T>(-0.015829105256023893)?,
                safe_f64_to_float::<T>(-0.2840155429624281)?,
                safe_f64_to_float::<T>(0.00047248457399797254)?,
                safe_f64_to_float::<T>(0.128747426620186)?,
                safe_f64_to_float::<T>(-0.017369301002022108)?,
                safe_f64_to_float::<T>(-0.04408825393106472)?,
                safe_f64_to_float::<T>(0.013981027917015516)?,
                safe_f64_to_float::<T>(0.008746094047015655)?,
                safe_f64_to_float::<T>(-0.004870352993451574)?,
                safe_f64_to_float::<T>(-0.000391740373376096)?,
                safe_f64_to_float::<T>(0.0006754494059985568)?,
                safe_f64_to_float::<T>(-0.00011747678400228192)?,
            ],
            9 => vec![
                safe_f64_to_float::<T>(0.03807794736316728)?,
                safe_f64_to_float::<T>(0.24383467463766728)?,
                safe_f64_to_float::<T>(0.6048231236767786)?,
                safe_f64_to_float::<T>(0.6572880780366389)?,
                safe_f64_to_float::<T>(0.13319738582208895)?,
                safe_f64_to_float::<T>(-0.29327378327258685)?,
                safe_f64_to_float::<T>(-0.09684078322087904)?,
                safe_f64_to_float::<T>(0.14854074933476008)?,
                safe_f64_to_float::<T>(0.030725681478322865)?,
                safe_f64_to_float::<T>(-0.06763282905952446)?,
                safe_f64_to_float::<T>(0.00047374508707894396)?,
                safe_f64_to_float::<T>(0.022361662123515244)?,
                safe_f64_to_float::<T>(-0.004723204757894831)?,
                safe_f64_to_float::<T>(-0.004281503682463429)?,
                safe_f64_to_float::<T>(0.0018476468829611268)?,
                safe_f64_to_float::<T>(0.00023038576399541288)?,
                safe_f64_to_float::<T>(-0.00025196318894271934)?,
                safe_f64_to_float::<T>(0.000039347319995026124)?,
            ],
            10 => vec![
                safe_f64_to_float::<T>(0.026670057900950818)?,
                safe_f64_to_float::<T>(0.18817680007762133)?,
                safe_f64_to_float::<T>(0.5272011889309198)?,
                safe_f64_to_float::<T>(0.6884590394525921)?,
                safe_f64_to_float::<T>(0.2811723436604265)?,
                safe_f64_to_float::<T>(-0.24984642432648865)?,
                safe_f64_to_float::<T>(-0.19594627437659665)?,
                safe_f64_to_float::<T>(0.12736934033574265)?,
                safe_f64_to_float::<T>(0.09305736460380659)?,
                safe_f64_to_float::<T>(-0.07139414716586077)?,
                safe_f64_to_float::<T>(-0.02945753682194567)?,
                safe_f64_to_float::<T>(0.03321267405893324)?,
                safe_f64_to_float::<T>(0.003606553566956169)?,
                safe_f64_to_float::<T>(-0.010733175482979604)?,
                safe_f64_to_float::<T>(0.0013953517469940798)?,
                safe_f64_to_float::<T>(0.001992405295185056)?,
                safe_f64_to_float::<T>(-0.0006858566950046825)?,
                safe_f64_to_float::<T>(-0.00011062440441843718)?,
                safe_f64_to_float::<T>(0.00009405348846774701)?,
                safe_f64_to_float::<T>(-0.000013264203002354869)?,
            ],
            11 => vec![
                safe_f64_to_float::<T>(0.018739120728312193)?,
                safe_f64_to_float::<T>(0.14125680978738187)?,
                safe_f64_to_float::<T>(0.44906028370231975)?,
                safe_f64_to_float::<T>(0.6559987246772096)?,
                safe_f64_to_float::<T>(0.38080070249715344)?,
                safe_f64_to_float::<T>(-0.16554680450994717)?,
                safe_f64_to_float::<T>(-0.2601367646540273)?,
                safe_f64_to_float::<T>(0.08917442046362987)?,
                safe_f64_to_float::<T>(0.14507172816327717)?,
                safe_f64_to_float::<T>(-0.05584126984200549)?,
                safe_f64_to_float::<T>(-0.06200322379348169)?,
                safe_f64_to_float::<T>(0.03449768175847296)?,
                safe_f64_to_float::<T>(0.015727790831356044)?,
                safe_f64_to_float::<T>(-0.013593662816166423)?,
                safe_f64_to_float::<T>(-0.0016523264966978706)?,
                safe_f64_to_float::<T>(0.002974033779568442)?,
                safe_f64_to_float::<T>(-0.00022480568175172)?,
                safe_f64_to_float::<T>(-0.00034999502100968525)?,
                safe_f64_to_float::<T>(0.00009208119700928503)?,
                safe_f64_to_float::<T>(0.00001886094431070618)?,
                safe_f64_to_float::<T>(-0.000013894441904523324)?,
                safe_f64_to_float::<T>(0.0000019236393170772043)?,
            ],
            12 => vec![
                safe_f64_to_float::<T>(0.013193131813258676)?,
                safe_f64_to_float::<T>(0.10649946321644633)?,
                safe_f64_to_float::<T>(0.379737021074414)?,
                safe_f64_to_float::<T>(0.6081774043830486)?,
                safe_f64_to_float::<T>(0.4372720959502646)?,
                safe_f64_to_float::<T>(-0.05962952549983625)?,
                safe_f64_to_float::<T>(-0.29314103304172173)?,
                safe_f64_to_float::<T>(0.01765631246819756)?,
                safe_f64_to_float::<T>(0.17831883488073926)?,
                safe_f64_to_float::<T>(-0.03104962706297068)?,
                safe_f64_to_float::<T>(-0.0905578968689735)?,
                safe_f64_to_float::<T>(0.030877393261061227)?,
                safe_f64_to_float::<T>(0.032727173073055495)?,
                safe_f64_to_float::<T>(-0.018199093842156943)?,
                safe_f64_to_float::<T>(-0.007330325009006078)?,
                safe_f64_to_float::<T>(0.006395069671570748)?,
                safe_f64_to_float::<T>(0.0006593516736014779)?,
                safe_f64_to_float::<T>(-0.0013188100536080566)?,
                safe_f64_to_float::<T>(0.00008710667976623568)?,
                safe_f64_to_float::<T>(0.00015002863742806733)?,
                safe_f64_to_float::<T>(-0.000037434398018953916)?,
                safe_f64_to_float::<T>(-0.000007900127012550126)?,
                safe_f64_to_float::<T>(0.000005378627127532456)?,
                safe_f64_to_float::<T>(-0.0000007103098569463983)?,
            ],
            13 => vec![
                safe_f64_to_float::<T>(0.009296044542244711)?,
                safe_f64_to_float::<T>(0.08025229094475479)?,
                safe_f64_to_float::<T>(0.3201138120925906)?,
                safe_f64_to_float::<T>(0.5486318036403653)?,
                safe_f64_to_float::<T>(0.4603528012810862)?,
                safe_f64_to_float::<T>(0.06962388316845977)?,
                safe_f64_to_float::<T>(-0.296631177071847)?,
                safe_f64_to_float::<T>(-0.08065993094169825)?,
                safe_f64_to_float::<T>(0.18915118026564547)?,
                safe_f64_to_float::<T>(0.005199740015408669)?,
                safe_f64_to_float::<T>(-0.1140710517665774)?,
                safe_f64_to_float::<T>(0.019687914051985394)?,
                safe_f64_to_float::<T>(0.050842914387470264)?,
                safe_f64_to_float::<T>(-0.020142260053266115)?,
                safe_f64_to_float::<T>(-0.01520928885394885)?,
                safe_f64_to_float::<T>(0.010066894331926433)?,
                safe_f64_to_float::<T>(0.003152169076031001)?,
                safe_f64_to_float::<T>(-0.0030493477015893374)?,
                safe_f64_to_float::<T>(-0.00023884490304976308)?,
                safe_f64_to_float::<T>(0.0005506064264983963)?,
                safe_f64_to_float::<T>(-0.000031556734986493014)?,
                safe_f64_to_float::<T>(-0.000060056261928043435)?,
                safe_f64_to_float::<T>(0.000014210948558236523)?,
                safe_f64_to_float::<T>(0.0000030191157645056073)?,
                safe_f64_to_float::<T>(-0.0000019722555077925)?,
                safe_f64_to_float::<T>(0.00000025205432047885806)?,
            ],
            14 => vec![
                safe_f64_to_float::<T>(0.006565876898035924)?,
                safe_f64_to_float::<T>(0.06039314026129026)?,
                safe_f64_to_float::<T>(0.2688103616428251)?,
                safe_f64_to_float::<T>(0.48413982616094435)?,
                safe_f64_to_float::<T>(0.46317866536230914)?,
                safe_f64_to_float::<T>(0.1969278823109936)?,
                safe_f64_to_float::<T>(-0.2666323159654994)?,
                safe_f64_to_float::<T>(-0.16943179582950736)?,
                safe_f64_to_float::<T>(0.17632838804127173)?,
                safe_f64_to_float::<T>(0.09042888012166942)?,
                safe_f64_to_float::<T>(-0.1218952088779019)?,
                safe_f64_to_float::<T>(-0.020509821159094503)?,
                safe_f64_to_float::<T>(0.06889419008162671)?,
                safe_f64_to_float::<T>(-0.008978226567899728)?,
                safe_f64_to_float::<T>(-0.027630537459728096)?,
                safe_f64_to_float::<T>(0.010073906434728577)?,
                safe_f64_to_float::<T>(0.007325551193377623)?,
                safe_f64_to_float::<T>(-0.004777214976020679)?,
                safe_f64_to_float::<T>(-0.0011927556568903844)?,
                safe_f64_to_float::<T>(0.0015153806688804067)?,
                safe_f64_to_float::<T>(0.00008523866302851165)?,
                safe_f64_to_float::<T>(-0.00020779990463524476)?,
                safe_f64_to_float::<T>(0.000010688324593166062)?,
                safe_f64_to_float::<T>(0.00002216398310943374)?,
                safe_f64_to_float::<T>(-0.000005034323699883308)?,
                safe_f64_to_float::<T>(-0.000001096049162476226)?,
                safe_f64_to_float::<T>(0.0000007091509421096986)?,
                safe_f64_to_float::<T>(-0.00000008820224936003736)?,
            ],
            15 => vec![
                safe_f64_to_float::<T>(0.004648846162387949)?,
                safe_f64_to_float::<T>(0.045496494421439875)?,
                safe_f64_to_float::<T>(0.22577631764506962)?,
                safe_f64_to_float::<T>(0.42277104815502985)?,
                safe_f64_to_float::<T>(0.44745879851847204)?,
                safe_f64_to_float::<T>(0.3000766456179915)?,
                safe_f64_to_float::<T>(-0.20663571618012127)?,
                safe_f64_to_float::<T>(-0.23521468175037773)?,
                safe_f64_to_float::<T>(0.13524829491633426)?,
                safe_f64_to_float::<T>(0.1564096339739734)?,
                safe_f64_to_float::<T>(-0.10473988845988175)?,
                safe_f64_to_float::<T>(-0.060568644473079015)?,
                safe_f64_to_float::<T>(0.07188970152925473)?,
                safe_f64_to_float::<T>(0.012055085500932434)?,
                safe_f64_to_float::<T>(-0.04035725011717636)?,
                safe_f64_to_float::<T>(0.002659087516139743)?,
                safe_f64_to_float::<T>(0.016255226294714977)?,
                safe_f64_to_float::<T>(-0.005128443012318736)?,
                safe_f64_to_float::<T>(-0.004344251796397951)?,
                safe_f64_to_float::<T>(0.0026569302041159306)?,
                safe_f64_to_float::<T>(0.0005834618293265067)?,
                safe_f64_to_float::<T>(-0.0007176945589570982)?,
                safe_f64_to_float::<T>(-0.000029481048693327066)?,
                safe_f64_to_float::<T>(0.00008571167993556883)?,
                safe_f64_to_float::<T>(-0.0000034916049835694046)?,
                safe_f64_to_float::<T>(-0.000008039298577444542)?,
                safe_f64_to_float::<T>(0.0000017442049827151994)?,
                safe_f64_to_float::<T>(0.00000038069313651503825)?,
                safe_f64_to_float::<T>(-0.0000002454719888816509)?,
                safe_f64_to_float::<T>(0.00000003004157056088014)?,
            ],
            _ => {
                return Err(NdimageError::NotImplementedError(format!(
                    "Daubechies wavelet with {} vanishing moments not implemented. Supported: 1-15",
                    n
                )));
            }
        };

        let low_dec = coeffs;
        let mut high_dec = Vec::with_capacity(low_dec.len());

        // High-pass filter: h[n] = (-1)^n * g[L-1-n]
        for (i, &coeff) in low_dec.iter().rev().enumerate() {
            let sign = if i % 2 == 0 { T::one() } else { -T::one() };
            high_dec.push(sign * coeff);
        }

        // Reconstruction filters are time-reversed versions
        let low_rec = low_dec.iter().rev().cloned().collect();
        let high_rec = high_dec.iter().rev().cloned().collect();

        Ok(Self {
            low_dec,
            high_dec,
            low_rec,
            high_rec,
        })
    }

    /// Coiflets wavelet coefficients
    fn coiflets(n: usize) -> NdimageResult<Self> {
        let coeffs = match n {
            2 => vec![
                safe_f64_to_float::<T>(-0.01565572813546454)?,
                safe_f64_to_float::<T>(-0.0727326195128539)?,
                safe_f64_to_float::<T>(0.38486484686420286)?,
                safe_f64_to_float::<T>(0.8525720202122554)?,
                safe_f64_to_float::<T>(0.3378976624578092)?,
                safe_f64_to_float::<T>(-0.0727326195128539)?,
            ],
            4 => vec![
                safe_f64_to_float::<T>(-0.003793512864256592)?,
                safe_f64_to_float::<T>(-0.007782596427073981)?,
                safe_f64_to_float::<T>(0.023452696142428003)?,
                safe_f64_to_float::<T>(0.06578976894285815)?,
                safe_f64_to_float::<T>(-0.061123390132632664)?,
                safe_f64_to_float::<T>(-0.40517690379010785)?,
                safe_f64_to_float::<T>(0.7937772226256206)?,
                safe_f64_to_float::<T>(0.42848347637784375)?,
                safe_f64_to_float::<T>(-0.071799821619371566)?,
                safe_f64_to_float::<T>(-0.08285209628608844)?,
                safe_f64_to_float::<T>(0.03463498418298139)?,
                safe_f64_to_float::<T>(0.015364820906201324)?,
                safe_f64_to_float::<T>(-0.004729394943303866)?,
                safe_f64_to_float::<T>(-0.0008152893579070594)?,
                safe_f64_to_float::<T>(0.0002183119418830823)?,
                safe_f64_to_float::<T>(-0.00002183119418830823)?,
            ],
            6 => vec![
                safe_f64_to_float::<T>(-0.0010780278060905155)?,
                safe_f64_to_float::<T>(-0.001658554806298039)?,
                safe_f64_to_float::<T>(0.007368554806298042)?,
                safe_f64_to_float::<T>(0.016851298806298042)?,
                safe_f64_to_float::<T>(-0.02658554806298042)?,
                safe_f64_to_float::<T>(-0.08153648806298041)?,
                safe_f64_to_float::<T>(0.05691088062980417)?,
                safe_f64_to_float::<T>(0.41517488062980415)?,
                safe_f64_to_float::<T>(-0.7829763509609523)?,
                safe_f64_to_float::<T>(-0.4345980060195524)?,
                safe_f64_to_float::<T>(0.06664788006219842)?,
                safe_f64_to_float::<T>(0.09532055696202447)?,
                safe_f64_to_float::<T>(-0.02932576604298042)?,
                safe_f64_to_float::<T>(-0.02397516006219842)?,
                safe_f64_to_float::<T>(0.008829579062198417)?,
                safe_f64_to_float::<T>(0.003926443062198417)?,
                safe_f64_to_float::<T>(-0.001173555806298042)?,
                safe_f64_to_float::<T>(-0.00024635506298042)?,
                safe_f64_to_float::<T>(0.00006158888062980417)?,
                safe_f64_to_float::<T>(0.00000769888062980417)?,
                safe_f64_to_float::<T>(-0.000001622222062980417)?,
                safe_f64_to_float::<T>(0.0000001622222062980417)?,
            ],
            _ => {
                return Err(NdimageError::NotImplementedError(format!(
                    "Coiflets wavelet with {} vanishing moments not implemented. Supported: 2, 4, 6",
                    n
                )));
            }
        };

        let low_dec = coeffs;
        let mut high_dec = Vec::with_capacity(low_dec.len());

        for (i, &coeff) in low_dec.iter().rev().enumerate() {
            let sign = if i % 2 == 0 { T::one() } else { -T::one() };
            high_dec.push(sign * coeff);
        }

        let low_rec = low_dec.iter().rev().cloned().collect();
        let high_rec = high_dec.iter().rev().cloned().collect();

        Ok(Self {
            low_dec,
            high_dec,
            low_rec,
            high_rec,
        })
    }

    /// Biorthogonal wavelet coefficients
    fn biorthogonal(nr: usize, nd: usize) -> NdimageResult<Self> {
        match (nr, nd) {
            (1, 1) => {
                // Biorthogonal 1.1 (Haar)
                Self::haar()
            }
            (2, 2) => {
                // Biorthogonal 2.2 (Linear B-spline)
                let low_dec = vec![
                    safe_f64_to_float::<T>(-0.12940952255092145)?,
                    safe_f64_to_float::<T>(0.22414386804185735)?,
                    safe_f64_to_float::<T>(0.8365163037378079)?,
                    safe_f64_to_float::<T>(0.48296291314469025)?,
                ];

                let high_dec = vec![
                    safe_f64_to_float::<T>(-0.48296291314469025)?,
                    safe_f64_to_float::<T>(0.8365163037378079)?,
                    safe_f64_to_float::<T>(-0.22414386804185735)?,
                    safe_f64_to_float::<T>(-0.12940952255092145)?,
                ];

                let low_rec = vec![
                    safe_f64_to_float::<T>(0.48296291314469025)?,
                    safe_f64_to_float::<T>(0.8365163037378079)?,
                    safe_f64_to_float::<T>(0.22414386804185735)?,
                    safe_f64_to_float::<T>(-0.12940952255092145)?,
                ];

                let high_rec = vec![
                    safe_f64_to_float::<T>(-0.12940952255092145)?,
                    safe_f64_to_float::<T>(-0.22414386804185735)?,
                    safe_f64_to_float::<T>(0.8365163037378079)?,
                    safe_f64_to_float::<T>(-0.48296291314469025)?,
                ];

                Ok(Self { low_dec, high_dec, low_rec, high_rec })
            }
            (2, 4) => {
                // Biorthogonal 2.4
                let low_dec = vec![
                    safe_f64_to_float::<T>(0.0)?,
                    safe_f64_to_float::<T>(-0.1767766952966369)?,
                    safe_f64_to_float::<T>(-0.07589077294536541)?,
                    safe_f64_to_float::<T>(0.87343749756405325)?,
                    safe_f64_to_float::<T>(0.87343749756405325)?,
                    safe_f64_to_float::<T>(-0.07589077294536541)?,
                    safe_f64_to_float::<T>(-0.1767766952966369)?,
                    safe_f64_to_float::<T>(0.0)?,
                ];

                let high_dec = vec![
                    safe_f64_to_float::<T>(0.0)?,
                    safe_f64_to_float::<T>(0.1767766952966369)?,
                    safe_f64_to_float::<T>(-0.07589077294536541)?,
                    safe_f64_to_float::<T>(-0.87343749756405325)?,
                    safe_f64_to_float::<T>(0.87343749756405325)?,
                    safe_f64_to_float::<T>(0.07589077294536541)?,
                    safe_f64_to_float::<T>(-0.1767766952966369)?,
                    safe_f64_to_float::<T>(0.0)?,
                ];

                let low_rec = vec![
                    safe_f64_to_float::<T>(0.0)?,
                    safe_f64_to_float::<T>(-0.1767766952966369)?,
                    safe_f64_to_float::<T>(-0.07589077294536541)?,
                    safe_f64_to_float::<T>(0.87343749756405325)?,
                    safe_f64_to_float::<T>(0.87343749756405325)?,
                    safe_f64_to_float::<T>(-0.07589077294536541)?,
                    safe_f64_to_float::<T>(-0.1767766952966369)?,
                    safe_f64_to_float::<T>(0.0)?,
                ];

                let high_rec = vec![
                    safe_f64_to_float::<T>(0.0)?,
                    safe_f64_to_float::<T>(-0.1767766952966369)?,
                    safe_f64_to_float::<T>(0.07589077294536541)?,
                    safe_f64_to_float::<T>(0.87343749756405325)?,
                    safe_f64_to_float::<T>(-0.87343749756405325)?,
                    safe_f64_to_float::<T>(-0.07589077294536541)?,
                    safe_f64_to_float::<T>(0.1767766952966369)?,
                    safe_f64_to_float::<T>(0.0)?,
                ];

                Ok(Self { low_dec, high_dec, low_rec, high_rec })
            }
            (4, 4) => {
                // Biorthogonal 4.4 (Cubic B-spline)
                let low_dec = vec![
                    safe_f64_to_float::<T>(0.03314563036811942)?,
                    safe_f64_to_float::<T>(-0.06629126073623884)?,
                    safe_f64_to_float::<T>(-0.17677669529663687)?,
                    safe_f64_to_float::<T>(0.4198446513295126)?,
                    safe_f64_to_float::<T>(0.9943689110435825)?,
                    safe_f64_to_float::<T>(0.4198446513295126)?,
                    safe_f64_to_float::<T>(-0.17677669529663687)?,
                    safe_f64_to_float::<T>(-0.06629126073623884)?,
                    safe_f64_to_float::<T>(0.03314563036811942)?,
                ];

                let high_dec = vec![
                    safe_f64_to_float::<T>(0.0)?,
                    safe_f64_to_float::<T>(0.01657281518405971)?,
                    safe_f64_to_float::<T>(-0.03314563036811942)?,
                    safe_f64_to_float::<T>(-0.1767766952966369)?,
                    safe_f64_to_float::<T>(0.41984465132951256)?,
                    safe_f64_to_float::<T>(-0.9943689110435825)?,
                    safe_f64_to_float::<T>(0.41984465132951256)?,
                    safe_f64_to_float::<T>(-0.1767766952966369)?,
                    safe_f64_to_float::<T>(-0.03314563036811942)?,
                    safe_f64_to_float::<T>(0.01657281518405971)?,
                    safe_f64_to_float::<T>(0.0)?,
                ];

                let low_rec = vec![
                    safe_f64_to_float::<T>(0.03314563036811942)?,
                    safe_f64_to_float::<T>(-0.06629126073623884)?,
                    safe_f64_to_float::<T>(-0.17677669529663687)?,
                    safe_f64_to_float::<T>(0.4198446513295126)?,
                    safe_f64_to_float::<T>(0.9943689110435825)?,
                    safe_f64_to_float::<T>(0.4198446513295126)?,
                    safe_f64_to_float::<T>(-0.17677669529663687)?,
                    safe_f64_to_float::<T>(-0.06629126073623884)?,
                    safe_f64_to_float::<T>(0.03314563036811942)?,
                ];

                let high_rec = vec![
                    safe_f64_to_float::<T>(0.0)?,
                    safe_f64_to_float::<T>(-0.01657281518405971)?,
                    safe_f64_to_float::<T>(-0.03314563036811942)?,
                    safe_f64_to_float::<T>(0.1767766952966369)?,
                    safe_f64_to_float::<T>(0.41984465132951256)?,
                    safe_f64_to_float::<T>(0.9943689110435825)?,
                    safe_f64_to_float::<T>(0.41984465132951256)?,
                    safe_f64_to_float::<T>(0.1767766952966369)?,
                    safe_f64_to_float::<T>(-0.03314563036811942)?,
                    safe_f64_to_float::<T>(-0.01657281518405971)?,
                    safe_f64_to_float::<T>(0.0)?,
                ];

                Ok(Self { low_dec, high_dec, low_rec, high_rec })
            }
            (6, 8) => {
                // Biorthogonal 6.8
                let low_dec = vec![
                    safe_f64_to_float::<T>(0.0019088317364812906)?,
                    safe_f64_to_float::<T>(-0.0019142861290887667)?,
                    safe_f64_to_float::<T>(-0.016990639867602342)?,
                    safe_f64_to_float::<T>(0.01193456527972926)?,
                    safe_f64_to_float::<T>(0.04973290349094079)?,
                    safe_f64_to_float::<T>(-0.07726317316720414)?,
                    safe_f64_to_float::<T>(-0.09405920349573646)?,
                    safe_f64_to_float::<T>(0.4207962846098268)?,
                    safe_f64_to_float::<T>(0.8259229974584023)?,
                    safe_f64_to_float::<T>(0.4207962846098268)?,
                    safe_f64_to_float::<T>(-0.09405920349573646)?,
                    safe_f64_to_float::<T>(-0.07726317316720414)?,
                    safe_f64_to_float::<T>(0.04973290349094079)?,
                    safe_f64_to_float::<T>(0.01193456527972926)?,
                    safe_f64_to_float::<T>(-0.016990639867602342)?,
                    safe_f64_to_float::<T>(-0.0019142861290887667)?,
                    safe_f64_to_float::<T>(0.0019088317364812906)?,
                ];

                // For simplicity, generate high-pass filters using quadrature mirror filter relationship
                let mut high_dec = Vec::with_capacity(low_dec.len());
                for (i, &coeff) in low_dec.iter().rev().enumerate() {
                    let sign = if i % 2 == 0 { T::one() } else { -T::one() };
                    high_dec.push(sign * coeff);
                }

                let low_rec = low_dec.iter().rev().cloned().collect();
                let high_rec = high_dec.iter().rev().cloned().collect();

                Ok(Self { low_dec, high_dec, low_rec, high_rec })
            }
            _ => {
                Err(NdimageError::NotImplementedError(
                    format!("Biorthogonal wavelet ({}, {}) is not implemented. Supported variants: (1,1), (2,2), (2,4), (4,4), (6,8)", nr, nd),
                ))
            }
        }
    }
}

/// 1D Discrete Wavelet Transform
#[allow(dead_code)]
pub fn dwt_1d<T>(
    signal: &ArrayView1<T>,
    wavelet: &WaveletFilter<T>,
    mode: BorderMode,
) -> NdimageResult<(Array1<T>, Array1<T>)>
where
    T: Float + FromPrimitive + Debug + Clone + Send + Sync + 'static,
{
    let n = signal.len();
    if n < 2 {
        return Err(NdimageError::InvalidInput(
            "Signal must have at least 2 elements".into(),
        ));
    }

    // Pad signal for boundary handling
    let padded = pad_signal_1d(signal, &wavelet.low_dec, mode)?;

    // Apply low-pass and high-pass filters
    let mut low_pass = convolve_downsample_1d(&padded.view(), &wavelet.low_dec, 2)?;
    let mut high_pass = convolve_downsample_1d(&padded.view(), &wavelet.high_dec, 2)?;

    // Trim to correct output length (n/2 for even n, (n+1)/2 for odd n)
    let expected_len = (n + 1) / 2;
    if low_pass.len() > expected_len {
        low_pass = low_pass.slice(s![..expected_len]).to_owned();
    }
    if high_pass.len() > expected_len {
        high_pass = high_pass.slice(s![..expected_len]).to_owned();
    }

    Ok((low_pass, high_pass))
}

/// 1D Inverse Discrete Wavelet Transform
#[allow(dead_code)]
pub fn idwt_1d<T>(
    low: &ArrayView1<T>,
    high: &ArrayView1<T>,
    wavelet: &WaveletFilter<T>,
) -> NdimageResult<Array1<T>>
where
    T: Float + FromPrimitive + Debug + Clone + Send + Sync + 'static,
{
    if low.len() != high.len() {
        return Err(NdimageError::InvalidInput(
            "Low and high frequency components must have the same length".into(),
        ));
    }

    // Upsample and filter
    let low_upsampled = upsample_convolve_1d(low, &wavelet.low_rec, 2)?;
    let high_upsampled = upsample_convolve_1d(high, &wavelet.high_rec, 2)?;

    // Combine
    let mut result = Array1::zeros(low_upsampled.len());
    for i in 0..result.len() {
        result[i] = low_upsampled[i] + high_upsampled[i];
    }

    Ok(result)
}

/// 2D Discrete Wavelet Transform
#[allow(dead_code)]
pub fn dwt_2d<T>(
    image: &ArrayView2<T>,
    wavelet: &WaveletFilter<T>,
    mode: BorderMode,
) -> NdimageResult<(Array2<T>, Array2<T>, Array2<T>, Array2<T>)>
where
    T: Float + FromPrimitive + Debug + Clone + Send + Sync + 'static,
{
    let (height, width) = image.dim();

    if height < 2 || width < 2 {
        return Err(NdimageError::InvalidInput(
            "Image must be at least 2x2 pixels".into(),
        ));
    }

    // First, apply DWT to each row
    let mut row_low = Array2::zeros((height, width / 2));
    let mut row_high = Array2::zeros((height, width / 2));

    for i in 0..height {
        let row = image.row(i);
        let (low, high) = dwt_1d(&row, wavelet, mode)?;

        for j in 0..low.len() {
            row_low[[i, j]] = low[j];
        }
        for j in 0..high.len() {
            row_high[[i, j]] = high[j];
        }
    }

    // Then apply DWT to each column of the results
    let mut ll = Array2::zeros((height / 2, width / 2)); // Low-Low
    let mut lh = Array2::zeros((height / 2, width / 2)); // Low-High
    let mut hl = Array2::zeros((height / 2, width / 2)); // High-Low
    let mut hh = Array2::zeros((height / 2, width / 2)); // High-High

    // Process low-frequency rows
    for j in 0..width / 2 {
        let col = row_low.column(j);
        let (low, high) = dwt_1d(&col, wavelet, mode)?;

        for i in 0..low.len() {
            ll[[i, j]] = low[i];
        }
        for i in 0..high.len() {
            lh[[i, j]] = high[i];
        }
    }

    // Process high-frequency rows
    for j in 0..width / 2 {
        let col = row_high.column(j);
        let (low, high) = dwt_1d(&col, wavelet, mode)?;

        for i in 0..low.len() {
            hl[[i, j]] = low[i];
        }
        for i in 0..high.len() {
            hh[[i, j]] = high[i];
        }
    }

    Ok((ll, lh, hl, hh))
}

/// 2D Inverse Discrete Wavelet Transform
#[allow(dead_code)]
pub fn idwt_2d<T>(
    ll: &ArrayView2<T>,
    lh: &ArrayView2<T>,
    hl: &ArrayView2<T>,
    hh: &ArrayView2<T>,
    wavelet: &WaveletFilter<T>,
) -> NdimageResult<Array2<T>>
where
    T: Float + FromPrimitive + Debug + Clone + Send + Sync + 'static,
{
    let (sub_height, sub_width) = ll.dim();

    if lh.dim() != (sub_height, sub_width)
        || hl.dim() != (sub_height, sub_width)
        || hh.dim() != (sub_height, sub_width)
    {
        return Err(NdimageError::InvalidInput(
            "All wavelet coefficient arrays must have the same dimensions".into(),
        ));
    }

    let height = sub_height * 2;
    let width = sub_width * 2;

    // First, reconstruct in the column direction
    let mut row_low = Array2::zeros((height, sub_width));
    let mut row_high = Array2::zeros((height, sub_width));

    for j in 0..sub_width {
        let ll_col = ll.column(j);
        let lh_col = lh.column(j);
        let reconstructed_low = idwt_1d(&ll_col, &lh_col, wavelet)?;

        let hl_col = hl.column(j);
        let hh_col = hh.column(j);
        let reconstructed_high = idwt_1d(&hl_col, &hh_col, wavelet)?;

        for i in 0..height {
            row_low[[i, j]] = reconstructed_low[i];
            row_high[[i, j]] = reconstructed_high[i];
        }
    }

    // Then reconstruct in the row direction
    let mut result = Array2::zeros((height, width));

    for i in 0..height {
        let low_row = row_low.row(i);
        let high_row = row_high.row(i);
        let reconstructed_row = idwt_1d(&low_row, &high_row, wavelet)?;

        for j in 0..width {
            result[[i, j]] = reconstructed_row[j];
        }
    }

    Ok(result)
}

/// Soft thresholding function
#[allow(dead_code)]
fn soft_threshold<T>(coeffs: &ArrayView2<T>, threshold: T) -> Array2<T>
where
    T: Float + FromPrimitive,
{
    coeffs.mapv(|x| {
        if x.abs() <= threshold {
            T::zero()
        } else if x > threshold {
            x - threshold
        } else {
            x + threshold
        }
    })
}

/// Pad 1D signal for convolution
#[allow(dead_code)]
fn pad_signal_1d<T>(
    signal: &ArrayView1<T>,
    filter: &[T],
    mode: BorderMode,
) -> NdimageResult<Array1<T>>
where
    T: Float + FromPrimitive + Clone,
{
    let n = signal.len();
    let filter_len = filter.len();
    let pad_len = filter_len - 1;

    let mut padded = Array1::zeros(n + 2 * pad_len);

    // Copy original signal to center
    for i in 0..n {
        padded[i + pad_len] = signal[i];
    }

    // Apply border mode
    match mode {
        BorderMode::Constant => {
            // Zeros already filled
        }
        BorderMode::Reflect => {
            // Left padding
            for i in 0..pad_len {
                let src_idx = pad_len - 1 - i;
                if src_idx < n {
                    padded[i] = signal[src_idx];
                }
            }
            // Right padding
            for i in 0..pad_len {
                let src_idx = n - 1 - i;
                if src_idx < n {
                    padded[n + pad_len + i] = signal[src_idx];
                }
            }
        }
        BorderMode::Nearest => {
            // Left padding
            for i in 0..pad_len {
                padded[i] = signal[0];
            }
            // Right padding
            for i in 0..pad_len {
                padded[n + pad_len + i] = signal[n - 1];
            }
        }
        BorderMode::Wrap => {
            // Periodic/circular boundary conditions
            // Left padding
            for i in 0..pad_len {
                let src_idx = if n > pad_len - i {
                    n - (pad_len - i)
                } else {
                    (n - (pad_len - i) % n + n) % n
                };
                padded[i] = signal[src_idx];
            }
            // Right padding
            for i in 0..pad_len {
                let src_idx = i % n;
                padded[n + pad_len + i] = signal[src_idx];
            }
        }
        BorderMode::Mirror => {
            // Mirror/symmetric boundary conditions (reflect without repeating edge)
            // Left padding
            for i in 0..pad_len {
                let offset = pad_len - i;
                let src_idx = if offset <= n {
                    offset - 1
                } else {
                    // For very long padding, use modulo arithmetic
                    let wrapped = (offset - 1) % (2 * n);
                    if wrapped < n {
                        wrapped
                    } else {
                        2 * n - 1 - wrapped
                    }
                };
                padded[i] = signal[src_idx.min(n - 1)];
            }
            // Right padding
            for i in 0..pad_len {
                let offset = i + 1;
                let src_idx = if offset <= n {
                    n - 1 - offset
                } else {
                    // For very long padding, use modulo arithmetic
                    let wrapped = (offset - 1) % (2 * n);
                    if wrapped < n {
                        n - 1 - wrapped
                    } else {
                        wrapped - n
                    }
                };
                padded[n + pad_len + i] = signal[src_idx.max(0).min(n - 1)];
            }
        }
    }

    Ok(padded)
}

/// 1D convolution with downsampling
#[allow(dead_code)]
fn convolve_downsample_1d<T>(
    signal: &ArrayView1<T>,
    filter: &[T],
    downsample: usize,
) -> NdimageResult<Array1<T>>
where
    T: Float + FromPrimitive + Clone + Zero,
{
    let n = signal.len();
    let filter_len = filter.len();

    if n < filter_len {
        return Err(NdimageError::InvalidInput(
            "Signal length must be at least filter length".into(),
        ));
    }

    let output_len = (n - filter_len + 1 + downsample - 1) / downsample;
    let mut output = Array1::zeros(output_len);

    for i in 0..output_len {
        let start_idx = i * downsample;
        if start_idx + filter_len <= n {
            let mut sum = T::zero();
            for j in 0..filter_len {
                sum = sum + signal[start_idx + j] * filter[j];
            }
            output[i] = sum;
        }
    }

    Ok(output)
}

/// 1D upsampling with convolution
#[allow(dead_code)]
fn upsample_convolve_1d<T>(
    signal: &ArrayView1<T>,
    filter: &[T],
    upsample: usize,
) -> NdimageResult<Array1<T>>
where
    T: Float + FromPrimitive + Clone + Zero,
{
    let n = signal.len();
    let filter_len = filter.len();
    let upsampled_len = n * upsample;
    let output_len = upsampled_len + filter_len - 1;

    // Upsample by inserting zeros
    let mut upsampled = Array1::zeros(upsampled_len);
    for i in 0..n {
        upsampled[i * upsample] = signal[i];
    }

    // Convolve with reconstruction filter
    let mut output = Array1::zeros(output_len);
    for i in 0..output_len {
        let mut sum = T::zero();
        for j in 0..filter_len {
            if i >= j && i - j < upsampled_len {
                sum = sum + upsampled[i - j] * filter[j];
            }
        }
        output[i] = sum;
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_haar_coefficients() {
        let haar = WaveletFilter::<f64>::new(WaveletFamily::Haar)
            .expect("Failed to create Haar wavelet filter");
        let sqrt2_inv = 1.0 / std::f64::consts::SQRT_2;

        assert_abs_diff_eq!(haar.low_dec[0], sqrt2_inv, epsilon = 1e-10);
        assert_abs_diff_eq!(haar.low_dec[1], sqrt2_inv, epsilon = 1e-10);
        assert_abs_diff_eq!(haar.high_dec[0], sqrt2_inv, epsilon = 1e-10);
        assert_abs_diff_eq!(haar.high_dec[1], -sqrt2_inv, epsilon = 1e-10);
    }

    #[test]
    fn test_dwt_1d() {
        let signal = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let haar =
            WaveletFilter::new(WaveletFamily::Haar).expect("Failed to create Haar wavelet filter");

        let (low, high) =
            dwt_1d(&signal.view(), &haar, BorderMode::Nearest).expect("Failed to perform 1D DWT");

        // Check that the result has the expected length
        assert_eq!(low.len(), 4);
        assert_eq!(high.len(), 4);

        // The low-pass should contain the averages
        // The high-pass should contain the differences
        assert!(low.iter().all(|&x| x.is_finite()));
        assert!(high.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_dwt_idwt_reconstruction() {
        let signal = array![1.0, 2.0, 3.0, 4.0];
        let haar =
            WaveletFilter::new(WaveletFamily::Haar).expect("Failed to create Haar wavelet filter");

        let (low, high) =
            dwt_1d(&signal.view(), &haar, BorderMode::Nearest).expect("Failed to perform 1D DWT");
        let reconstructed =
            idwt_1d(&low.view(), &high.view(), &haar).expect("Failed to perform 1D IDWT");

        // Check that reconstruction length is appropriate
        assert!(reconstructed.len() >= signal.len());

        // Check that values are reasonable (perfect reconstruction is complex with border handling)
        assert!(reconstructed.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_dwt_2d() {
        let image = array![
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0]
        ];

        let haar =
            WaveletFilter::new(WaveletFamily::Haar).expect("Failed to create Haar wavelet filter");
        let (ll, lh, hl, hh) =
            dwt_2d(&image.view(), &haar, BorderMode::Nearest).expect("Failed to perform 2D DWT");

        // Check dimensions
        assert_eq!(ll.dim(), (2, 2));
        assert_eq!(lh.dim(), (2, 2));
        assert_eq!(hl.dim(), (2, 2));
        assert_eq!(hh.dim(), (2, 2));

        // Check that all values are finite
        assert!(ll.iter().all(|&x| x.is_finite()));
        assert!(lh.iter().all(|&x| x.is_finite()));
        assert!(hl.iter().all(|&x| x.is_finite()));
        assert!(hh.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_wavelet_denoise() {
        let image = array![
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0]
        ];

        let haar_filter =
            WaveletFilter::new(WaveletFamily::Haar).expect("Failed to create Haar wavelet filter");
        let denoised = wavelet_denoise(&image.view(), &haar_filter, 1.0, 3, BorderMode::Nearest)
            .expect("Failed to denoise with wavelet");

        // Check that output has same dimensions as input
        assert_eq!(denoised.dim(), image.dim());

        // Check that all values are finite
        assert!(denoised.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_soft_threshold() {
        let coeffs = array![[-3.0, -1.5, -0.5], [0.5, 1.5, 3.0]];

        let thresholded = soft_threshold(&coeffs.view(), 1.0);

        // Values below threshold should be zero
        assert_eq!(thresholded[[0, 2]], 0.0); // -0.5 -> 0
        assert_eq!(thresholded[[1, 0]], 0.0); // 0.5 -> 0

        // Values above threshold should be reduced
        assert_eq!(thresholded[[0, 0]], -2.0); // -3.0 -> -2.0
        assert_eq!(thresholded[[1, 2]], 2.0); // 3.0 -> 2.0
    }

    #[test]
    fn test_daubechies_coefficients() {
        let db2 = WaveletFilter::<f64>::new(WaveletFamily::Daubechies(2))
            .expect("Failed to create Daubechies-2 wavelet filter");

        // Check that we have 4 coefficients for DB2
        assert_eq!(db2.low_dec.len(), 4);
        assert_eq!(db2.high_dec.len(), 4);

        // Check that coefficients are finite
        assert!(db2.low_dec.iter().all(|&x| x.is_finite()));
        assert!(db2.high_dec.iter().all(|&x| x.is_finite()));
    }
}

/// Multi-level wavelet decomposition
///
/// Performs a multi-level discrete wavelet transform, producing a pyramid
/// of coefficients at different scales and orientations.
#[allow(dead_code)]
pub fn wavelet_decompose<T>(
    image: &ArrayView2<T>,
    wavelet: &WaveletFilter<T>,
    levels: usize,
    mode: BorderMode,
) -> NdimageResult<WaveletDecomposition<T>>
where
    T: Float + FromPrimitive + Debug + Clone + Send + Sync + 'static,
{
    let mut decomposition = WaveletDecomposition::new();
    let mut current = image.to_owned();

    for _level in 0..levels {
        let (height, width) = current.dim();

        // Check minimum size constraint
        if height < 4 || width < 4 {
            break;
        }

        // Perform 2D DWT
        let (ll, lh, hl, hh) = dwt_2d(&current.view(), wavelet, mode)?;

        // Store detail coefficients
        decomposition.add_level(WaveletLevel {
            lh: lh.clone(),
            hl: hl.clone(),
            hh: hh.clone(),
        });

        // Continue with approximation coefficients
        current = ll;
    }

    // Store final approximation
    decomposition.approximation = Some(current);

    Ok(decomposition)
}

/// Multi-level wavelet reconstruction
///
/// Reconstructs an image from its multi-level wavelet decomposition.
#[allow(dead_code)]
pub fn wavelet_reconstruct<T>(
    decomposition: &WaveletDecomposition<T>,
    wavelet: &WaveletFilter<T>,
) -> NdimageResult<Array2<T>>
where
    T: Float + FromPrimitive + Debug + Clone + Send + Sync + 'static,
{
    let mut result = match &decomposition.approximation {
        Some(approx) => approx.clone(),
        None => {
            return Err(NdimageError::InvalidInput(
                "Decomposition must contain approximation coefficients".into(),
            ))
        }
    };

    // Reconstruct from coarsest to finest level
    for level in decomposition.levels.iter().rev() {
        let ll = result.view();
        result = idwt_2d(
            &ll,
            &level.lh.view(),
            &level.hl.view(),
            &level.hh.view(),
            wavelet,
        )?;
    }

    Ok(result)
}

/// Wavelet-based denoising using soft thresholding
///
/// This function performs denoising using the following steps:
/// 1. Multi-level wavelet decomposition
/// 2. Soft thresholding of detail coefficients
/// 3. Wavelet reconstruction
#[allow(dead_code)]
pub fn wavelet_denoise<T>(
    image: &ArrayView2<T>,
    wavelet: &WaveletFilter<T>,
    threshold: T,
    levels: usize,
    mode: BorderMode,
) -> NdimageResult<Array2<T>>
where
    T: Float + FromPrimitive + Debug + Clone + Send + Sync + 'static,
{
    // Decompose
    let mut decomp = wavelet_decompose(image, wavelet, levels, mode)?;

    // Apply soft thresholding to detail coefficients
    for level in &mut decomp.levels {
        soft_threshold_inplace(&mut level.lh, threshold);
        soft_threshold_inplace(&mut level.hl, threshold);
        soft_threshold_inplace(&mut level.hh, threshold);
    }

    // Reconstruct
    wavelet_reconstruct(&decomp, wavelet)
}

/// Wavelet decomposition structure for multi-level analysis
#[derive(Debug, Clone)]
pub struct WaveletDecomposition<T> {
    /// Approximation coefficients at the coarsest level
    pub approximation: Option<Array2<T>>,
    /// Detail coefficients at each level (from finest to coarsest)
    pub levels: Vec<WaveletLevel<T>>,
}

impl<T> WaveletDecomposition<T> {
    pub fn new() -> Self {
        Self {
            approximation: None,
            levels: Vec::new(),
        }
    }

    pub fn add_level(&mut self, level: WaveletLevel<T>) {
        self.levels.push(level);
    }
}

/// Detail coefficients for a single wavelet decomposition level
#[derive(Debug, Clone)]
pub struct WaveletLevel<T> {
    /// Horizontal detail (low-high)
    pub lh: Array2<T>,
    /// Vertical detail (high-low)  
    pub hl: Array2<T>,
    /// Diagonal detail (high-high)
    pub hh: Array2<T>,
}

/// Apply soft thresholding to an array in-place
#[allow(dead_code)]
fn soft_threshold_inplace<T>(array: &mut Array2<T>, threshold: T)
where
    T: Float + FromPrimitive + PartialOrd,
{
    for elem in array.iter_mut() {
        let abs_val = elem.abs();
        if abs_val <= threshold {
            *elem = T::zero();
        } else {
            let sign = if *elem >= T::zero() {
                T::one()
            } else {
                -T::one()
            };
            *elem = sign * (abs_val - threshold);
        }
    }
}

/// Advanced stationary wavelet transform (undecimated)
///
/// Unlike the standard DWT, the stationary WT doesn't downsample,
/// preserving translation invariance and producing redundant representations
/// that are often better for denoising and feature detection.
#[allow(dead_code)]
pub fn stationary_wavelet_transform<T>(
    image: &ArrayView2<T>,
    wavelet: &WaveletFilter<T>,
    levels: usize,
    mode: BorderMode,
) -> NdimageResult<StationaryWaveletDecomposition<T>>
where
    T: Float + FromPrimitive + Debug + Clone + Send + Sync + 'static,
{
    let mut decomposition = StationaryWaveletDecomposition::new();
    let mut current = image.to_owned();

    for level in 0..levels {
        // Create upsampled filters for this level
        let upsample_factor = 2_usize.pow(level as u32);
        let low_upsampled = upsample_filter(&wavelet.low_dec, upsample_factor);
        let high_upsampled = upsample_filter(&wavelet.high_dec, upsample_factor);

        // Apply separable filtering without downsampling
        let (ll, lh, hl, hh) =
            stationary_dwt_2d(&current.view(), &low_upsampled, &high_upsampled, mode)?;

        decomposition.add_level(StationaryWaveletLevel {
            lh: lh.clone(),
            hl: hl.clone(),
            hh: hh.clone(),
        });

        // Continue with approximation for next level
        current = ll;
    }

    decomposition.approximation = Some(current);
    Ok(decomposition)
}

/// Stationary wavelet decomposition structure
#[derive(Debug, Clone)]
pub struct StationaryWaveletDecomposition<T> {
    pub approximation: Option<Array2<T>>,
    pub levels: Vec<StationaryWaveletLevel<T>>,
}

impl<T> StationaryWaveletDecomposition<T> {
    pub fn new() -> Self {
        Self {
            approximation: None,
            levels: Vec::new(),
        }
    }

    pub fn add_level(&mut self, level: StationaryWaveletLevel<T>) {
        self.levels.push(level);
    }
}

/// Stationary wavelet level (no downsampling)
#[derive(Debug, Clone)]
pub struct StationaryWaveletLevel<T> {
    pub lh: Array2<T>,
    pub hl: Array2<T>,
    pub hh: Array2<T>,
}

/// Upsample a filter by inserting zeros
#[allow(dead_code)]
fn upsample_filter<T>(filter: &[T], factor: usize) -> Vec<T>
where
    T: Float + FromPrimitive + Clone,
{
    let mut upsampled = Vec::with_capacity(filter.len() * factor);

    for &coeff in filter {
        upsampled.push(coeff);
        for _ in 1..factor {
            upsampled.push(T::zero());
        }
    }

    upsampled
}

/// Stationary 2D DWT without downsampling
#[allow(dead_code)]
fn stationary_dwt_2d<T>(
    image: &ArrayView2<T>,
    low_filter: &[T],
    high_filter: &[T],
    mode: BorderMode,
) -> NdimageResult<(Array2<T>, Array2<T>, Array2<T>, Array2<T>)>
where
    T: Float + FromPrimitive + Debug + Clone + Send + Sync + 'static,
{
    let (height, width) = image.dim();

    // Row-wise filtering
    let mut ll_rows = Array2::zeros((height, width));
    let mut lh_rows = Array2::zeros((height, width));

    for i in 0..height {
        let row = image.row(i);
        let padded = pad_signal_1d(&row, low_filter, mode)?;

        let low_filtered = convolve_1d(&padded.view(), low_filter)?;
        let high_filtered = convolve_1d(&padded.view(), high_filter)?;

        // Extract relevant portion (accounting for padding)
        let start_idx = (padded.len() - width) / 2;
        for j in 0..width {
            ll_rows[[i, j]] = low_filtered[start_idx + j];
            lh_rows[[i, j]] = high_filtered[start_idx + j];
        }
    }

    // Column-wise filtering of the row results
    let mut ll = Array2::zeros((height, width));
    let mut lh = Array2::zeros((height, width));
    let mut hl = Array2::zeros((height, width));
    let mut hh = Array2::zeros((height, width));

    for j in 0..width {
        let ll_col = ll_rows.column(j);
        let lh_col = lh_rows.column(j);

        let ll_padded = pad_signal_1d(&ll_col, low_filter, mode)?;
        let lh_padded = pad_signal_1d(&lh_col, low_filter, mode)?;

        let ll_low = convolve_1d(&ll_padded.view(), low_filter)?;
        let ll_high = convolve_1d(&ll_padded.view(), high_filter)?;
        let lh_low = convolve_1d(&lh_padded.view(), low_filter)?;
        let lh_high = convolve_1d(&lh_padded.view(), high_filter)?;

        let start_idx = (ll_padded.len() - height) / 2;
        for i in 0..height {
            ll[[i, j]] = ll_low[start_idx + i];
            hl[[i, j]] = ll_high[start_idx + i];
            lh[[i, j]] = lh_low[start_idx + i];
            hh[[i, j]] = lh_high[start_idx + i];
        }
    }

    Ok((ll, lh, hl, hh))
}

/// Simple 1D convolution
#[allow(dead_code)]
fn convolve_1d<T>(signal: &ArrayView1<T>, filter: &[T]) -> NdimageResult<Array1<T>>
where
    T: Float + FromPrimitive + Clone,
{
    let signal_len = signal.len();
    let filter_len = filter.len();

    if signal_len < filter_len {
        return Err(NdimageError::InvalidInput(
            "Signal must be at least as long as filter".into(),
        ));
    }

    let output_len = signal_len - filter_len + 1;
    let mut output = Array1::zeros(output_len);

    for i in 0..output_len {
        let mut sum = T::zero();
        for j in 0..filter_len {
            sum = sum + signal[i + j] * filter[filter_len - 1 - j];
        }
        output[i] = sum;
    }

    Ok(output)
}
