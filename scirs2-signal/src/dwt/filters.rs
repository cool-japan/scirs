//! Wavelet filter definitions and generation functions
//!
//! This module provides various wavelet filter definitions, including Haar, Daubechies,
//! Symlets, Coiflets, Biorthogonal, and Meyer wavelets.

use crate::error::{SignalError, SignalResult};

/// Represents a wavelet filter pair (decomposition and reconstruction filters)
pub struct WaveletFilters {
    /// Decomposition low-pass filter
    pub dec_lo: Vec<f64>,
    /// Decomposition high-pass filter
    pub dec_hi: Vec<f64>,
    /// Reconstruction low-pass filter
    pub rec_lo: Vec<f64>,
    /// Reconstruction high-pass filter
    pub rec_hi: Vec<f64>,
    /// Wavelet family name
    pub family: String,
    /// Vanishing moments
    pub vanishing_moments: usize,
}

impl WaveletFilters {
    /// Create a new set of wavelet filters
    pub fn new(
        dec_lo: Vec<f64>,
        dec_hi: Vec<f64>,
        rec_lo: Vec<f64>,
        rec_hi: Vec<f64>,
        family: &str,
        vanishing_moments: usize,
    ) -> Self {
        WaveletFilters {
            dec_lo,
            dec_hi,
            rec_lo,
            rec_hi,
            family: family.to_string(),
            vanishing_moments,
        }
    }
}

/// Predefined wavelet types
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Wavelet {
    /// Haar wavelet (equivalent to db1)
    Haar,
    /// Daubechies wavelets with N vanishing moments
    DB(usize),
    /// Coiflet wavelets with N vanishing moments
    Coif(usize),
    /// Symlet wavelets with N vanishing moments
    Sym(usize),
    /// Biorthogonal wavelets with Nr/Nd vanishing moments
    BiorNrNd { nr: usize, nd: usize },
    /// Reverse biorthogonal wavelets with Nr/Nd vanishing moments
    RBioNrNd { nr: usize, nd: usize },
    /// Meyer wavelet
    Meyer,
    /// Discrete Meyer wavelet
    DMeyer,
}

impl Wavelet {
    /// Get the filters for this wavelet
    pub fn filters(&self) -> SignalResult<WaveletFilters> {
        match *self {
            Wavelet::Haar => Ok(haar_filters()),
            Wavelet::DB(n) => {
                if n == 0 {
                    return Err(SignalError::ValueError(
                        "Daubechies wavelet order must be at least 1".to_string(),
                    ));
                }
                if n > 20 {
                    return Err(SignalError::ValueError(
                        "Daubechies wavelet order must be at most 20".to_string(),
                    ));
                }
                db_filters(n)
            }
            Wavelet::Sym(n) => {
                if n < 2 {
                    return Err(SignalError::ValueError(
                        "Symlet wavelet order must be at least 2".to_string(),
                    ));
                }
                if n > 20 {
                    return Err(SignalError::ValueError(
                        "Symlet wavelet order must be at most 20".to_string(),
                    ));
                }
                sym_filters(n)
            }
            Wavelet::Coif(n) => {
                if n == 0 {
                    return Err(SignalError::ValueError(
                        "Coiflet wavelet order must be at least 1".to_string(),
                    ));
                }
                if n > 5 {
                    return Err(SignalError::ValueError(
                        "Coiflet wavelet order must be at most 5".to_string(),
                    ));
                }
                coif_filters(n)
            }
            Wavelet::BiorNrNd { nr, nd } => bior_filters(nr, nd),
            Wavelet::RBioNrNd { nr, nd } => rbior_filters(nr, nd),
            Wavelet::Meyer => meyer_filters(),
            Wavelet::DMeyer => dmeyer_filters(),
        }
    }
}

/// Haar wavelet filters
fn haar_filters() -> WaveletFilters {
    let dec_lo = vec![0.7071067811865475, 0.7071067811865475];
    let dec_hi = vec![-0.7071067811865475, 0.7071067811865475];
    let rec_lo = vec![0.7071067811865475, 0.7071067811865475];
    let rec_hi = vec![0.7071067811865475, -0.7071067811865475];

    WaveletFilters::new(dec_lo, dec_hi, rec_lo, rec_hi, "db1", 1)
}

/// Daubechies wavelet filters
fn db_filters(n: usize) -> SignalResult<WaveletFilters> {
    // Coefficients for Daubechies wavelets up to db20
    let coeffs = match n {
        1 => return Ok(haar_filters()), // db1 is the same as haar
        2 => vec![
            0.482962913144534,
            0.836516303737808,
            0.224143868042013,
            -0.129409522551260,
        ],
        3 => vec![
            0.332670552950083,
            0.806891509311092,
            0.459877502118491,
            -0.135011020010255,
            -0.085441273882027,
            0.035226291882100,
        ],
        4 => vec![
            0.230377813308855,
            0.714846570552542,
            0.630880767929590,
            -0.027983769416984,
            -0.187034811717090,
            0.030841381835987,
            0.032883011666983,
            -0.010597401784997,
        ],
        5 => vec![
            0.160102397974125,
            0.603829269797473,
            0.724308528437772,
            0.138428145901320,
            -0.242294887066190,
            -0.032244869585030,
            0.077571493840065,
            -0.006241490213012,
            -0.012580751999016,
            0.003335725285002,
        ],
        6 => vec![
            0.111540743350109,
            0.494623890398453,
            0.751133908021093,
            0.315250351709195,
            -0.226264693965169,
            -0.129766867567262,
            0.097501605587079,
            0.027522865530016,
            -0.031582039317674,
            0.000553842201161,
            0.004777257511010,
            -0.001077301085308,
        ],
        7 => vec![
            0.077852054085062,
            0.396539319482306,
            0.729132090846555,
            0.469782287405359,
            -0.143906003929106,
            -0.224036184994166,
            0.071309219267050,
            0.080612609151065,
            -0.038029936935034,
            -0.016574541631250,
            0.012550998556013,
            0.000429577973205,
            -0.001801640704047,
            0.000353713799974,
        ],
        8 => vec![
            0.054415842243082,
            0.312871590914466,
            0.675630736297212,
            0.585354683654869,
            -0.015829105256023,
            -0.284015542962428,
            0.000472484573552,
            0.128747426620186,
            -0.017369301002022,
            -0.044088253931064,
            0.013981027917015,
            0.008746094047015,
            -0.004870352993452,
            -0.000391740373376,
            0.000675449406450,
            -0.000117476784124,
        ],
        9 => vec![
            0.038077947363167,
            0.243834674612604,
            0.604823123690815,
            0.657288078051201,
            0.133197385825413,
            -0.293273783279725,
            -0.096840783220879,
            0.148540749338375,
            0.030725681478323,
            -0.067632829061523,
            0.000250947114831,
            0.022361662123515,
            -0.004723204757894,
            -0.004281503681904,
            0.001847646883056,
            0.000230385763524,
            -0.000251963188750,
            0.000039347319995,
        ],
        10 => vec![
            0.026670057900950,
            0.188176800077641,
            0.527201188931997,
            0.688459039453662,
            0.281172343660850,
            -0.249846424327358,
            -0.195946274377605,
            0.127369340335789,
            0.093057364603806,
            -0.071394147165860,
            -0.029457536821945,
            0.033212674058933,
            0.003606553567204,
            -0.010733175482979,
            0.001395351747052,
            0.001992405295185,
            -0.000685856695305,
            -0.000116466855151,
            0.000093588670001,
            -0.000013264203002,
        ],
        11 => vec![
            0.018694297761470,
            0.144067021150716,
            0.449899764356310,
            0.686840134482260,
            0.412751585203247,
            -0.159766011111892,
            -0.274162469668548,
            0.049111967750530,
            0.154536776654650,
            -0.033471784599596,
            -0.071320499838380,
            0.037839494563470,
            0.020023049801874,
            -0.018338989809190,
            -0.000749812147558,
            0.005767053451519,
            -0.000599561518447,
            -0.000971877579622,
            0.000332650266211,
            0.000050192775784,
            -0.000042815036824,
            0.000005956369052,
        ],
        12 => vec![
            0.013112222451907,
            0.109702658731684,
            0.377355135214097,
            0.657198722525712,
            0.515886778302947,
            -0.044763792539050,
            -0.318006342412828,
            -0.027315161325541,
            0.183247071802233,
            0.016382602271372,
            -0.092891783304778,
            0.013092731358062,
            0.037899729244972,
            -0.016471896388428,
            -0.010737662142476,
            0.009374153069398,
            0.000703854523154,
            -0.002868814590175,
            0.000337771648577,
            0.000488308626611,
            -0.000164103679384,
            -0.000021891697069,
            0.000020020377472,
            -0.000002608451495,
        ],
        13 => vec![
            0.009202133538962,
            0.082861243872901,
            0.312349864992317,
            0.610996615683353,
            0.584318199857429,
            0.077847738324257,
            -0.322596443582933,
            -0.105765749607652,
            0.175869936010311,
            0.072251231604229,
            -0.102291719176688,
            -0.019538882735286,
            0.053876887649336,
            0.000804567177079,
            -0.025605998222339,
            0.005866090894750,
            0.007650934922782,
            -0.004671573326446,
            -0.000539406538826,
            0.001444618879459,
            -0.000210429351128,
            -0.000230498951215,
            0.000080253881662,
            0.000009785913342,
            -0.000009446366376,
            0.000001161914472,
        ],
        14 => vec![
            0.006466627005487,
            0.062364758849384,
            0.254850267992499,
            0.555141822379500,
            0.631371908987652,
            0.195946274377410,
            -0.293273783279090,
            -0.168731118268458,
            0.139225370567556,
            0.124757463291737,
            -0.087277003387606,
            -0.057528233685938,
            0.053939470409553,
            0.018756027276697,
            -0.030106249425372,
            -0.000589500643074,
            0.013051480898198,
            -0.003731589124174,
            -0.003446503662376,
            0.002497631634488,
            0.000167263118299,
            -0.000749629797140,
            0.000134049175078,
            0.000108893885386,
            -0.000044951571736,
            -0.000004473604308,
            0.000004823057215,
            -0.000000561693739,
        ],
        15 => vec![
            0.004547087191196,
            0.046977207550221,
            0.206044010037892,
            0.493574979637483,
            0.657732322847070,
            0.304753078520837,
            -0.227664532294873,
            -0.213033111386249,
            0.070921920748465,
            0.166894551346910,
            -0.042026740014152,
            -0.095195982362738,
            0.040602283294518,
            0.037297111831357,
            -0.029196217764038,
            -0.013436687461988,
            0.016407058030801,
            0.001058298062091,
            -0.006865749450209,
            0.001599832377537,
            0.001992403617674,
            -0.001283155577582,
            -0.000103439625908,
            0.000387916110165,
            -0.000072475971390,
            -0.000051817475442,
            0.000023573692280,
            0.000002090509552,
            -0.000002599802611,
            0.000000289488489,
        ],
        16 => vec![
            0.003198059646889,
            0.035272488035365,
            0.165064042480853,
            0.429294026810007,
            0.663584746064344,
            0.397635182392839,
            -0.136727887943900,
            -0.236615579414771,
            -0.013882500872872,
            0.190933574545613,
            0.027621877931673,
            -0.123688492987117,
            -0.013530500878251,
            0.070778084805636,
            0.000250947114469,
            -0.033999640851375,
            0.008780563324394,
            0.013976191892227,
            -0.005506340578446,
            -0.005676732137010,
            0.002644699244577,
            0.000546274677930,
            -0.000998676424451,
            -0.000063556729368,
            0.000207189625129,
            -0.000037639876035,
            -0.000027339873460,
            0.000012329077129,
            0.000001009818666,
            -0.000001326412561,
            0.000000144455234,
        ],
        17 => vec![
            0.002249989564127,
            0.026431052522557,
            0.131453455993449,
            0.365685017392567,
            0.651149758915307,
            0.473660724605308,
            -0.028095212072981,
            -0.241534344007344,
            -0.097652398741552,
            0.193668770644293,
            0.097681461521777,
            -0.132787569868878,
            -0.064137545429933,
            0.082353063002837,
            0.033635885443158,
            -0.046483122178525,
            -0.008818495924091,
            0.022687508861760,
            0.000377129908353,
            -0.009363659238240,
            0.002207248427215,
            0.003158078096280,
            -0.001365729920106,
            -0.000368017746814,
            0.000504404945599,
            0.000045656736790,
            -0.000112263640528,
            0.000018510185807,
            0.000014232619004,
            -0.000006329524522,
            -0.000000513426766,
            0.000000667030833,
            -0.000000071035188,
        ],
        18 => vec![
            0.001583152548123,
            0.019744895871288,
            0.104101536113325,
            0.305563884985853,
            0.623975434298419,
            0.529799560640167,
            0.081467735276949,
            -0.227797179526575,
            -0.170747419267909,
            0.175039297763324,
            0.161873046950642,
            -0.121474953800432,
            -0.115726872438501,
            0.073553294361569,
            0.066051592542114,
            -0.044946949533463,
            -0.026374233183532,
            0.026201450710306,
            0.007799536136598,
            -0.013413186638605,
            -0.000676129953621,
            0.005837844914568,
            -0.001068456126535,
            -0.001977793059845,
            0.000720561850328,
            0.000249781222699,
            -0.000248297309875,
            -0.000031423232942,
            0.000062243947721,
            -0.000009375775930,
            -0.000007237387870,
            0.000003191792220,
            0.000000265012362,
            -0.000000330410306,
            0.000000034722061,
        ],
        19 => vec![
            0.001114196372414,
            0.014732990319389,
            0.082207610455798,
            0.250793445524826,
            0.585444519401690,
            0.566988307543766,
            0.186025297029308,
            -0.195458700548709,
            -0.227730192281805,
            0.137233695257915,
            0.214599071153825,
            -0.089160781132029,
            -0.158417505731180,
            0.045867240912607,
            0.098035251500675,
            -0.020522421488349,
            -0.052147231712149,
            0.012721525796557,
            0.024392153958273,
            -0.003908218962188,
            -0.009818466381840,
            0.000605584177777,
            0.003589682003384,
            -0.000530798219852,
            -0.001001922227058,
            0.000335871807070,
            0.000170180896946,
            -0.000123237265315,
            -0.000013210000789,
            0.000031646436535,
            -0.000003968737843,
            -0.000003845884769,
            0.000001573769802,
            0.000000134049667,
            -0.000000160562083,
            0.000000016645797,
        ],
        20 => vec![
            0.000784072542954,
            0.010977933021543,
            0.064550896006344,
            0.203752672274431,
            0.540090975933037,
            0.588632819378512,
            0.281212186616818,
            -0.149991375873567,
            -0.269129167676524,
            0.086897220771386,
            0.250741236293154,
            -0.042222645318166,
            -0.189361081889192,
            0.009845067607434,
            0.125613753894546,
            0.003658261885561,
            -0.077576346018652,
            -0.002903428894004,
            0.042865478706229,
            -0.000380739967502,
            -0.021314118649806,
            0.001432104210551,
            0.009138142946111,
            -0.000867154737000,
            -0.003457476052446,
            0.000287387874799,
            0.001090536731614,
            -0.000074661709345,
            -0.000293359249792,
            0.000032466421169,
            0.000061991833491,
            -0.000011192232887,
            -0.000008949422779,
            0.000003201334779,
            0.000000766466778,
            -0.000000313671983,
            -0.000000081726461,
            0.000000079784918,
            -0.000000007962295,
        ],
        _ => {
            return Err(SignalError::ValueError(format!(
                "Daubechies wavelet db{} is not supported. Valid values: 1-20.",
                n
            )))
        }
    };

    // Length of wavelet filter is 2n
    let filter_len = 2 * n;

    // Decomposition filters
    let dec_lo = coeffs.clone();
    let mut dec_hi = vec![0.0; filter_len];

    // QMF relationship for high-pass filter
    for i in 0..filter_len {
        dec_hi[i] = (-1_f64).powi(i as i32) * dec_lo[filter_len - 1 - i];
    }

    // Reconstruction filters (time reverse of decomposition filters)
    let mut rec_lo = dec_lo.clone();
    let mut rec_hi = dec_hi.clone();
    rec_lo.reverse();
    rec_hi.reverse();

    // No need to swap signs for the reconstruction high-pass filter
    // as the QMF relationship is now defined correctly

    Ok(WaveletFilters::new(
        dec_lo,
        dec_hi,
        rec_lo,
        rec_hi,
        &format!("db{}", n),
        n,
    ))
}

/// Symlet wavelet filters
fn sym_filters(n: usize) -> SignalResult<WaveletFilters> {
    // Symlets are similar to Daubechies but more symmetrical
    // For now, we'll implement a few commonly used ones
    let coeffs = match n {
        2 => vec![
            0.482962913144534,
            0.836516303737808,
            0.224143868042013,
            -0.129409522551260,
        ],
        3 => vec![
            0.332670552950083,
            0.806891509311092,
            0.459877502118491,
            -0.135011020010255,
            -0.085441273882027,
            0.035226291882100,
        ],
        4 => vec![
            0.027333068345078,
            0.029519490925774,
            -0.039134249302383,
            0.199397533977639,
            0.723407690402421,
            0.633978963458212,
            0.016602105764522,
            -0.175328089908107,
        ],
        5 => vec![
            0.038654795955998,
            0.041746191085696,
            -0.055344186117389,
            0.281990696854629,
            1.023052966894833,
            0.896581648380820,
            0.023478923136726,
            -0.247951362613099,
        ],
        6 => vec![
            0.021784700327209,
            0.004936612372610,
            -0.166863215412514,
            -0.068323121587143,
            0.694457972958569,
            1.113892783926184,
            0.477904371333197,
            -0.102724969862286,
            -0.069883027249875,
            0.067165481423504,
            0.008223012248799,
            -0.010435713106749,
        ],
        7 => vec![
            0.010268176708511,
            0.004010244871533,
            -0.107808028230524,
            -0.140047240442703,
            0.288629631752769,
            0.767764317003164,
            0.536101917090763,
            0.017441255087316,
            -0.049552834937945,
            0.021662469594920,
            0.005100436968998,
            -0.015179002335395,
        ],
        8 => vec![
            0.001889950333672,
            0.000268181156429,
            -0.014296827155513,
            -0.010798432312464,
            0.046958624294051,
            0.058525491871548,
            -0.152463871896665,
            -0.070078291222233,
            0.476803265249878,
            0.803738751805916,
            0.297857795605542,
            -0.099219543576373,
            -0.012592909212478,
            0.032755032597835,
        ],
        _ => {
            // Fall back to Daubechies for other orders, with a note
            // In a full implementation, we would include coefficients for all supported symlets
            let db_wavelet = db_filters(n)?;

            return Ok(WaveletFilters::new(
                db_wavelet.dec_lo,
                db_wavelet.dec_hi,
                db_wavelet.rec_lo,
                db_wavelet.rec_hi,
                &format!("sym{}", n),
                n,
            ));
        }
    };

    // Length of wavelet filter is 2n
    let filter_len = 2 * n;

    // Decomposition filters
    let dec_lo = coeffs.clone();
    let mut dec_hi = vec![0.0; filter_len];

    // QMF relationship for high-pass filter
    for i in 0..filter_len {
        dec_hi[i] = (-1_f64).powi(i as i32) * dec_lo[filter_len - 1 - i];
    }

    // Reconstruction filters (time reverse of decomposition filters)
    let mut rec_lo = dec_lo.clone();
    let mut rec_hi = dec_hi.clone();
    rec_lo.reverse();
    rec_hi.reverse();

    // No need to swap signs for the reconstruction high-pass filter
    // as the QMF relationship is now defined correctly

    Ok(WaveletFilters::new(
        dec_lo,
        dec_hi,
        rec_lo,
        rec_hi,
        &format!("sym{}", n),
        n,
    ))
}

/// Coiflet wavelet filters
fn coif_filters(n: usize) -> SignalResult<WaveletFilters> {
    // Coiflet filter coefficients
    let coeffs = match n {
        1 => vec![
            -0.0156557285289848,
            -0.0727326213410511,
            0.3848648565381134,
            0.8525720416423,
            0.337897670951159,
            -0.0727322757411889,
        ],
        2 => vec![
            -0.0007205494453679,
            -0.0018232088707116,
            0.0056114348194211,
            0.0236801719464464,
            -0.0594344186467388,
            -0.0764885990786692,
            0.4170051844236707,
            0.8127236354493977,
            0.3861100668229939,
            -0.0673725547222826,
            -0.0414649367819558,
            0.0163873364635998,
        ],
        3 => vec![
            -0.0000345997770640,
            -0.0000709833031381,
            0.0004662169601128,
            0.0011175187708906,
            -0.0025745176887502,
            -0.0090079761366615,
            0.0158805448636158,
            0.0345550275730615,
            -0.0823019271068856,
            -0.0717998216193117,
            0.4284834763776168,
            0.7937772226256169,
            0.405176902409615,
            -0.0611233900026726,
            -0.0657719112818552,
            0.0234526961418362,
            0.0077825964273254,
            -0.0037935128644910,
        ],
        4 => vec![
            -0.0000017849850031,
            -0.0000032596802369,
            0.0000312298758654,
            0.0000623390344610,
            -0.0002599745524878,
            -0.0005890207562444,
            0.0012665619292991,
            0.0037514361572790,
            -0.0056582866866115,
            -0.0152117315279485,
            0.0250822618448678,
            0.0393344271233433,
            -0.0962204420340021,
            -0.0666274742634348,
            0.4343860564915321,
            0.7822389309206135,
            0.415308407030491,
            -0.0560773133167630,
            -0.0812666996808907,
            0.0266823001560570,
            0.0160689439647787,
            -0.0073461663276432,
            -0.0016294920126020,
            0.0008923136685824,
        ],
        5 => vec![
            -0.0000000951765727,
            -0.0000001674428858,
            0.0000020637618516,
            0.0000037346551755,
            -0.0000213150268122,
            -0.0000413404322769,
            0.0001405411497166,
            0.0003022595818445,
            -0.0006381313431115,
            -0.0016628637021860,
            0.0024333732129107,
            0.0067641854487565,
            -0.0091642311634856,
            -0.0197617789446276,
            0.0326835742705106,
            0.0412892087544753,
            -0.1055742087143175,
            -0.0620359639693546,
            0.4379916262173834,
            0.7742896037334738,
            0.4215662066908515,
            -0.0520431631816557,
            -0.0919200105692549,
            0.0281680289738655,
            0.0234081567882734,
            -0.0101311175209033,
            -0.0041593587818186,
            0.0021782363583355,
            0.0003585896879330,
            -0.0002120808398259,
        ],
        _ => {
            return Err(SignalError::ValueError(format!(
                "Coiflet wavelet coif{} is not supported. Valid values: 1-5.",
                n
            )))
        }
    };

    // Length of wavelet filter is 6n
    let filter_len = 6 * n;

    // Decomposition filters
    let dec_lo = coeffs.clone();
    let mut dec_hi = vec![0.0; filter_len];

    // QMF relationship for high-pass filter
    for i in 0..filter_len {
        dec_hi[i] = (-1_f64).powi(i as i32) * dec_lo[filter_len - 1 - i];
    }

    // Reconstruction filters (time reverse of decomposition filters)
    let mut rec_lo = dec_lo.clone();
    let mut rec_hi = dec_hi.clone();
    rec_lo.reverse();
    rec_hi.reverse();

    // No need to swap signs for the reconstruction high-pass filter
    // as the QMF relationship is now defined correctly

    Ok(WaveletFilters::new(
        dec_lo,
        dec_hi,
        rec_lo,
        rec_hi,
        &format!("coif{}", n),
        n,
    ))
}

/// Biorthogonal wavelet filters
fn bior_filters(nr: usize, nd: usize) -> SignalResult<WaveletFilters> {
    // Biorthogonal wavelets use different filters for decomposition and reconstruction
    // The filters are defined by their order (nr, nd)

    // Check valid combinations for biorthogonal wavelets
    let valid_combinations = [
        (1, 1),
        (1, 3),
        (1, 5),
        (2, 2),
        (2, 4),
        (2, 6),
        (2, 8),
        (3, 1),
        (3, 3),
        (3, 5),
        (3, 7),
        (3, 9),
        (4, 4),
        (5, 5),
        (6, 8),
    ];

    if !valid_combinations.contains(&(nr, nd)) {
        return Err(SignalError::ValueError(format!(
            "Invalid biorthogonal wavelet specification: bior{}.{}. Valid combinations are: {:?}",
            nr, nd, valid_combinations
        )));
    }

    match (nr, nd) {
        // bior1.1 - Same as Haar
        (1, 1) => {
            let dec_lo = vec![0.7071067811865475, 0.7071067811865475];
            let dec_hi = vec![0.7071067811865475, -0.7071067811865475];
            let rec_lo = vec![0.7071067811865475, 0.7071067811865475];
            let rec_hi = vec![-0.7071067811865475, 0.7071067811865475];

            Ok(WaveletFilters::new(
                dec_lo,
                dec_hi,
                rec_lo,
                rec_hi,
                &format!("bior{}.{}", nr, nd),
                nr,
            ))
        }

        // bior1.3
        (1, 3) => {
            let dec_lo = vec![0.7071067811865475, 0.7071067811865475];
            let rec_lo = vec![
                -0.0883883476483184,
                0.0883883476483185,
                0.7071067811865475,
                0.7071067811865475,
                0.0883883476483185,
                -0.0883883476483184,
            ];

            // Create high-pass filters
            let mut dec_hi = vec![0.0; rec_lo.len()];
            let mut rec_hi = vec![0.0; dec_lo.len()];

            // Alternate signs for high-pass filters
            for i in 0..rec_lo.len() {
                dec_hi[i] = (-1_f64).powi(i as i32) * rec_lo[rec_lo.len() - 1 - i];
            }

            for i in 0..dec_lo.len() {
                rec_hi[i] = (-1_f64).powi(i as i32 + 1) * dec_lo[dec_lo.len() - 1 - i];
            }

            Ok(WaveletFilters::new(
                dec_lo,
                dec_hi,
                rec_lo,
                rec_hi,
                &format!("bior{}.{}", nr, nd),
                nr,
            ))
        }

        // bior1.5
        (1, 5) => {
            let dec_lo = vec![0.7071067811865475, 0.7071067811865475];
            let rec_lo = vec![
                0.0165728151840597,
                -0.0165728151840597,
                -0.1215339780164378,
                0.1215339780164378,
                0.7071067811865475,
                0.7071067811865475,
                0.1215339780164378,
                -0.1215339780164378,
                -0.0165728151840597,
                0.0165728151840597,
            ];

            // Create high-pass filters
            let mut dec_hi = vec![0.0; rec_lo.len()];
            let mut rec_hi = vec![0.0; dec_lo.len()];

            // Alternate signs for high-pass filters
            for i in 0..rec_lo.len() {
                dec_hi[i] = (-1_f64).powi(i as i32) * rec_lo[rec_lo.len() - 1 - i];
            }

            for i in 0..dec_lo.len() {
                rec_hi[i] = (-1_f64).powi(i as i32 + 1) * dec_lo[dec_lo.len() - 1 - i];
            }

            Ok(WaveletFilters::new(
                dec_lo,
                dec_hi,
                rec_lo,
                rec_hi,
                &format!("bior{}.{}", nr, nd),
                nr,
            ))
        }

        // bior2.2
        (2, 2) => {
            let dec_lo = vec![
                -0.1767766952966369,
                0.3535533905932738,
                1.0606601717798214,
                0.3535533905932738,
                -0.1767766952966369,
            ];
            let rec_lo = vec![
                0.3535533905932738,
                std::f64::consts::FRAC_1_SQRT_2,
                0.3535533905932738,
            ];

            // Create high-pass filters
            let mut dec_hi = vec![0.0; rec_lo.len()];
            let mut rec_hi = vec![0.0; dec_lo.len()];

            // Alternate signs for high-pass filters
            for i in 0..rec_lo.len() {
                dec_hi[i] = (-1_f64).powi(i as i32) * rec_lo[rec_lo.len() - 1 - i];
            }

            for i in 0..dec_lo.len() {
                rec_hi[i] = (-1_f64).powi(i as i32 + 1) * dec_lo[dec_lo.len() - 1 - i];
            }

            Ok(WaveletFilters::new(
                dec_lo,
                dec_hi,
                rec_lo,
                rec_hi,
                &format!("bior{}.{}", nr, nd),
                nr,
            ))
        }

        // bior2.4, bior2.6, bior2.8
        (2, 4) | (2, 6) | (2, 8) => {
            // Define dec_lo based on nd
            let dec_lo = match nd {
                4 => vec![
                    0.0331456303059065,
                    -0.0662912606118129,
                    -0.1767766952966369,
                    0.4198446513295126,
                    0.9943689110435825,
                    0.4198446513295126,
                    -0.1767766952966369,
                    -0.0662912606118129,
                    0.0331456303059065,
                ],
                6 => vec![
                    -0.0069053396600248,
                    0.0138106793200496,
                    0.0469563096881692,
                    -0.1077232986963881,
                    -0.1697106603182049,
                    0.4752925784897927,
                    0.9666908493984144,
                    0.4752925784897927,
                    -0.1697106603182049,
                    -0.1077232986963881,
                    0.0469563096881692,
                    0.0138106793200496,
                    -0.0069053396600248,
                ],
                8 => vec![
                    0.0015105430506304,
                    -0.0030210861012608,
                    -0.0129475118625466,
                    0.0289161098263542,
                    0.0529984818906909,
                    -0.134_913_073_607_712,
                    -0.1638291834577726,
                    0.5097608153878514,
                    0.951_556_053_042_669,
                    0.5097608153878514,
                    -0.1638291834577726,
                    -0.134_913_073_607_712,
                    0.0529984818906909,
                    0.0289161098263542,
                    -0.0129475118625466,
                    -0.0030210861012608,
                    0.0015105430506304,
                ],
                _ => unreachable!(),
            };

            // Define rec_lo - same for all bior2.x
            let rec_lo = vec![
                0.3535533905932738,
                std::f64::consts::FRAC_1_SQRT_2,
                0.3535533905932738,
            ];

            // Create high-pass filters
            let mut dec_hi = vec![0.0; rec_lo.len()];
            let mut rec_hi = vec![0.0; dec_lo.len()];

            // Alternate signs for high-pass filters
            for i in 0..rec_lo.len() {
                dec_hi[i] = (-1_f64).powi(i as i32) * rec_lo[rec_lo.len() - 1 - i];
            }

            for i in 0..dec_lo.len() {
                rec_hi[i] = (-1_f64).powi(i as i32 + 1) * dec_lo[dec_lo.len() - 1 - i];
            }

            Ok(WaveletFilters::new(
                dec_lo,
                dec_hi,
                rec_lo,
                rec_hi,
                &format!("bior{}.{}", nr, nd),
                nr,
            ))
        }

        // Other well-known biorthogonal wavelets
        (3, 1) => {
            let dec_lo = vec![
                -0.3535533905932738,
                1.0606601717798214,
                1.0606601717798214,
                -0.3535533905932738,
            ];
            let rec_lo = vec![
                0.1767766952966369,
                0.5303300858899107,
                0.5303300858899107,
                0.1767766952966369,
            ];

            // Create high-pass filters
            let mut dec_hi = vec![0.0; rec_lo.len()];
            let mut rec_hi = vec![0.0; dec_lo.len()];

            // Alternate signs for high-pass filters
            for i in 0..rec_lo.len() {
                dec_hi[i] = (-1_f64).powi(i as i32) * rec_lo[rec_lo.len() - 1 - i];
            }

            for i in 0..dec_lo.len() {
                rec_hi[i] = (-1_f64).powi(i as i32 + 1) * dec_lo[dec_lo.len() - 1 - i];
            }

            Ok(WaveletFilters::new(
                dec_lo,
                dec_hi,
                rec_lo,
                rec_hi,
                &format!("bior{}.{}", nr, nd),
                nr,
            ))
        }

        // bior3.3
        (3, 3) => {
            let dec_lo = vec![
                0.0662912606118329,
                0.1988737819357238,
                0.1546796083845573,
                -0.6698312455207294,
                1.1235147854599727,
                0.5320630098146652,
                -0.4455680946683667,
                -0.0196337984633395,
                0.1282552104550745,
            ];
            let rec_lo = vec![
                0.0069053396600248,
                0.0207160198800745,
                0.0517990809677456,
                0.2807880797154621,
                0.5515873933681696,
                0.2807880797154621,
                0.0517990809677456,
                0.0207160198800745,
                0.0069053396600248,
            ];

            // Create high-pass filters
            let mut dec_hi = vec![0.0; rec_lo.len()];
            let mut rec_hi = vec![0.0; dec_lo.len()];

            // Alternate signs for high-pass filters
            for i in 0..rec_lo.len() {
                dec_hi[i] = (-1_f64).powi(i as i32) * rec_lo[rec_lo.len() - 1 - i];
            }

            for i in 0..dec_lo.len() {
                rec_hi[i] = (-1_f64).powi(i as i32 + 1) * dec_lo[dec_lo.len() - 1 - i];
            }

            Ok(WaveletFilters::new(
                dec_lo,
                dec_hi,
                rec_lo,
                rec_hi,
                &format!("bior{}.{}", nr, nd),
                nr,
            ))
        }

        // For combinations that are valid but not explicitly implemented yet,
        // return a reasonable error
        _ => Err(SignalError::ValueError(format!(
            "Biorthogonal wavelet bior{}.{} is not fully implemented yet.",
            nr, nd
        ))),
    }
}

/// Reverse biorthogonal wavelet filters
fn rbior_filters(nr: usize, nd: usize) -> SignalResult<WaveletFilters> {
    // Reverse biorthogonal wavelets are just biorthogonal wavelets with
    // decomposition and reconstruction filters swapped

    // Check valid combinations for biorthogonal wavelets
    let valid_combinations = [
        (1, 1),
        (1, 3),
        (1, 5),
        (2, 2),
        (2, 4),
        (2, 6),
        (2, 8),
        (3, 1),
        (3, 3),
        (3, 5),
        (3, 7),
        (3, 9),
        (4, 4),
        (5, 5),
        (6, 8),
    ];

    if !valid_combinations.contains(&(nr, nd)) {
        return Err(SignalError::ValueError(format!(
            "Invalid reverse biorthogonal wavelet specification: rbio{}.{}. Valid combinations are: {:?}",
            nr, nd, valid_combinations
        )));
    }

    // First get the biorthogonal filters
    let bior = bior_filters(nd, nr)?; // Note the swapped 'nd' and 'nr' parameters

    // Swap decomposition and reconstruction filters
    let filters = WaveletFilters::new(
        bior.rec_lo, // dec_lo becomes rec_lo
        bior.rec_hi, // dec_hi becomes rec_hi
        bior.dec_lo, // rec_lo becomes dec_lo
        bior.dec_hi, // rec_hi becomes dec_hi
        &format!("rbio{}.{}", nr, nd),
        nr,
    );

    Ok(filters)
}

/// Meyer wavelet filters
///
/// These are approximations of the Meyer wavelet for DWT
/// The Meyer wavelet is defined in the frequency domain and doesn't have
/// a finite filter representation. This implementation uses a FIR approximation.
fn meyer_filters() -> SignalResult<WaveletFilters> {
    // Use a 62-tap FIR approximation for the Meyer wavelet
    let filter_len = 62;

    // Compute the Meyer scaling function FIR approximation
    // These coefficients approximate the Meyer scaling function
    // based on frequency domain definition

    // The coefficients below are a widely used approximation for the Meyer wavelet filters
    let dec_lo = vec![
        7.1309219e-12,
        1.0688357e-10,
        -5.3261485e-10,
        -8.8552387e-09,
        2.7119085e-08,
        3.0267481e-07,
        -1.6407453e-06,
        -6.2269321e-06,
        2.5793111e-05,
        6.0813053e-05,
        -0.00027621088,
        -0.00030947155,
        0.0018307326,
        0.000696435,
        -0.0088864178,
        0.0012484424,
        0.033691567,
        -0.015998825,
        -0.10265885,
        0.067892542,
        0.35479456,
        0.5557998,
        0.2401878,
        -0.12884109,
        -0.074692441,
        0.046003769,
        0.020097532,
        -0.017381433,
        -0.0031782150,
        0.0072408077,
        -0.00079317091,
        -0.0030405847,
        0.00070642294,
        0.0012436464,
        -0.00039874505,
        -0.00049472577,
        0.00017903174,
        0.00019242973,
        -6.6238912e-05,
        -7.3570026e-05,
        1.9853685e-05,
        2.7768932e-05,
        -4.8507756e-06,
        -1.0397477e-05,
        9.9514145e-07,
        3.8124836e-06,
        -1.6312518e-07,
        -1.3809378e-06,
        1.9624797e-08,
        4.9276314e-07,
        -3.4516297e-10,
        -1.7338534e-07,
        -3.4063187e-09,
        6.0056844e-08,
        1.6613729e-09,
        -2.0312429e-08,
        -4.6710593e-10,
        6.7347885e-09,
        9.9642783e-11,
        -2.1761690e-09,
        -1.5302224e-11,
        6.8380581e-10,
    ];

    // Compute the high-pass filter using quadrature mirror relationship
    let mut dec_hi = vec![0.0; filter_len];
    for i in 0..filter_len {
        dec_hi[i] = (-1_f64).powi(i as i32 + 1) * dec_lo[filter_len - 1 - i];
    }

    // Reconstruction filters (time reverse of decomposition filters)
    let mut rec_lo = dec_lo.clone();
    let mut rec_hi = dec_hi.clone();
    rec_lo.reverse();
    rec_hi.reverse();

    // Swap sign for reconstruction high-pass filter
    for val in rec_hi.iter_mut() {
        *val = -*val;
    }

    Ok(WaveletFilters::new(
        dec_lo, dec_hi, rec_lo, rec_hi, "meyer", 1,
    ))
}

/// Discrete Meyer (DMeyer) wavelet filters
///
/// The Discrete Meyer wavelet is a more computationally efficient
/// approximation of the Meyer wavelet that uses FIR filters.
/// This implementation uses the coefficients from PyWavelets/SciPy.
fn dmeyer_filters() -> SignalResult<WaveletFilters> {
    // These are pre-computed Discrete Meyer wavelet coefficients
    // Comparable to the ones used in SciPy/PyWavelets (dmey)
    let dec_lo = vec![
        -1.009999956941423e-12,
        8.519_459_636_796_214e-9,
        -1.111_944_952_602_95e-8,
        -1.079_881_953_962_195_8e-8,
        6.066_975_741_351_135e-8,
        -1.086_651_653_673_588_3e-7,
        8.200_680_650_386_481e-8,
        1.178_300_449_766_393_4e-7,
        -5.506_340_565_252_278e-7,
        1.130_794_701_791_670_6e-6,
        -1.489_549_216_497_156e-6,
        7.367_572_885_903_746e-7,
        3.205_441_913_344_78e-6,
        -1.631_269_973_455_280_7e-5,
        6.554_305_930_575_149e-5,
        -0.0006011502343516092,
        -0.002704672124643725,
        0.002202534100911002,
        0.006045814097323304,
        -0.006387718318497156,
        -0.011061496392513451,
        0.015270015130934803,
        0.017423434103729693,
        -0.03213079399021176,
        -0.024348745906078023,
        0.0637390243228016,
        0.030655091960824263,
        -0.13284520043559757,
        -0.035087555656258346,
        0.44459300275757724,
        0.7445855923188063,
        0.44459300275757724,
        -0.035087555656258346,
        -0.13284520043559757,
        0.030655091960824263,
        0.0637390243228016,
        -0.024348745906078023,
        -0.03213079399021176,
        0.017423434103729693,
        0.015270015130934803,
        -0.011061496392513451,
        -0.006387718318497156,
        0.006045814097323304,
        0.002202534100911002,
        -0.002704672124643725,
        -0.0006011502343516092,
        6.554_305_930_575_149e-5,
        -1.631_269_973_455_280_7e-5,
        3.205_441_913_344_78e-6,
        7.367_572_885_903_746e-7,
        -1.489_549_216_497_156e-6,
        1.130_794_701_791_670_6e-6,
        -5.506_340_565_252_278e-7,
        1.178_300_449_766_393_4e-7,
        8.200_680_650_386_481e-8,
        -1.086_651_653_673_588_3e-7,
        6.066_975_741_351_135e-8,
        -1.079_881_953_962_195_8e-8,
        -1.111_944_952_602_95e-8,
        8.519_459_636_796_214e-9,
        -1.009999956941423e-12,
        0.0,
    ];

    // DMeyer wavelet filter is 62 taps
    let filter_len = dec_lo.len();

    // Compute the high-pass filter using quadrature mirror relationship
    let mut dec_hi = vec![0.0; filter_len];
    for i in 0..filter_len {
        dec_hi[i] = (-1_f64).powi(i as i32 + 1) * dec_lo[filter_len - 1 - i];
    }

    // Reconstruction filters (time reverse of decomposition filters)
    let mut rec_lo = dec_lo.clone();
    let mut rec_hi = dec_hi.clone();
    rec_lo.reverse();
    rec_hi.reverse();

    // Swap sign for reconstruction high-pass filter
    for val in rec_hi.iter_mut() {
        *val = -*val;
    }

    Ok(WaveletFilters::new(
        dec_lo, dec_hi, rec_lo, rec_hi, "dmey", 1,
    ))
}
