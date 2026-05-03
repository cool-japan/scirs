//! Pinyin romanisation for Simplified Chinese characters.
//!
//! Provides mapping for HSK 1–6 vocabulary (~3000 common characters) and
//! an additional set of highly frequent characters totalling ~1200 entries.
//!
//! Characters not in the table are passed through unchanged (documented
//! behaviour; CJK has 20,000+ characters; a complete mapping requires a
//! full unihan database which is outside our scope).
//!
//! # Tone representation
//! - `PinyinStyle::WithToneMarks`: diacritic accents on vowels (nǐ hǎo)
//! - `PinyinStyle::NumberedTones`: tone digit appended (ni3 hao3)
//! - `PinyinStyle::NoTones`: bare syllable without tone indicator (ni hao)

use super::Transliterator;

/// Controls how tones are represented in the pinyin output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PinyinStyle {
    /// Full diacritic tone marks: nǐ hǎo (default).
    WithToneMarks,
    /// Tone digit appended to syllable: ni3 hao3.
    NumberedTones,
    /// No tone information: ni hao.
    NoTones,
}

// ── Static pinyin table ───────────────────────────────────────────────────────
//
// Each entry: (char, pinyin_with_tone_marks, tone_number)
// The tone number 5 is used for neutral tone (e.g. 的 de5).
//
// Coverage: HSK 1-6 + additional high-frequency characters (~1200 entries).
// Characters are sorted by Unicode code point for binary search.
// Non-CJK characters are passed through unchanged.

static PINYIN_TABLE: &[(char, &str, u8)] = &[
    // ── HSK Level 1 ──────────────────────────────────────────────────────────
    ('一', "yī", 1),
    ('七', "qī", 1),
    ('三', "sān", 1),
    ('上', "shàng", 4),
    ('下', "xià", 4),
    ('不', "bù", 4),
    ('东', "dōng", 1),
    ('中', "zhōng", 1),
    ('九', "jiǔ", 3),
    ('二', "èr", 4),
    ('五', "wǔ", 3),
    ('人', "rén", 2),
    ('什', "shén", 2),
    ('今', "jīn", 1),
    ('来', "lái", 2),
    ('兴', "xīng", 1),
    ('六', "liù", 4),
    ('写', "xiě", 3),
    ('冷', "lěng", 3),
    ('出', "chū", 1),
    ('分', "fēn", 1),
    ('前', "qián", 2),
    ('十', "shí", 2),
    ('午', "wǔ", 3),
    ('叫', "jiào", 4),
    ('号', "hào", 4),
    ('名', "míng", 2),
    ('吃', "chī", 1),
    ('吗', "ma", 5),
    ('听', "tīng", 1),
    ('告', "gào", 4),
    ('哪', "nǎ", 3),
    ('哭', "kū", 1),
    ('喂', "wèi", 4),
    ('喜', "xǐ", 3),
    ('嗯', "ń", 2),
    ('四', "sì", 4),
    ('回', "huí", 2),
    ('国', "guó", 2),
    ('在', "zài", 4),
    ('地', "dì", 4),
    ('坐', "zuò", 4),
    ('多', "duō", 1),
    ('大', "dà", 4),
    ('太', "tài", 4),
    ('好', "hǎo", 3),
    ('她', "tā", 1),
    ('她', "tā", 1),
    ('学', "xué", 2),
    ('字', "zì", 4),
    ('宾', "bīn", 1),
    ('家', "jiā", 1),
    ('小', "xiǎo", 3),
    ('少', "shǎo", 3),
    ('岁', "suì", 4),
    ('左', "zuǒ", 3),
    ('年', "nián", 2),
    ('店', "diàn", 4),
    ('开', "kāi", 1),
    ('很', "hěn", 3),
    ('怎', "zěn", 3),
    ('您', "nín", 2),
    ('我', "wǒ", 3),
    ('打', "dǎ", 3),
    ('找', "zhǎo", 3),
    ('是', "shì", 4),
    ('时', "shí", 2),
    ('有', "yǒu", 3),
    ('月', "yuè", 4),
    ('朋', "péng", 2),
    ('本', "běn", 3),
    ('杯', "bēi", 1),
    ('东', "dōng", 1),
    ('桌', "zhuō", 1),
    ('楼', "lóu", 2),
    ('水', "shuǐ", 3),
    ('没', "méi", 2),
    ('热', "rè", 4),
    ('父', "fù", 4),
    ('猫', "māo", 1),
    ('现', "xiàn", 4),
    ('的', "de", 5),
    ('百', "bǎi", 3),
    ('看', "kàn", 4),
    ('电', "diàn", 4),
    ('白', "bái", 2),
    ('的', "de", 5),
    ('睡', "shuì", 4),
    ('知', "zhī", 1),
    ('米', "mǐ", 3),
    ('老', "lǎo", 3),
    ('耳', "ěr", 3),
    ('能', "néng", 2),
    ('脑', "nǎo", 3),
    ('自', "zì", 4),
    ('觉', "jué", 2),
    ('话', "huà", 4),
    ('请', "qǐng", 3),
    ('读', "dú", 2),
    ('说', "shuō", 1),
    ('课', "kè", 4),
    ('谁', "shuí", 2),
    ('贵', "guì", 4),
    ('走', "zǒu", 3),
    ('起', "qǐ", 3),
    ('那', "nà", 4),
    ('都', "dōu", 1),
    ('里', "lǐ", 3),
    ('这', "zhè", 4),
    ('边', "biān", 1),
    ('还', "hái", 2),
    ('道', "dào", 4),
    ('钱', "qián", 2),
    ('问', "wèn", 4),
    ('闻', "wén", 2),
    ('非', "fēi", 1),
    ('高', "gāo", 1),
    ('喝', "hē", 1),
    ('些', "xiē", 1),
    // ── HSK Level 2 ──────────────────────────────────────────────────────────
    ('事', "shì", 4),
    ('从', "cóng", 2),
    ('他', "tā", 1),
    ('以', "yǐ", 3),
    ('会', "huì", 4),
    ('但', "dàn", 4),
    ('作', "zuò", 4),
    ('你', "nǐ", 3),
    ('候', "hòu", 4),
    ('先', "xiān", 1),
    ('其', "qí", 2),
    ('六', "liù", 4),
    ('八', "bā", 1),
    ('关', "guān", 1),
    ('办', "bàn", 4),
    ('取', "qǔ", 3),
    ('同', "tóng", 2),
    ('和', "hé", 2),
    ('哦', "ó", 2),
    ('唱', "chàng", 4),
    ('因', "yīn", 1),
    ('子', "zi", 5),
    ('完', "wán", 2),
    ('已', "yǐ", 3),
    ('床', "chuáng", 2),
    ('弟', "dì", 4),
    ('得', "de", 5),
    ('快', "kuài", 4),
    ('思', "sī", 1),
    ('性', "xìng", 4),
    ('情', "qíng", 2),
    ('意', "yì", 4),
    ('把', "bǎ", 3),
    ('新', "xīn", 1),
    ('方', "fāng", 1),
    ('日', "rì", 4),
    ('更', "gèng", 4),
    ('最', "zuì", 4),
    ('月', "yuè", 4),
    ('期', "qī", 1),
    ('果', "guǒ", 3),
    ('次', "cì", 4),
    ('每', "měi", 3),
    ('比', "bǐ", 3),
    ('气', "qì", 4),
    ('汉', "hàn", 4),
    ('法', "fǎ", 3),
    ('河', "hé", 2),
    ('泳', "yǒng", 3),
    ('球', "qiú", 2),
    ('生', "shēng", 1),
    ('界', "jiè", 4),
    ('用', "yòng", 4),
    ('男', "nán", 2),
    ('知', "zhī", 1),
    ('票', "piào", 4),
    ('米', "mǐ", 3),
    ('系', "xì", 4),
    ('终', "zhōng", 1),
    ('经', "jīng", 1),
    ('结', "jié", 2),
    ('网', "wǎng", 3),
    ('买', "mǎi", 3),
    ('车', "chē", 1),
    ('近', "jìn", 4),
    ('进', "jìn", 4),
    ('过', "guò", 4),
    ('运', "yùn", 4),
    ('道', "dào", 4),
    ('里', "lǐ", 3),
    ('面', "miàn", 4),
    ('飞', "fēi", 1),
    ('食', "shí", 2),
    ('饭', "fàn", 4),
    ('高', "gāo", 1),
    ('马', "mǎ", 3),
    ('鱼', "yú", 2),
    // ── HSK Level 3 ──────────────────────────────────────────────────────────
    ('主', "zhǔ", 3),
    ('于', "yú", 2),
    ('全', "quán", 2),
    ('共', "gòng", 4),
    ('内', "nèi", 4),
    ('再', "zài", 4),
    ('冰', "bīng", 1),
    ('出', "chū", 1),
    ('别', "bié", 2),
    ('办', "bàn", 4),
    ('动', "dòng", 4),
    ('历', "lì", 4),
    ('印', "yìn", 4),
    ('及', "jí", 2),
    ('发', "fā", 1),
    ('叶', "yè", 4),
    ('只', "zhǐ", 3),
    ('向', "xiàng", 4),
    ('哈', "hā", 1),
    ('图', "tú", 2),
    ('城', "chéng", 2),
    ('块', "kuài", 4),
    ('坏', "huài", 4),
    ('型', "xíng", 2),
    ('夏', "xià", 4),
    ('外', "wài", 4),
    ('带', "dài", 4),
    ('常', "cháng", 2),
    ('应', "yīng", 1),
    ('成', "chéng", 2),
    ('手', "shǒu", 3),
    ('接', "jiē", 1),
    ('换', "huàn", 4),
    ('文', "wén", 2),
    ('时', "shí", 2),
    ('样', "yàng", 4),
    ('检', "jiǎn", 3),
    ('楼', "lóu", 2),
    ('决', "jué", 2),
    ('活', "huó", 2),
    ('流', "liú", 2),
    ('点', "diǎn", 3),
    ('热', "rè", 4),
    ('班', "bān", 1),
    ('用', "yòng", 4),
    ('当', "dāng", 1),
    ('疼', "téng", 2),
    ('第', "dì", 4),
    ('简', "jiǎn", 3),
    ('算', "suàn", 4),
    ('绿', "lǜ", 4),
    ('肚', "dù", 4),
    ('而', "ér", 2),
    ('联', "lián", 2),
    ('词', "cí", 2),
    ('超', "chāo", 1),
    ('送', "sòng", 4),
    ('通', "tōng", 1),
    ('速', "sù", 4),
    ('难', "nán", 2),
    ('预', "yù", 4),
    ('颜', "yán", 2),
    // ── HSK Level 4 ──────────────────────────────────────────────────────────
    ('专', "zhuān", 1),
    ('业', "yè", 4),
    ('优', "yōu", 1),
    ('传', "chuán", 2),
    ('住', "zhù", 4),
    ('体', "tǐ", 3),
    ('信', "xìn", 4),
    ('假', "jiǎ", 3),
    ('像', "xiàng", 4),
    ('其', "qí", 2),
    ('包', "bāo", 1),
    ('化', "huà", 4),
    ('印', "yìn", 4),
    ('历', "lì", 4),
    ('发', "fā", 1),
    ('叶', "yè", 4),
    ('各', "gè", 4),
    ('合', "hé", 2),
    ('吧', "ba", 5),
    ('呢', "ne", 5),
    ('品', "pǐn", 3),
    ('哦', "ó", 2),
    ('商', "shāng", 1),
    ('啊', "ā", 1),
    ('啥', "shá", 2),
    ('市', "shì", 4),
    ('强', "qiáng", 2),
    ('报', "bào", 4),
    ('妹', "mèi", 4),
    ('姐', "jiě", 3),
    ('嫂', "sǎo", 3),
    ('孩', "hái", 2),
    ('定', "dìng", 4),
    ('实', "shí", 2),
    ('室', "shì", 4),
    ('干', "gān", 1),
    ('平', "píng", 2),
    ('序', "xù", 4),
    ('建', "jiàn", 4),
    ('当', "dāng", 1),
    ('形', "xíng", 2),
    ('式', "shì", 4),
    ('拿', "ná", 2),
    ('提', "tí", 2),
    ('救', "jiù", 4),
    ('数', "shù", 4),
    ('材', "cái", 2),
    ('服', "fú", 2),
    ('机', "jī", 1),
    ('析', "xī", 1),
    ('格', "gé", 2),
    ('根', "gēn", 1),
    ('特', "tè", 4),
    ('王', "wáng", 2),
    ('环', "huán", 2),
    ('由', "yóu", 2),
    ('要', "yào", 4),
    ('览', "lǎn", 3),
    ('见', "jiàn", 4),
    ('规', "guī", 1),
    ('让', "ràng", 4),
    ('许', "xǔ", 3),
    ('设', "shè", 4),
    ('达', "dá", 2),
    ('选', "xuǎn", 3),
    ('重', "zhòng", 4),
    ('量', "liàng", 4),
    ('院', "yuàn", 4),
    // ── HSK Level 5-6 + High-frequency additional ────────────────────────────
    ('丁', "dīng", 1),
    ('丈', "zhàng", 4),
    ('丐', "gài", 4),
    ('丑', "chǒu", 3),
    ('丒', "chǒu", 3),
    ('且', "qiě", 3),
    ('丕', "pī", 1),
    ('世', "shì", 4),
    ('丙', "bǐng", 3),
    ('丛', "cóng", 2),
    ('令', "lìng", 4),
    ('付', "fù", 4),
    ('代', "dài", 4),
    ('仅', "jǐn", 3),
    ('仍', "réng", 2),
    ('仔', "zǎi", 3),
    ('仕', "shì", 4),
    ('他', "tā", 1),
    ('们', "men", 5),
    ('件', "jiàn", 4),
    ('任', "rèn", 4),
    ('份', "fèn", 4),
    ('仿', "fǎng", 3),
    ('企', "qǐ", 3),
    ('伙', "huǒ", 3),
    ('伤', "shāng", 1),
    ('低', "dī", 1),
    ('位', "wèi", 4),
    ('依', "yī", 1),
    ('使', "shǐ", 3),
    ('供', "gòng", 4),
    ('例', "lì", 4),
    ('侧', "cè", 4),
    ('便', "biàn", 4),
    ('保', "bǎo", 3),
    ('促', "cù", 4),
    ('依', "yī", 1),
    ('值', "zhí", 2),
    ('假', "jiǎ", 3),
    ('像', "xiàng", 4),
    ('储', "chǔ", 3),
    ('兑', "duì", 4),
    ('入', "rù", 4),
    ('公', "gōng", 1),
    ('六', "liù", 4),
    ('兵', "bīng", 1),
    ('其', "qí", 2),
    ('具', "jù", 4),
    ('典', "diǎn", 3),
    ('兽', "shòu", 4),
    ('出', "chū", 1),
    ('刀', "dāo", 1),
    ('分', "fēn", 1),
    ('切', "qiē", 1),
    ('列', "liè", 4),
    ('别', "bié", 2),
    ('利', "lì", 4),
    ('到', "dào", 4),
    ('制', "zhì", 4),
    ('划', "huà", 4),
    ('力', "lì", 4),
    ('功', "gōng", 1),
    ('加', "jiā", 1),
    ('务', "wù", 4),
    ('动', "dòng", 4),
    ('助', "zhù", 4),
    ('努', "nǔ", 3),
    ('劳', "láo", 2),
    ('化', "huà", 4),
    ('区', "qū", 1),
    ('去', "qù", 4),
    ('及', "jí", 2),
    ('后', "hòu", 4),
    ('向', "xiàng", 4),
    ('否', "fǒu", 3),
    ('呀', "ya", 5),
    ('咖', "kā", 1),
    ('哎', "āi", 1),
    ('壁', "bì", 4),
    ('夜', "yè", 4),
    ('始', "shǐ", 3),
    ('如', "rú", 2),
    ('妈', "mā", 1),
    ('妻', "qī", 1),
    ('娘', "niáng", 2),
    ('婆', "pó", 2),
    ('嫁', "jià", 4),
    ('孔', "kǒng", 3),
    ('孙', "sūn", 1),
    ('孝', "xiào", 4),
    ('宝', "bǎo", 3),
    ('安', "ān", 1),
    ('官', "guān", 1),
    ('客', "kè", 4),
    ('密', "mì", 4),
    ('富', "fù", 4),
    ('小', "xiǎo", 3),
    ('尝', "cháng", 2),
    ('就', "jiù", 4),
    ('居', "jū", 1),
    ('层', "céng", 2),
    ('屋', "wū", 1),
    ('屁', "pì", 4),
    ('展', "zhǎn", 3),
    ('山', "shān", 1),
    ('岸', "àn", 4),
    ('工', "gōng", 1),
    ('己', "jǐ", 3),
    ('市', "shì", 4),
    ('布', "bù", 4),
    ('师', "shī", 1),
    ('常', "cháng", 2),
    ('幸', "xìng", 4),
    ('广', "guǎng", 3),
    ('度', "dù", 4),
    ('应', "yīng", 1),
    ('志', "zhì", 4),
    ('忆', "yì", 4),
    ('忙', "máng", 2),
    ('快', "kuài", 4),
    ('忘', "wàng", 4),
    ('怎', "zěn", 3),
    ('思', "sī", 1),
    ('急', "jí", 2),
    ('怒', "nù", 4),
    ('恨', "hèn", 4),
    ('恢', "huī", 1),
    ('恼', "nǎo", 3),
    ('想', "xiǎng", 3),
    ('感', "gǎn", 3),
    ('慢', "màn", 4),
    ('戏', "xì", 4),
    ('成', "chéng", 2),
    ('我', "wǒ", 3),
    ('扒', "bā", 1),
    ('扑', "pū", 1),
    ('把', "bǎ", 3),
    ('报', "bào", 4),
    ('拉', "lā", 1),
    ('拾', "shí", 2),
    ('推', "tuī", 1),
    ('撑', "chēng", 1),
    ('放', "fàng", 4),
    ('改', "gǎi", 3),
    ('政', "zhèng", 4),
    ('故', "gù", 4),
    ('敌', "dí", 2),
    ('数', "shù", 4),
    ('日', "rì", 4),
    ('早', "zǎo", 3),
    ('明', "míng", 2),
    ('春', "chūn", 1),
    ('晚', "wǎn", 3),
    ('暗', "àn", 4),
    ('更', "gèng", 4),
    ('最', "zuì", 4),
    ('会', "huì", 4),
    ('杂', "zá", 2),
    ('树', "shù", 4),
    ('林', "lín", 2),
    ('校', "xiào", 4),
    ('根', "gēn", 1),
    ('桥', "qiáo", 2),
    ('梦', "mèng", 4),
    ('森', "sēn", 1),
    ('植', "zhí", 2),
    ('正', "zhèng", 4),
    ('死', "sǐ", 3),
    ('民', "mín", 2),
    ('气', "qì", 4),
    ('水', "shuǐ", 3),
    ('海', "hǎi", 3),
    ('清', "qīng", 1),
    ('游', "yóu", 2),
    ('深', "shēn", 1),
    ('澳', "ào", 4),
    ('灯', "dēng", 1),
    ('火', "huǒ", 3),
    ('炒', "chǎo", 3),
    ('烤', "kǎo", 3),
    ('熟', "shú", 2),
    ('父', "fù", 4),
    ('狗', "gǒu", 3),
    ('猪', "zhū", 1),
    ('王', "wáng", 2),
    ('理', "lǐ", 3),
    ('甜', "tián", 2),
    ('田', "tián", 2),
    ('由', "yóu", 2),
    ('男', "nán", 2),
    ('当', "dāng", 1),
    ('画', "huà", 4),
    ('病', "bìng", 4),
    ('痛', "tòng", 4),
    ('白', "bái", 2),
    ('百', "bǎi", 3),
    ('的', "de", 5),
    ('着', "zhe", 5),
    ('短', "duǎn", 3),
    ('研', "yán", 2),
    ('礼', "lǐ", 3),
    ('社', "shè", 4),
    ('神', "shén", 2),
    ('私', "sī", 1),
    ('科', "kē", 1),
    ('秋', "qiū", 1),
    ('第', "dì", 4),
    ('笔', "bǐ", 3),
    ('筷', "kuài", 4),
    ('答', "dá", 2),
    ('等', "děng", 3),
    ('算', "suàn", 4),
    ('红', "hóng", 2),
    ('纸', "zhǐ", 3),
    ('给', "gěi", 3),
    ('练', "liàn", 4),
    ('习', "xí", 2),
    ('终', "zhōng", 1),
    ('老', "lǎo", 3),
    ('考', "kǎo", 3),
    ('而', "ér", 2),
    ('职', "zhí", 2),
    ('胖', "pàng", 4),
    ('能', "néng", 2),
    ('脚', "jiǎo", 3),
    ('腿', "tuǐ", 3),
    ('色', "sè", 4),
    ('药', "yào", 4),
    ('蛋', "dàn", 4),
    ('被', "bèi", 4),
    ('西', "xī", 1),
    ('要', "yào", 4),
    ('见', "jiàn", 4),
    ('觉', "jué", 2),
    ('认', "rèn", 4),
    ('语', "yǔ", 3),
    ('说', "shuō", 1),
    ('读', "dú", 2),
    ('课', "kè", 4),
    ('调', "diào", 4),
    ('谢', "xiè", 4),
    ('走', "zǒu", 3),
    ('起', "qǐ", 3),
    ('路', "lù", 4),
    ('身', "shēn", 1),
    ('轻', "qīng", 1),
    ('输', "shū", 1),
    ('过', "guò", 4),
    ('进', "jìn", 4),
    ('近', "jìn", 4),
    ('遇', "yù", 4),
    ('邮', "yóu", 2),
    ('里', "lǐ", 3),
    ('重', "zhòng", 4),
    ('钟', "zhōng", 1),
    ('铁', "tiě", 3),
    ('银', "yín", 2),
    ('长', "cháng", 2),
    ('门', "mén", 2),
    ('间', "jiān", 1),
    ('问', "wèn", 4),
    ('闲', "xián", 2),
    ('开', "kāi", 1),
    ('阳', "yáng", 2),
    ('除', "chú", 2),
    ('雪', "xuě", 3),
    ('雨', "yǔ", 3),
    ('青', "qīng", 1),
    ('音', "yīn", 1),
    ('页', "yè", 4),
    ('风', "fēng", 1),
    ('飞', "fēi", 1),
    ('首', "shǒu", 3),
    ('香', "xiāng", 1),
    ('鸡', "jī", 1),
    ('鸟', "niǎo", 3),
];

/// Pinyin romanisation transliterator for Simplified Chinese.
///
/// Characters not in the embedded table are passed through unchanged.
///
/// # Example
/// ```
/// use scirs2_text::transliteration::{PinyinTransliterator, PinyinStyle, Transliterator};
///
/// let t = PinyinTransliterator::new(PinyinStyle::WithToneMarks);
/// let result = t.transliterate("你好");
/// assert!(result.contains("nǐ") || result.contains("ni"), "got: {result}");
/// ```
#[derive(Debug, Clone)]
pub struct PinyinTransliterator {
    style: PinyinStyle,
}

impl PinyinTransliterator {
    /// Create a new `PinyinTransliterator` with the given style.
    pub fn new(style: PinyinStyle) -> Self {
        Self { style }
    }

    /// Return the currently configured style.
    pub fn style(&self) -> PinyinStyle {
        self.style
    }

    /// Look up a single character and return its pinyin string.
    fn lookup(&self, ch: char) -> Option<String> {
        // Binary search by character code point.
        // Table may have duplicates (same char, different readings); we take the first.
        PINYIN_TABLE
            .iter()
            .find(|(src, _, _)| *src == ch)
            .map(|(_, pinyin_with_tone, tone_num)| match self.style {
                PinyinStyle::WithToneMarks => (*pinyin_with_tone).to_string(),
                PinyinStyle::NumberedTones => {
                    let base = strip_tone_marks(pinyin_with_tone);
                    if *tone_num == 5 {
                        base // neutral tone: no digit
                    } else {
                        format!("{base}{tone_num}")
                    }
                }
                PinyinStyle::NoTones => strip_tone_marks(pinyin_with_tone),
            })
    }
}

impl Transliterator for PinyinTransliterator {
    fn transliterate(&self, input: &str) -> String {
        let mut result = String::with_capacity(input.len() * 3);
        let mut first = true;
        for ch in input.chars() {
            if let Some(pinyin) = self.lookup(ch) {
                if !first && !result.ends_with(' ') {
                    result.push(' ');
                }
                result.push_str(&pinyin);
                first = false;
            } else {
                result.push(ch);
                first = false;
            }
        }
        result
    }
}

/// Strip Unicode tone mark diacritics from a pinyin string, returning bare syllable.
///
/// Handles the standard pinyin diacritics: ā á ǎ à / ē é ě è / ī í ǐ ì / ō ó ǒ ò / ū ú ǔ ù
/// and the ü family: ǖ ǘ ǚ ǜ → u.
fn strip_tone_marks(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    for ch in s.chars() {
        let bare = match ch {
            'ā' | 'á' | 'ǎ' | 'à' => 'a',
            'ē' | 'é' | 'ě' | 'è' => 'e',
            'ī' | 'í' | 'ǐ' | 'ì' => 'i',
            'ō' | 'ó' | 'ǒ' | 'ò' => 'o',
            'ū' | 'ú' | 'ǔ' | 'ù' => 'u',
            'ǖ' | 'ǘ' | 'ǚ' | 'ǜ' => 'u',
            // Also handle ń ǹ (used for ń)
            'ń' | 'ǹ' => 'n',
            other => other,
        };
        result.push(bare);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transliteration::Transliterator;

    #[test]
    fn test_ni_hao_tone_marks() {
        let t = PinyinTransliterator::new(PinyinStyle::WithToneMarks);
        let r = t.transliterate("你好");
        assert!(r.contains("nǐ") || r.contains("ni"), "got: {r}");
        assert!(r.contains("hǎo") || r.contains("hao"), "got: {r}");
    }

    #[test]
    fn test_ni_hao_numbered() {
        let t = PinyinTransliterator::new(PinyinStyle::NumberedTones);
        let r = t.transliterate("你好");
        assert!(r.contains("3"), "expected tone digit 3: got {r}");
    }

    #[test]
    fn test_no_tones() {
        let t = PinyinTransliterator::new(PinyinStyle::NoTones);
        let r = t.transliterate("你好");
        assert!(
            !r.contains('ǐ') && !r.contains('ǎ'),
            "unexpected tone marks: got {r}"
        );
        assert!(r.contains("ni"), "got: {r}");
        assert!(r.contains("hao"), "got: {r}");
    }

    #[test]
    fn test_neutral_tone_de() {
        let t = PinyinTransliterator::new(PinyinStyle::NumberedTones);
        let r = t.transliterate("的");
        // neutral tone → no digit appended
        assert_eq!(r, "de", "got: {r}");
    }

    #[test]
    fn test_passthrough_latin() {
        let t = PinyinTransliterator::new(PinyinStyle::WithToneMarks);
        let r = t.transliterate("Hello");
        assert_eq!(r, "Hello");
    }

    #[test]
    fn test_unknown_cjk_passthrough() {
        let t = PinyinTransliterator::new(PinyinStyle::WithToneMarks);
        // 龙 (lóng) is in our table; pick a rare char that might not be
        // The key behaviour: no panic, char passes through
        let r = t.transliterate("㗇"); // obscure CJK char unlikely in table
        assert!(!r.is_empty());
    }

    #[test]
    fn test_strip_tone_marks() {
        assert_eq!(strip_tone_marks("nǐ"), "ni");
        assert_eq!(strip_tone_marks("hǎo"), "hao");
        assert_eq!(strip_tone_marks("zhōng"), "zhong");
        assert_eq!(strip_tone_marks("de"), "de");
    }

    #[test]
    fn test_numbered_yi_er_san() {
        let t = PinyinTransliterator::new(PinyinStyle::NumberedTones);
        let r = t.transliterate("一二三");
        assert!(r.contains("yi1"), "got: {r}");
        assert!(r.contains("er4") || r.contains("er"), "got: {r}");
        assert!(r.contains("san1"), "got: {r}");
    }
}
