//! EEG channel position lookup from embedded standard montage files.
//!
//! Six MNE-Python ASA `.elc` montage files are embedded at compile time via
//! `include_str!`.  Positions are in the ASA head coordinate frame (x=right,
//! y=anterior, z=superior) and are scaled to a head radius of **0.085 m** so
//! that all electrode positions stay comfortably within the ZUNA
//! [`DataConfig`](crate::config::DataConfig) bounding box of ±0.12 m.
//!
//! ## Layouts
//!
//! | Variant | File | Channels |
//! |---|---|---|
//! | `Standard1020` | `standard_1020.elc` | 94 (10-20 + 10-10 rim) |
//! | `Standard1005` | `standard_1005.elc` | 343 (full 10-5 system)  |
//! | `StandardAlphabetic` | `standard_alphabetic.elc` | 65 |
//! | `StandardPostfixed`  | `standard_postfixed.elc`  | 100 |
//! | `StandardPrefixed`   | `standard_prefixed.elc`   | 74  |
//! | `StandardPrimed`     | `standard_primed.elc`     | 100 |
//!
//! ## Quick start
//!
//! ```rust
//! use zuna_rs::channel_positions::{channel_xyz, MontageLayout, montage_channels};
//!
//! // Look up a single channel (searches Standard1005 first, then others)
//! let xyz = channel_xyz("Cz").unwrap();
//! println!("Cz: {xyz:?}");
//!
//! // Iterate all channels in the 10-20 montage
//! for (name, xyz) in montage_channels(MontageLayout::Standard1020) {
//!     println!("{name}: {xyz:?}");
//! }
//! ```

use std::collections::HashMap;
use std::sync::OnceLock;

// ── Embedded source files ─────────────────────────────────────────────────────

const ELC_1020:        &str = include_str!("montages/standard_1020.elc");
const ELC_1005:        &str = include_str!("montages/standard_1005.elc");
const ELC_ALPHABETIC:  &str = include_str!("montages/standard_alphabetic.elc");
const ELC_POSTFIXED:   &str = include_str!("montages/standard_postfixed.elc");
const ELC_PREFIXED:    &str = include_str!("montages/standard_prefixed.elc");
const ELC_PRIMED:      &str = include_str!("montages/standard_primed.elc");

// ── Public API ────────────────────────────────────────────────────────────────

/// One of the six standard montage layouts bundled with zuna-rs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MontageLayout {
    /// MNE `standard_1020` — 10-20 system + 10-10 rim electrodes (~94 ch).
    Standard1020,
    /// MNE `standard_1005` — full 10-5 system (~343 ch).  Most comprehensive.
    Standard1005,
    /// MNE `standard_alphabetic` — alternate single-letter/number naming (~65 ch).
    StandardAlphabetic,
    /// MNE `standard_postfixed` — postfix-style naming (~100 ch).
    StandardPostfixed,
    /// MNE `standard_prefixed` — prefix-style naming (~74 ch).
    StandardPrefixed,
    /// MNE `standard_primed` — prime-style naming (~100 ch).
    StandardPrimed,
}

impl MontageLayout {
    /// All six layouts, ordered from most to least comprehensive.
    pub const ALL: &'static [MontageLayout] = &[
        MontageLayout::Standard1005,
        MontageLayout::Standard1020,
        MontageLayout::StandardPostfixed,
        MontageLayout::StandardPrimed,
        MontageLayout::StandardPrefixed,
        MontageLayout::StandardAlphabetic,
    ];

    /// Human-readable name of this layout.
    pub fn name(self) -> &'static str {
        match self {
            Self::Standard1020       => "standard_1020",
            Self::Standard1005       => "standard_1005",
            Self::StandardAlphabetic => "standard_alphabetic",
            Self::StandardPostfixed  => "standard_postfixed",
            Self::StandardPrefixed   => "standard_prefixed",
            Self::StandardPrimed     => "standard_primed",
        }
    }

}

/// Return all channels and their XYZ positions (metres) for a given layout.
///
/// Parsed and scaled **once** on first access; subsequent calls return a
/// reference to the cached map.  Fiducials (Nz, LPA, RPA) are included.
pub fn montage_channels(layout: MontageLayout) -> &'static HashMap<String, [f32; 3]> {
    // One OnceLock per layout variant.
    static C1020:  OnceLock<HashMap<String,[f32;3]>> = OnceLock::new();
    static C1005:  OnceLock<HashMap<String,[f32;3]>> = OnceLock::new();
    static CALPHA: OnceLock<HashMap<String,[f32;3]>> = OnceLock::new();
    static CPOST:  OnceLock<HashMap<String,[f32;3]>> = OnceLock::new();
    static CPRE:   OnceLock<HashMap<String,[f32;3]>> = OnceLock::new();
    static CPRIME: OnceLock<HashMap<String,[f32;3]>> = OnceLock::new();

    let (lock, src) = match layout {
        MontageLayout::Standard1020       => (&C1020,  ELC_1020),
        MontageLayout::Standard1005       => (&C1005,  ELC_1005),
        MontageLayout::StandardAlphabetic => (&CALPHA, ELC_ALPHABETIC),
        MontageLayout::StandardPostfixed  => (&CPOST,  ELC_POSTFIXED),
        MontageLayout::StandardPrefixed   => (&CPRE,   ELC_PREFIXED),
        MontageLayout::StandardPrimed     => (&CPRIME, ELC_PRIMED),
    };

    lock.get_or_init(|| parse_elc(src))
}

/// Look up the XYZ position (metres) for a channel by name.
///
/// Name matching is **case-insensitive** and ignores spaces, hyphens, and
/// underscores.  Searches [`MontageLayout::Standard1005`] first (most
/// comprehensive), then all remaining layouts in order.
///
/// Returns `None` if the name is not found in any bundled montage.
pub fn channel_xyz(name: &str) -> Option<[f32; 3]> {
    let key = normalise(name);
    for &layout in MontageLayout::ALL {
        let map = montage_channels(layout);
        // Try exact normalised key first, then the raw name
        let found = map.iter().find(|(k, _)| normalise(k) == key);
        if let Some((_, &xyz)) = found {
            return Some(xyz);
        }
    }
    None
}

/// Return the index of the nearest channel in `candidates` to `target_xyz`,
/// measured by Euclidean distance.  Returns `None` if `candidates` is empty.
pub fn nearest_channel(
    target_xyz: [f32; 3],
    candidates: &[([f32; 3], usize)],  // (xyz, original_index)
) -> Option<usize> {
    candidates.iter()
        .min_by(|(a, _), (b, _)| {
            let da = dist2(*a, target_xyz);
            let db = dist2(*b, target_xyz);
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(_, idx)| *idx)
}

// ── ELC parser ────────────────────────────────────────────────────────────────

/// Parse one ASA `.elc` file and return `{canonical_name → [x,y,z] metres}`.
///
/// Positions are scaled so that the **median** electrode-to-origin distance
/// equals [`HEAD_SIZE`] metres.
fn parse_elc(src: &str) -> HashMap<String, [f32; 3]> {
    /// Target median head radius in metres. Keeps positions within ±0.12 m.
    const HEAD_SIZE: f32 = 0.085;

    // Detect unit scale (mm → m, or m → m)
    let mm_scale: f32 = {
        let mut s = 1e-3_f32; // default mm
        for line in src.lines() {
            if line.contains("UnitPosition") {
                s = if line.contains('m') && !line.contains("mm") { 1.0 } else { 1e-3 };
                break;
            }
        }
        s
    };

    // Parse Positions block
    let mut raw: Vec<[f32; 3]> = Vec::new();
    let mut in_pos = false;
    let mut in_lbl = false;
    let mut labels: Vec<String> = Vec::new();

    for line in src.lines() {
        let t = line.trim();
        if t == "Positions" || t.starts_with("Positions") { in_pos = true; in_lbl = false; continue; }
        if t == "Labels"    || t.starts_with("Labels")    { in_lbl = true; in_pos = false; continue; }

        if in_pos {
            // Handle both "old" format (`x y z`) and "new" format (`E01 : x y z`)
            let nums: Vec<f32> = if t.contains(':') {
                t.split(':').nth(1).unwrap_or("").split_whitespace()
                    .filter_map(|s| s.parse().ok()).collect()
            } else {
                t.split_whitespace().filter_map(|s| s.parse().ok()).collect()
            };
            if nums.len() == 3 {
                raw.push([nums[0], nums[1], nums[2]]);
            }
        } else if in_lbl && !t.is_empty() {
            labels.push(t.to_string());
        }
    }

    assert_eq!(raw.len(), labels.len(),
        "ELC parse mismatch: {} positions vs {} labels", raw.len(), labels.len());

    // Convert to metres
    let mut pos_m: Vec<[f32; 3]> = raw.iter()
        .map(|p| [p[0] * mm_scale, p[1] * mm_scale, p[2] * mm_scale])
        .collect();

    // Scale so median norm == HEAD_SIZE (same as MNE's head_size parameter)
    let mut norms: Vec<f32> = pos_m.iter()
        .map(|p| (p[0]*p[0] + p[1]*p[1] + p[2]*p[2]).sqrt())
        .filter(|&n| n > 1e-6)
        .collect();
    norms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = norms[norms.len() / 2];
    if median > 1e-6 {
        let scale = HEAD_SIZE / median;
        for p in &mut pos_m { p[0] *= scale; p[1] *= scale; p[2] *= scale; }
    }

    labels.into_iter().zip(pos_m).collect()
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Normalise a channel name for case-insensitive comparison.
/// Strips spaces, hyphens, underscores and converts to uppercase.
pub fn normalise(name: &str) -> String {
    name.chars()
        .filter(|c| !matches!(c, ' ' | '-' | '_'))
        .flat_map(|c| c.to_uppercase())
        .collect()
}

fn dist2(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0]-b[0]; let dy = a[1]-b[1]; let dz = a[2]-b[2];
    dx*dx + dy*dy + dz*dz
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_all_montages() {
        for &layout in MontageLayout::ALL {
            let map = montage_channels(layout);
            assert!(!map.is_empty(), "{} parsed empty", layout.name());
            println!("{}: {} channels", layout.name(), map.len());
        }
    }

    #[test]
    fn known_channels_present() {
        for name in &["Cz", "Fz", "Pz", "C3", "C4", "Fp1", "Fp2", "O1", "O2",
                      "T7", "T8", "TP9", "TP10", "AF7", "AF8"] {
            let xyz = channel_xyz(name);
            assert!(xyz.is_some(), "channel '{name}' not found");
            let [x, y, z] = xyz.unwrap();
            // All positions must be within DataConfig bounds
            assert!(x.abs() <= 0.12, "{name} x={x} out of bounds");
            assert!(y.abs() <= 0.12, "{name} y={y} out of bounds");
            assert!(z.abs() <= 0.12, "{name} z={z} out of bounds");
        }
    }

    #[test]
    fn case_insensitive_lookup() {
        let a = channel_xyz("cz");
        let b = channel_xyz("CZ");
        let c = channel_xyz("Cz");
        assert!(a.is_some() && a == b && b == c);
    }

    #[test]
    fn old_aliases_present() {
        // T3/T7 and T4/T8 should both resolve
        let t3 = channel_xyz("T3");
        let t7 = channel_xyz("T7");
        assert!(t3.is_some(), "T3 not found");
        assert!(t7.is_some(), "T7 not found");
    }

    #[test]
    fn nearest_channel_finds_closest() {
        let cz = channel_xyz("Cz").unwrap();
        let c3 = channel_xyz("C3").unwrap();
        let c4 = channel_xyz("C4").unwrap();
        let candidates = vec![(cz, 0usize), (c3, 1usize), (c4, 2usize)];

        // A query point right on top of C3 must return C3.
        let at_c3 = c3;
        assert_eq!(nearest_channel(at_c3, &candidates).unwrap(), 1,
            "query at C3 should return C3 (idx=1)");

        // A query point right on top of C4 must return C4.
        let at_c4 = c4;
        assert_eq!(nearest_channel(at_c4, &candidates).unwrap(), 2,
            "query at C4 should return C4 (idx=2)");

        // A query point just left of midpoint between Cz and C3 should be
        // closer to C3 than to Cz/C4.
        let near_c3 = [
            c3[0] * 0.8 + cz[0] * 0.2,
            c3[1] * 0.8 + cz[1] * 0.2,
            c3[2] * 0.8 + cz[2] * 0.2,
        ];
        assert_eq!(nearest_channel(near_c3, &candidates).unwrap(), 1,
            "80% toward C3 from Cz should return C3");
    }

    #[test]
    fn positions_within_dataconfig_bounds() {
        // No channel in any montage should fall outside ±0.12 m on any axis
        for &layout in MontageLayout::ALL {
            for (name, &[x, y, z]) in montage_channels(layout) {
                assert!(x.abs() <= 0.12, "{}/{}: x={x:.4}", layout.name(), name);
                assert!(y.abs() <= 0.12, "{}/{}: y={y:.4}", layout.name(), name);
                assert!(z.abs() <= 0.12, "{}/{}: z={z:.4}", layout.name(), name);
            }
        }
    }
}
