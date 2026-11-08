//! A small, exact sRGB color with 8-bit channels, parsed from a CSS-style hex
//! string.
//!
//! Themes are written as hex strings, so this is where a theme's text turns into
//! numbers. It accepts the forms people actually write — `#RGB`, `#RGBA`,
//! `#RRGGBB`, and `#RRGGBBAA`, in any case — and nothing else. Anything else is
//! an error, not a silent fallback.

use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// An sRGB color plus an alpha channel. Not premultiplied.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Rgba {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

/// Why a hex string was not a color.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ParseColorError {
    /// The string did not start with `#`.
    MissingHash,
    /// There weren't 3, 4, 6, or 8 hex digits after `#`.
    BadLength(usize),
    /// A character after `#` was not a hex digit.
    BadDigit(char),
}

impl core::fmt::Display for ParseColorError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::MissingHash => write!(f, "color must start with '#'"),
            Self::BadLength(n) => {
                write!(f, "color must have 3, 4, 6, or 8 hex digits, found {n}")
            }
            Self::BadDigit(c) => write!(f, "'{c}' is not a hex digit"),
        }
    }
}

impl std::error::Error for ParseColorError {}

impl Rgba {
    /// Build a fully opaque color from its channels.
    pub const fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b, a: 0xff }
    }

    /// Parse a CSS-style hex color. Short forms double each digit, so `#1a2`
    /// means `#11aa22`, the same way browsers read `#rgb`.
    pub fn parse(s: &str) -> Result<Self, ParseColorError> {
        let hex = s.strip_prefix('#').ok_or(ParseColorError::MissingHash)?;
        if let Some(c) = hex.chars().find(|c| !c.is_ascii_hexdigit()) {
            return Err(ParseColorError::BadDigit(c));
        }
        let nib = |c: u8| (c as char).to_digit(16).unwrap() as u8;
        let bytes = hex.as_bytes();
        // Each channel is one hex digit (doubled) or two (a full byte).
        let (chan, has_alpha) = match hex.len() {
            3 => (1usize, false),
            4 => (1, true),
            6 => (2, false),
            8 => (2, true),
            n => return Err(ParseColorError::BadLength(n)),
        };
        let read = |i: usize| -> u8 {
            if chan == 1 {
                let v = nib(bytes[i]);
                v << 4 | v
            } else {
                nib(bytes[2 * i]) << 4 | nib(bytes[2 * i + 1])
            }
        };
        Ok(Self {
            r: read(0),
            g: read(1),
            b: read(2),
            a: if has_alpha { read(3) } else { 0xff },
        })
    }

    /// The color as linear f32 channels in `0.0..=1.0`, the form a GPU wants.
    pub fn to_linear_f32(self) -> [f32; 4] {
        [
            srgb_to_linear(self.r),
            srgb_to_linear(self.g),
            srgb_to_linear(self.b),
            self.a as f32 / 255.0,
        ]
    }
}

/// Convert one 8-bit sRGB channel to a linear f32, using the sRGB formula.
fn srgb_to_linear(c: u8) -> f32 {
    let c = c as f32 / 255.0;
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

impl core::fmt::Display for Rgba {
    /// Prints `#RRGGBB` when opaque and `#RRGGBBAA` otherwise, so alpha survives
    /// a round trip.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if self.a == 0xff {
            write!(f, "#{:02x}{:02x}{:02x}", self.r, self.g, self.b)
        } else {
            write!(
                f,
                "#{:02x}{:02x}{:02x}{:02x}",
                self.r, self.g, self.b, self.a
            )
        }
    }
}

impl Serialize for Rgba {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&self.to_string())
    }
}

impl<'de> Deserialize<'de> for Rgba {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        Rgba::parse(&s).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_every_accepted_length() {
        assert_eq!(Rgba::parse("#fff").unwrap(), Rgba::rgb(255, 255, 255));
        assert_eq!(Rgba::parse("#1a2").unwrap(), Rgba::rgb(0x11, 0xaa, 0x22));
        assert_eq!(
            Rgba::parse("#1a2f").unwrap(),
            Rgba {
                r: 0x11,
                g: 0xaa,
                b: 0x22,
                a: 0xff
            }
        );
        assert_eq!(Rgba::parse("#10203f").unwrap(), Rgba::rgb(0x10, 0x20, 0x3f));
        assert_eq!(
            Rgba::parse("#10203040").unwrap(),
            Rgba {
                r: 0x10,
                g: 0x20,
                b: 0x30,
                a: 0x40
            }
        );
    }

    #[test]
    fn is_case_insensitive() {
        assert_eq!(
            Rgba::parse("#AbCdEf").unwrap(),
            Rgba::parse("#abcdef").unwrap()
        );
    }

    #[test]
    fn rejects_malformed_input() {
        assert_eq!(Rgba::parse("fff"), Err(ParseColorError::MissingHash));
        assert_eq!(Rgba::parse("#ff"), Err(ParseColorError::BadLength(2)));
        assert_eq!(Rgba::parse("#fffff"), Err(ParseColorError::BadLength(5)));
        assert_eq!(Rgba::parse("#12345g"), Err(ParseColorError::BadDigit('g')));
    }

    #[test]
    fn display_drops_alpha_only_when_opaque() {
        assert_eq!(Rgba::rgb(0x10, 0x20, 0x30).to_string(), "#102030");
        assert_eq!(
            Rgba {
                r: 0x10,
                g: 0x20,
                b: 0x30,
                a: 0x40
            }
            .to_string(),
            "#10203040"
        );
    }

    #[test]
    fn round_trips_through_json() {
        let c = Rgba {
            r: 1,
            g: 2,
            b: 3,
            a: 4,
        };
        let json = serde_json::to_string(&c).unwrap();
        assert_eq!(json, "\"#01020304\"");
        assert_eq!(serde_json::from_str::<Rgba>(&json).unwrap(), c);
    }

    #[test]
    fn linear_endpoints_are_exact() {
        assert_eq!(Rgba::rgb(0, 0, 0).to_linear_f32(), [0.0, 0.0, 0.0, 1.0]);
        let white = Rgba::rgb(255, 255, 255).to_linear_f32();
        assert!((white[0] - 1.0).abs() < 1e-6);
    }
}
