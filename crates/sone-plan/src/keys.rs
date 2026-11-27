//! How two keys fit, on the Camelot wheel ("8A", "10B").

/// 0: same key. 1: relative (same number) or neighbour (same letter, next number).
/// 2 and more: they clash; nothing pitched of both may sound together for long.
pub fn distance(a: &str, b: &str) -> u8 {
    let (Some((na, la)), Some((nb, lb))) = (parse(a), parse(b)) else {
        return 3;
    };
    let steps = {
        let d = (i32::from(na) - i32::from(nb)).rem_euclid(12);
        d.min(12 - d) as u8
    };
    match (steps, la == lb) {
        (0, true) => 0,
        (0, false) | (1, true) => 1,
        (s, same) => s + u8::from(!same),
    }
}

/// The distance of two songs' keys, taking their second-best keys into account.
pub fn song_distance(a: &str, a_alt: &str, b: &str, b_alt: &str) -> u8 {
    let mut best = distance(a, b);
    for x in [a, a_alt] {
        for y in [b, b_alt] {
            if !x.is_empty() && !y.is_empty() {
                best = best.min(distance(x, y));
            }
        }
    }
    best
}

pub fn fit_name(distance: u8, same_letter_and_number: bool) -> &'static str {
    match distance {
        0 => "same",
        1 if same_letter_and_number => "same",
        1 => "related",
        _ => "clash",
    }
}

fn parse(code: &str) -> Option<(u8, char)> {
    let code = code.trim();
    let letter = code.chars().last()?.to_ascii_uppercase();
    if letter != 'A' && letter != 'B' {
        return None;
    }
    let number: u8 = code[..code.len() - 1].parse().ok()?;
    (1..=12).contains(&number).then_some((number, letter))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wheel() {
        assert_eq!(distance("10A", "10A"), 0);
        assert_eq!(distance("5B", "5A"), 1); // E flat major, C minor
        assert_eq!(distance("9A", "8A"), 1);
        assert_eq!(distance("12A", "1A"), 1);
        assert_eq!(distance("2B", "10A"), 5); // F sharp major, B minor
        assert_eq!(distance("5A", "11A"), 6); // a tritone apart
        assert_eq!(distance("x", "5A"), 3);
    }
}
