//! Helpers for writing into the UI band every env appends to its field of
//! view (see [`UI_HEIGHT`](super::UI_HEIGHT)).

use ndarray::ArrayViewMut1;

use crate::vocab::VocabId;

/// Writes `value` into `row` as a right-aligned run of digit tokens, most
/// significant digit leftmost, leaving the cells left of the number as they
/// are. Right alignment keeps each place value in a fixed column, so the ones
/// digit is always the last cell. A value too wide for the row saturates to
/// all nines rather than dropping its leading digits.
///
/// `digits[d]` is the token drawing digit `d`; envs pass their own vocab's
/// entries for [`UI_DIGITS`](crate::symbols::UI_DIGITS).
pub fn write_number<T: Copy + Into<VocabId>>(
    mut row: ArrayViewMut1<'_, VocabId>,
    value: u32,
    digits: &[T; 10],
) {
    let width = row.len();
    if width == 0 {
        return;
    }
    let max = 10u64.saturating_pow(width as u32).saturating_sub(1);
    let mut rest = u64::from(value).min(max);

    for cell in (0..width).rev() {
        row[cell] = digits[(rest % 10) as usize].into();
        rest /= 10;
        if rest == 0 {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    const BLANK: VocabId = 99;
    const DIGITS: [VocabId; 10] = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19];

    fn written(width: usize, value: u32) -> Vec<VocabId> {
        let mut row = Array1::from_elem(width, BLANK);
        write_number(row.view_mut(), value, &DIGITS);
        row.to_vec()
    }

    #[test]
    fn numbers_are_right_aligned_with_the_most_significant_digit_first() {
        assert_eq!(written(5, 407), vec![BLANK, BLANK, 14, 10, 17]);
    }

    #[test]
    fn zero_is_one_digit() {
        assert_eq!(written(3, 0), vec![BLANK, BLANK, 10]);
    }

    #[test]
    fn values_too_wide_saturate_to_nines() {
        assert_eq!(written(2, 12345), vec![19, 19]);
        assert_eq!(written(0, 7), Vec::<VocabId>::new());
    }
}
