//! Deferred normalization for field arithmetic.
//!
//! This module provides the [`DeferredField`] trait and a wide [`Product`]
//! accumulator. Together they enable accumulating multiple unreduced
//! Montgomery products before performing a single expensive reduction.
//! This is useful for operations like inner products where many
//! multiplications feed into a sum.

use core::fmt::Debug;

use crate::arithmetic::{adc, mac};

/// A trait for fields that support deferred reduction of products.
///
/// Instead of reducing each multiplication result immediately, callers
/// accumulate products into an [`Accumulator`](Self::Accumulator) via
/// [`mul_accumulate`](Self::mul_accumulate) and
/// [`square_accumulate`](Self::square_accumulate), then perform a single
/// reduction at the end with [`reduce`](Self::reduce).
pub trait DeferredField: ff::Field {
    /// A wide accumulator for unreduced products.
    type Accumulator: Copy + Clone + Debug + Default;

    /// Multiplies `a` by `b` and adds the result into `acc`.
    fn mul_accumulate(acc: &mut Self::Accumulator, a: &Self, b: &Self);

    /// Squares `a` and adds the result into `acc`.
    fn square_accumulate(acc: &mut Self::Accumulator, a: &Self);

    /// Reduces the accumulator to a canonical field element.
    fn reduce(acc: Self::Accumulator) -> Self;
}

/// A wide accumulator for unreduced Montgomery products over field `F`.
///
/// This stores a running sum of 512-bit products with a 64-bit carry for
/// overflow beyond 512 bits. Products are added internally by
/// [`DeferredField::mul_accumulate`] and [`DeferredField::square_accumulate`].
///
/// Call [`DeferredField::reduce`] to fold the carry back into range and
/// perform Montgomery reduction.
#[derive(Clone, Copy, Debug)]
pub struct Product<F> {
    pub(crate) limbs: [u64; 8],
    pub(crate) carry: u64,
    _marker: core::marker::PhantomData<F>,
}

impl<F> Default for Product<F> {
    fn default() -> Self {
        Self::ZERO
    }
}

impl<F> Product<F> {
    /// The zero (additive identity) accumulator.
    pub const ZERO: Self = Product {
        limbs: [0; 8],
        carry: 0,
        _marker: core::marker::PhantomData,
    };

    /// Folds `carry` (bits 512+) and `limbs[7]` (bits 448–511) into the lower
    /// 448 bits using precomputed residues of $2^{448}$ and $2^{512}$ modulo
    /// the field prime.
    ///
    /// The result fits in 8 limbs with value $< 2^{449} < Rp$, safe for
    /// Montgomery reduction.
    #[cfg_attr(not(feature = "uninline-portable"), inline)]
    pub(crate) fn partial_reduce(&self, b448: &[u64; 4], r2: &[u64; 4]) -> [u64; 8] {
        let b7 = self.limbs[7];
        let b8 = self.carry;

        // Compute b7 * b448 (5 limbs)
        let (t0, c) = mac(0, b7, b448[0], 0);
        let (t1, c) = mac(0, b7, b448[1], c);
        let (t2, c) = mac(0, b7, b448[2], c);
        let (t3, c) = mac(0, b7, b448[3], c);
        let t4 = c;

        // Accumulate b8 * r2
        let (t0, c) = mac(t0, b8, r2[0], 0);
        let (t1, c) = mac(t1, b8, r2[1], c);
        let (t2, c) = mac(t2, b8, r2[2], c);
        let (t3, c) = mac(t3, b8, r2[3], c);
        let (t4, t5) = adc(t4, 0, c);
        debug_assert!(
            t5 == 0,
            "folding term overflow: t4 + carry does not fit in 64 bits"
        );

        // Add to lower 7 limbs
        let (d0, c) = adc(self.limbs[0], t0, 0);
        let (d1, c) = adc(self.limbs[1], t1, c);
        let (d2, c) = adc(self.limbs[2], t2, c);
        let (d3, c) = adc(self.limbs[3], t3, c);
        let (d4, c) = adc(self.limbs[4], t4, c);
        let (d5, c) = adc(self.limbs[5], 0, c);
        let (d6, c) = adc(self.limbs[6], 0, c);
        let (d7, _) = adc(0, 0, c);

        // B448 < 2^253 and r2 < 2^252, so the folding term
        // b7 * B448 + b8 * r2 < 2^317 + 2^316 < 2^318.
        // The full value is < 2^448 + 2^318 < 2^449, so d7 is at most 1.
        debug_assert!(d7 <= 1);

        [d0, d1, d2, d3, d4, d5, d6, d7]
    }
}
