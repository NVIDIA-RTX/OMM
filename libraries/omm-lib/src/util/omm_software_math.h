/*
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef OMM_SOFTWARE_MATH_H
#define OMM_SOFTWARE_MATH_H

// Shared C++/HLSL implementation. All arithmetic in sqrt and division is uint32;
// the host and shader differ only in how they bit-cast their float arguments.
#ifdef __cplusplus
namespace omm::util
{
using uint = unsigned int;
static_assert(sizeof(uint) == 4 && sizeof(float) == sizeof(uint) && sizeof(int) == sizeof(uint));
// MSVC, Clang and GCC use this primitive for C++20 std::bit_cast. Keep standard
// headers out of the shared file: ShaderMake scans even inactive includes.
inline uint asuint(float value) { return __builtin_bit_cast(uint, value); }
inline float asfloat(uint value) { return __builtin_bit_cast(float, value); }
#endif

// Correctly rounded binary32 square root using only 32-bit integer operations.
// Restoring recurrence: Li and Chu, ICCD'97, Section 2.1 (Restoring Square Root):
// https://yamin.cis.k.hosei.ac.jp/papers/ICCD97.pdf
// Here it consumes a 48-bit radicand in 24 steps; the exact remainder supplies
// binary32 round-to-nearest. See the midpoint derivation after the loop.
inline float OmmSoftwareSqrt(float value)
{
    const uint bits = asuint(value);
    const uint magnitude = bits & 0x7FFFFFFFu;

    if (magnitude == 0u)
        return asfloat(bits);
    // Negative nonzero values and all NaN payloads produce one quiet NaN.
    if ((bits & 0x80000000u) != 0u || magnitude > 0x7F800000u)
        return asfloat(0x7FC00000u);
    if (magnitude == 0x7F800000u)
        return asfloat(bits);

    const uint exponentField = magnitude >> 23u;
    uint significand = magnitude & 0x007FFFFFu;
    int exponent = int(exponentField) - 127;
    if (exponentField == 0u)
    {
        exponent = -126;
        // A nonzero subnormal needs at most 23 shifts to restore the hidden bit.
        while ((significand & 0x00800000u) == 0u)
        {
            significand <<= 1u;
            --exponent;
        }
    }
    else
    {
        significand |= 0x00800000u;
    }

    // sqrt(value) has significand sqrt(significand * 2^(23 + exponentParity)).
    // The integer radicand is at most 48 bits; keep it as a 16:32-bit pair.
    const uint radicandShift = 23u + (uint(exponent) & 1u);
    uint radicandHigh = significand >> (32u - radicandShift);
    uint radicandLow = significand << radicandShift;
    uint root = 0u;
    uint remainder = 0u;
    for (uint digit = 0u; digit < 24u; ++digit)
    {
        const uint pair = radicandHigh >> 14u;
        radicandHigh = ((radicandHigh << 2u) | (radicandLow >> 30u)) & 0xFFFFu;
        radicandLow <<= 2u;
        // Consume two radicand bits per root bit. The partial remainder is below
        // 2*root+1, so even its shifted value fits in 26 bits in all 24 steps.
        remainder = (remainder << 2u) | pair;
        const uint trial = (root << 2u) | 1u;
        root <<= 1u;
        if (remainder >= trial)
        {
            remainder -= trial;
            root |= 1u;
        }
    }

    // radicand = root^2 + remainder. The midpoint (root+0.5)^2 is
    // root^2+root+0.25. An integer radicand cannot be a midpoint, so rounding
    // upward exactly when remainder > root implements round-to-nearest-even.
    if (remainder > root)
        ++root;
    // Every positive binary32 square root is normal, even for input 2^-149.
    // Adding 254 before halving avoids signed rounding/shift ambiguity.
    uint resultExponent = uint(exponent + 254) >> 1u;
    if (root == 0x01000000u)
    {
        root >>= 1u;
        ++resultExponent;
    }
    return asfloat((resultExponent << 23u) | (root & 0x007FFFFFu));
}

// Correctly rounded binary32 division using only 32-bit integer operations.
// Restoring division: Structure of Computer Systems, Section 3.3.1:
// https://users.utcluj.ro/~baruch/en/media/book_ssce/SSCE-Basic-Division.pdf
// This compare-before-subtract variant generates fractional quotient bits from
// normalized significands, then rounds using the exact final remainder.
// Preserve subnormal operands/results, signed zero, and canonicalize all NaNs.
inline float OmmSoftwareDivide(float numerator, float denominator)
{
    const uint numeratorBits = asuint(numerator);
    const uint denominatorBits = asuint(denominator);
    const uint sign = (numeratorBits ^ denominatorBits) & 0x80000000u;
    const uint a = numeratorBits & 0x7FFFFFFFu;
    const uint b = denominatorBits & 0x7FFFFFFFu;

    if (a > 0x7F800000u || b > 0x7F800000u ||
        (a == 0u && b == 0u) || (a == 0x7F800000u && b == 0x7F800000u))
        return asfloat(0x7FC00000u);
    if (a == 0x7F800000u || b == 0u)
        return asfloat(sign | 0x7F800000u);
    if (a == 0u || b == 0x7F800000u)
        return asfloat(sign);

    uint significandA = a & 0x007FFFFFu;
    uint significandB = b & 0x007FFFFFu;
    int exponentA = int(a >> 23u) - 127;
    int exponentB = int(b >> 23u) - 127;
    if ((a >> 23u) == 0u)
    {
        exponentA = -126;
        while ((significandA & 0x00800000u) == 0u)
        {
            significandA <<= 1u;
            --exponentA;
        }
    }
    else
        significandA |= 0x00800000u;
    if ((b >> 23u) == 0u)
    {
        exponentB = -126;
        while ((significandB & 0x00800000u) == 0u)
        {
            significandB <<= 1u;
            --exponentB;
        }
    }
    else
        significandB |= 0x00800000u;

    int exponent = exponentA - exponentB;
    if (significandA < significandB)
    {
        significandA <<= 1u;
        --exponent;
    }
    // Quotient is in [1,2). At or below half minsub, ties round to even zero.
    if (exponent > 127)
        return asfloat(sign | 0x7F800000u);
    if (exponent < -150)
        return asfloat(sign);
    if (exponent == -150)
        return asfloat(sign | (significandA > significandB ? 1u : 0u));

    const bool subnormal = exponent < -126;
    const uint fractionalBits = subnormal ? uint(exponent + 149) : 23u;
    uint quotient = 1u;
    uint remainder = significandA - significandB;
    for (uint digit = 0u; digit < fractionalBits; ++digit)
    {
        // remainder < significandB < 2^24, so the shift fits in 25 bits.
        remainder <<= 1u;
        quotient <<= 1u;
        if (remainder >= significandB)
        {
            remainder -= significandB;
            quotient |= 1u;
        }
    }
    // The discarded fraction is remainder/significandB: compare twice the
    // remainder with the divisor, rounding exact halfway cases to an even bit.
    const uint twiceRemainder = remainder << 1u;
    if (twiceRemainder > significandB ||
        (twiceRemainder == significandB && (quotient & 1u) != 0u))
        ++quotient;
    // A subnormal carry to 0x00800000 directly encodes the smallest normal.
    if (subnormal)
        return asfloat(sign | quotient);
    if (quotient == 0x01000000u)
    {
        quotient >>= 1u;
        ++exponent;
    }
    return asfloat(sign | (uint(exponent + 127) << 23u) | (quotient & 0x007FFFFFu));
}

#ifdef __cplusplus
} // namespace omm::util
#endif

#endif // OMM_SOFTWARE_MATH_H
