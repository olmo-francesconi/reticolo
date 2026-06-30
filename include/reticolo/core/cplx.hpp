#pragma once

#include <reticolo/core/hd.hpp>

// A minimal host/device complex number. std::complex is not usable in device
// code (its members are not __device__), so the per-site formulas that must run
// on BOTH the CPU backend and the CUDA kernels (the "one source of truth" rule)
// are written in terms of cplx<T> instead. It is standard-layout {re, im}, hence
// layout-compatible with std::complex<T> (which the standard guarantees is
// reinterpretable as T[2]) — a device buffer of cplx<T> flat-copies a host
// Lattice<std::complex<T>> with no transpose. RETICOLO_HD on every operation.

namespace reticolo {

template <class T>
struct cplx {
    T re;
    T im;

    RETICOLO_HD constexpr cplx() : re{T{0}}, im{T{0}} {}
    RETICOLO_HD constexpr cplx(T r, T i) : re{r}, im{i} {}

    RETICOLO_HD constexpr cplx& operator+=(cplx o) {
        re += o.re;
        im += o.im;
        return *this;
    }
    RETICOLO_HD constexpr cplx& operator-=(cplx o) {
        re -= o.re;
        im -= o.im;
        return *this;
    }
};

template <class T>
RETICOLO_HD constexpr cplx<T> operator+(cplx<T> a, cplx<T> b) {
    return {a.re + b.re, a.im + b.im};
}
template <class T>
RETICOLO_HD constexpr cplx<T> operator-(cplx<T> a, cplx<T> b) {
    return {a.re - b.re, a.im - b.im};
}
// Complex product.
template <class T>
RETICOLO_HD constexpr cplx<T> operator*(cplx<T> a, cplx<T> b) {
    return {(a.re * b.re) - (a.im * b.im), (a.re * b.im) + (a.im * b.re)};
}
// Real scaling (both orders).
template <class T>
RETICOLO_HD constexpr cplx<T> operator*(T s, cplx<T> a) {
    return {s * a.re, s * a.im};
}
template <class T>
RETICOLO_HD constexpr cplx<T> operator*(cplx<T> a, T s) {
    return {a.re * s, a.im * s};
}

// |z|² = re² + im².
template <class T>
RETICOLO_HD constexpr T norm2(cplx<T> z) {
    return (z.re * z.re) + (z.im * z.im);
}
template <class T>
RETICOLO_HD constexpr cplx<T> conj(cplx<T> z) {
    return {z.re, -z.im};
}
// Re(conj(a)·b) and Im(conj(a)·b) without the discarded half of the product.
template <class T>
RETICOLO_HD constexpr T re_conj_mul(cplx<T> a, cplx<T> b) {
    return (a.re * b.re) + (a.im * b.im);
}
template <class T>
RETICOLO_HD constexpr T im_conj_mul(cplx<T> a, cplx<T> b) {
    return (a.re * b.im) - (a.im * b.re);
}

static_assert(sizeof(cplx<double>) == 2 * sizeof(double) &&
              alignof(cplx<double>) == alignof(double));
static_assert(sizeof(cplx<float>) == 2 * sizeof(float) && alignof(cplx<float>) == alignof(float));

}  // namespace reticolo
