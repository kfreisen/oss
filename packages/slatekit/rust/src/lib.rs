//! PyO3 bindings for the slatekit kernel.
//!
//! This file holds only the `#[pymodule]` registration and thin `#[pyfunction]`
//! shims. All real logic lives in sibling modules so it can be unit-tested from
//! Rust and measured by `cargo llvm-cov` without a Python interpreter in the loop.

use pyo3::prelude::*;

pub mod roster;
pub mod simd;

/// Returns the SIMD instruction set the kernel selected at runtime.
///
/// Exposed because wheels are built without `target-cpu=native` and dispatch
/// happens per-process: users debugging a performance report need to be able to
/// see which path their machine actually took.
#[pyfunction]
fn active_isa() -> &'static str {
    simd::active_isa()
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(active_isa, m)?)?;
    Ok(())
}
