"""
Main Rust extension module for MRMRS acceleration.
Exports:
- LSH indexing for sublinear search
- Quantized vector distance computation
- Parallel metric calculation
"""

// Build with: maturin develop

use pyo3::prelude::*;
use ndarray::Array2;
use std::collections::HashMap;

mod lsh;
mod metrics;
mod quantize;

use lsh::LSHIndex;
use metrics::{compute_distances_simd, compute_semantic_batch};
use quantize::quantize_vectors_batch;

#[pymodule]
fn mrmrs_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<LSHIndex>()?;
    m.add_function(wrap_pyfunction!(compute_distances_simd, m)?)?;
    m.add_function(wrap_pyfunction!(compute_semantic_batch, m)?)?;
    m.add_function(wrap_pyfunction!(quantize_vectors_batch, m)?)?;
    
    Ok(())
}
