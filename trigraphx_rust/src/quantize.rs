"""
Vector quantization: float32 -> int8 for 75% memory savings.
"""

use pyo3::prelude::*;
use rayon::prelude::*;

#[pyfunction]
pub fn quantize_vectors_batch(
    vectors: Vec<Vec<f32>>,
    scale: f32,
) -> PyResult<Vec<Vec<i8>>> {
    """Quantize floating-point vectors to int8."""
    
    let quantized: Vec<Vec<i8>> = vectors
        .par_iter()
        .map(|vec| {
            vec.iter()
                .map(|&x| {
                    let scaled = (x / scale * 127.0) as i32;
                    scaled.max(-128).min(127) as i8
                })
                .collect()
        })
        .collect();
    
    Ok(quantized)
}

#[pyfunction]
pub fn dequantize_vectors_batch(
    vectors: Vec<Vec<i8>>,
    scale: f32,
) -> PyResult<Vec<Vec<f32>>> {
    """Dequantize int8 vectors back to float32."""
    
    let dequantized: Vec<Vec<f32>> = vectors
        .par_iter()
        .map(|vec| {
            vec.iter()
                .map(|&x| (x as f32 / 127.0) * scale)
                .collect()
        })
        .collect();
    
    Ok(dequantized)
}

#[pyfunction]
pub fn compute_quantized_distances(
    query: Vec<i8>,
    database: Vec<Vec<i8>>,
) -> PyResult<Vec<f32>> {
    """Compute distances using quantized vectors (int8 dot product)."""
    
    let distances: Vec<f32> = database
        .par_iter()
        .map(|db_vec| {
            let dot: i32 = query
                .iter()
                .zip(db_vec.iter())
                .map(|(a, b)| (*a as i32) * (*b as i32))
                .sum();
            (dot as f32).abs()  // Use absolute dot product as distance proxy
        })
        .collect();
    
    Ok(distances)
}
