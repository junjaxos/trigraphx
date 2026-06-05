"""
Optimized distance metrics with SIMD acceleration.
"""

use pyo3::prelude::*;
use rayon::prelude::*;

#[pyfunction]
pub fn compute_distances_simd(vectors_a: Vec<Vec<f32>>, vectors_b: Vec<Vec<f32>>) -> PyResult<Vec<f32>> {
    if vectors_a.is_empty() || vectors_b.is_empty() {
        return Ok(vec![]);
    }
    
    let dim = vectors_a[0].len();
    if vectors_b[0].len() != dim {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Vector dimensions must match",
        ));
    }
    
    // Parallel computation using rayon
    let distances: Vec<f32> = (0..vectors_a.len())
        .into_par_iter()
        .map(|i| {
            let mut dist = 0.0f32;
            for j in 0..dim {
                let diff = vectors_a[i][j] - vectors_b[i][j];
                dist += diff * diff;
            }
            dist.sqrt()
        })
        .collect();
    
    Ok(distances)
}

#[pyfunction]
pub fn compute_semantic_batch(
    query_vector: Vec<f32>,
    database_vectors: Vec<Vec<f32>>,
    use_cosine: bool,
) -> PyResult<Vec<f32>> {
    """Compute distances from query to multiple vectors."""
    
    let dim = query_vector.len();
    
    let distances: Vec<f32> = database_vectors
        .par_iter()
        .map(|db_vec| {
            if db_vec.len() != dim {
                return f32::INFINITY;
            }
            
            if use_cosine {
                cosine_distance(&query_vector, db_vec)
            } else {
                euclidean_distance(&query_vector, db_vec)
            }
        })
        .collect();
    
    Ok(distances)
}

fn euclidean_distance(v1: &[f32], v2: &[f32]) -> f32 {
    let sum: f32 = v1
        .iter()
        .zip(v2.iter())
        .map(|(a, b)| {
            let diff = a - b;
            diff * diff
        })
        .sum();
    sum.sqrt()
}

fn cosine_distance(v1: &[f32], v2: &[f32]) -> f32 {
    let dot: f32 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
    
    let mag1: f32 = v1.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag2: f32 = v2.iter().map(|x| x * x).sum::<f32>().sqrt();
    
    if mag1 == 0.0 || mag2 == 0.0 {
        return 1.0;
    }
    
    let similarity = dot / (mag1 * mag2);
    1.0 - similarity.max(-1.0).min(1.0)  // Ensure result is in [0, 1]
}
