use std::fmt;

#[derive(Debug)]
pub enum TensorError {
    ShapeMismatch,
    InvalidShape,
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TensorError::ShapeMismatch => write!(f, "tensor shape mismatch"),
            TensorError::InvalidShape => write!(f, "invalid tensor shape"),
        }
    }
}

impl std::error::Error for TensorError {}
