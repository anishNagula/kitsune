use crate::error::TensorError;

#[derive(Debug, Clone)]
pub struct Tensor {
    data: Vec<f32>,
    shape: Vec<usize>,
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Result<Self, TensorError> {
        let expected: usize = shape.iter().product();

        if expected != data.len() {
            return Err(TensorError::InvalidShape);
        }

        Ok(Self {data, shape })
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        let size: usize = shape.iter().product();

        Self {
            data: vec![0.0; size],
            shape,
        }
    }

    pub fn ones(shape: Vec<usize>) -> Self {
        let size: usize = shape.iter().product();

        Self {
            data: vec![1.0; size],
            shape,
        }
    }

    pub fn reshape(&self, shape: Vec<usize>) -> Result<Self, TensorError> {
        let expected: usize = shape.iter().product();

        if expected != self.data.len() {
            return Err(TensorError::InvalidShape);
        }

        Ok(Self {
            data: self.data.clone(),
            shape,
        })
    }

    pub fn transpose(&self) -> Result<Self, TensorError> {
        if self.shape.len() != 2 {
            return Err(TensorError::InvalidShape);
        }

        let rows = self.shape[0];
        let cols = self.shape[1];

        let mut result = vec![0.0; self.data.len()];

        for r in 0..rows {
            for c in 0..cols {
                result[c * rows + r] = self.data[r * cols + c];
            }
        }

        Ok(Self {
            data: result,
            shape: vec![cols, rows],
        })
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }
}
