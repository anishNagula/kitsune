pub struct MatrixView<'a> {
    pub data: &'a [f32],
    pub rows: usize,
    pub cols: usize,
}

impl<'a> MatrixView<'a> {
    pub fn row(&self, row: usize) -> &[f32] {
        let start = row * self.cols;
        let end = start + self.cols;

        &self.data[start..end]
    }
}
