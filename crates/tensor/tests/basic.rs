use tensor::Tensor;

#[test]
fn create_tensor() {
    let t = Tensor::new(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
    )
    .unwrap();

    assert_eq!(t.shape(), &[2, 2]);
    assert_eq!(t.len(), 4);
}

#[test]
fn zeros_tensor() {
    let t = Tensor::zeros(vec![3, 3]);

    assert_eq!(t.len(), 9);

    for value in t.data() {
        assert_eq!(*value, 0.0);
    }
}

#[test]
fn ones_tensor() {
    let t = Tensor::ones(vec![2, 2]);

    for value in t.data() {
        assert_eq!(*value, 1.0);
    }
}

#[test]
fn reshape_tensor() {
    let t = Tensor::new(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2,2],
    )
    .unwrap();

    let reshaped = t.reshape(vec![4]).unwrap();

    assert_eq!(reshaped.shape(), &[4]);
    assert_eq!(reshaped.len(), 4);
}

#[test]
fn transpose_tensor() {
    let t = Tensor::new(
        vec![
            1.0, 2.0,
            3.0, 4.0,
        ],
        vec![2, 2],
    )
    .unwrap();

    let transposed = t.transpose().unwrap();

    assert_eq!(
        transposed.data(),
        &[1.0, 3.0, 2.0, 4.0]
    );

    assert_eq!(transposed.shape(), &[2,2]);
}
