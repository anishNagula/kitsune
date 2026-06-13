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

#[test]
fn add_tensors() {
    let a = Tensor::new(
        vec![1.0, 2.0, 3.0],
        vec![3],
    )
    .unwrap();

    let b = Tensor::new(
        vec![4.0, 5.0, 6.0],
        vec![3],
    )
    .unwrap();

    let c = a.add(&b).unwrap();

    assert_eq!(
        c.data(),
        &[5.0, 7.0, 9.0]
    );
}

#[test]
fn sub_tensors() {
    let a = Tensor::new(
        vec![5.0, 7.0, 9.0],
        vec![3],
    )
    .unwrap();

    let b = Tensor::new(
        vec![4.0, 5.0, 6.0],
        vec![3],
    )
    .unwrap();

    let c = a.sub(&b).unwrap();

    assert_eq!(
        c.data(),
        &[1.0, 2.0, 3.0]
    );
}

#[test]
fn mul_tensors() {
    let a = Tensor::new(
        vec![1.0, 2.0, 3.0],
        vec![3],
    )
    .unwrap();

    let b = Tensor::new(
        vec![4.0, 5.0, 6.0],
        vec![3],
    )
    .unwrap();

    let c = a.mul(&b).unwrap();

    assert_eq!(
        c.data(),
        &[4.0, 10.0, 18.0]
    );
}
