use super::*;
use ndarray::array;

fn assert_close(actual: &Array2<f64>, expected: &Array2<f64>, atol: f64, rtol: f64) {
    assert_eq!(actual.dim(), expected.dim());
    for ((row, col), &want) in expected.indexed_iter() {
        let got = actual[[row, col]];
        let tolerance = atol + rtol * want.abs();
        assert!(
            (got - want).abs() <= tolerance,
            "entry ({row}, {col}): got {got}, expected {want}, tolerance {tolerance}"
        );
    }
}

#[test]
fn su_single_step_is_input_jacobian() {
    let a = vec![array![[2.0, -3.0], [4.0, 0.5]]];
    let b = vec![array![[1.0, 2.0], [-2.0, 3.0]]];
    assert_close(&build_Su(&a, &b), &b[0], 0.0, 0.0);
}

#[test]
fn su_noncommuting_blocks_have_later_transitions_on_left() {
    let a = vec![
        array![[2.0, 0.0], [0.0, 3.0]], // A[0] must not affect input sensitivities.
        array![[1.0, 1.0], [0.0, 1.0]],
        array![[1.0, 0.0], [1.0, 1.0]],
    ];
    let b = vec![
        array![[1.0], [2.0]],
        array![[-1.0], [1.0]],
        array![[2.0], [-1.0]],
    ];
    // Rows stack x[1], x[2], x[3]; columns stack u[0], u[1], u[2].
    // A[2] A[1] B[0] = [3, 5], whereas A[1] A[2] B[0] = [4, 3].
    let expected = array![
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, -1.0, 0.0],
        [2.0, 1.0, 0.0],
        [3.0, -1.0, 2.0],
        [5.0, 0.0, -1.0],
    ];
    assert_close(&build_Su(&a, &b), &expected, 0.0, 0.0);
}

// Independent oracle: advance the state recurrence with scalar sums, without
// forming transition products or calling build_Su.
fn propagate_perturbations(a: &[Array2<f64>], b: &[Array2<f64>], du: &Array1<f64>) -> Array1<f64> {
    let n = a[0].nrows();
    let m = b[0].ncols();
    let mut dx = vec![0.0; n]; // The initial state is fixed: delta_x[0] = 0.
    let mut stacked = Array1::zeros(n * a.len());
    for k in 0..a.len() {
        let mut next = vec![0.0; n];
        for row in 0..n {
            for col in 0..n {
                next[row] += a[k][[row, col]] * dx[col];
            }
            for col in 0..m {
                next[row] += b[k][[row, col]] * du[k * m + col];
            }
            stacked[k * n + row] = next[row];
        }
        dx = next;
    }
    stacked
}

#[test]
fn su_matches_independent_time_varying_perturbation_propagation() {
    let a = vec![
        array![[2.0, -0.5], [0.3, 1.0]],
        array![[1.0, 1.0], [0.0, 1.0]],
        array![[1.0, 0.0], [1.0, 1.0]],
        array![[0.9, -0.4], [0.3, 1.2]],
    ];
    let b = vec![
        array![[1.0, 0.2], [-0.3, 1.0]],
        array![[-0.5, 1.0], [2.0, 0.4]],
        array![[0.7, -1.0], [0.2, 0.9]],
        array![[1.1, 0.3], [-0.8, 0.6]],
    ];
    let su = build_Su(&a, &b);
    let mut expected = Array2::zeros(su.dim());
    // Every basis input checks every block, including diagonal and causal zeros.
    for col in 0..su.ncols() {
        let mut du = Array1::zeros(su.ncols());
        du[col] = 1.0;
        expected
            .column_mut(col)
            .assign(&propagate_perturbations(&a, &b, &du));
    }
    assert_close(&su, &expected, 1e-12, 1e-12);

    // Also check simultaneous perturbations across both controls and all steps.
    let du = array![0.2, -0.4, 0.7, 0.1, -0.3, 0.9, -0.6, 0.5];
    let predicted = su.dot(&du).insert_axis(ndarray::Axis(1));
    let propagated = propagate_perturbations(&a, &b, &du).insert_axis(ndarray::Axis(1));
    assert_close(&predicted, &propagated, 1e-12, 1e-12);
}

#[test]
fn su_matches_nonlinear_rollout_finite_differences() {
    // Nonzero attitude, velocities, angular rates, and varying gimbal/thrust
    // keep this away from hover and the zero-thrust/zero-angular-rate branches.
    let mut x0 = array![
        1.2, -0.7, 8.0, 0.12, -0.18, 0.08, 0.97, 0.4, -0.2, -0.6, 0.3, -0.25, 0.2
    ];
    let q_norm = x0.slice(s![3..7]).mapv(|v: f64| v * v).sum().sqrt();
    x0.slice_mut(s![3..7]).mapv_inplace(|v| v / q_norm);
    let controls = vec![
        array![0.08, -0.06, 810.0],
        array![-0.04, 0.09, 760.0],
        array![0.10, 0.03, 850.0],
        array![-0.07, -0.08, 730.0],
        array![0.02, 0.05, 800.0],
    ];
    let mass = 80.0;
    let moi = array![[70.0, 0.8, -0.3], [0.8, 65.0, 0.2], [-0.3, 0.2, 4.0]];
    let dt = 0.08;
    let xs = rollout(&x0, &controls, mass, &moi, dt);
    let (a, b) = linearize_trajectory(&xs, &controls, mass, &moi, dt);
    let commutator = a[2].dot(&a[1]) - a[1].dot(&a[2]);
    assert!(commutator.iter().any(|v| v.abs() > 1e-4));
    let su = build_Su(&a, &b);
    let n = x0.len();
    let m = controls[0].len();

    // Different steps from compute_jacobian's 1e-6; check two scales to avoid
    // relying on a single finite-difference step. Thrust has different units.
    for angle_step in [1e-4, 1e-5] {
        let mut finite_difference = Array2::zeros(su.dim());
        for col in 0..su.ncols() {
            let j = col / m;
            let input = col % m;
            let step = if input == 2 {
                100.0 * angle_step
            } else {
                angle_step
            };
            let mut plus = controls.clone();
            let mut minus = controls.clone();
            plus[j][input] += step;
            minus[j][input] -= step;
            // Perturb the actual nonlinear rollout, independently of Jacobians.
            let xs_plus = rollout(&x0, &plus, mass, &moi, dt);
            let xs_minus = rollout(&x0, &minus, mass, &moi, dt);
            for k in 0..controls.len() {
                for row in 0..n {
                    finite_difference[[k * n + row, col]] =
                        (xs_plus[k + 1][row] - xs_minus[k + 1][row]) / (2.0 * step);
                }
            }
        }
        let max_error = (&su - &finite_difference)
            .iter()
            .fold(0.0_f64, |max, error| max.max(error.abs()));
        eprintln!(
            "nonlinear Su finite difference: angle step={angle_step:e}, max error={max_error:e}"
        );
        assert_close(&su, &finite_difference, 2e-6, 2e-5);
    }
}
