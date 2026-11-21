use super::*;
use ndarray::array;

const MASS: f64 = 80.0;
const DT: f64 = 0.2;

fn initial_state() -> Array1<f64> {
    let mut x = Array1::zeros(13);
    x[2] = 8.0;
    x[6] = 1.0;
    x
}

fn inertia() -> Array2<f64> {
    Array2::from_diag(&array![6.65, 6.65, 2.3])
}

fn bounds() -> (Array1<f64>, Array1<f64>) {
    let angle = 15.0_f64.to_radians();
    (array![-angle, -angle, 300.0], array![angle, angle, 1200.0])
}

fn shifted(u: &[Array1<f64>], d: &Array1<f64>, step: f64) -> Vec<Array1<f64>> {
    u.iter()
        .enumerate()
        .map(|(k, value)| value + step * &d.slice(s![3 * k..3 * (k + 1)]))
        .collect()
}

fn close(actual: f64, expected: f64, tolerance: f64) {
    assert!((actual - expected).abs() <= tolerance,
            "actual={actual:.10e}, expected={expected:.10e}, tolerance={tolerance:.2e}");
}

#[test]
fn absolute_control_cost_has_linear_term_at_nonzero_nominal() {
    let x = initial_state();
    let moi = inertia();
    let u = vec![array![0.05, -0.02, 800.0]];
    let xs = rollout(&x, &u, MASS, &moi, DT);
    let (a, b) = linearize_trajectory(&xs, &u, MASS, &moi, DT);
    let (lo, hi) = bounds();
    let q = Array2::zeros((13, 13));
    let r = Array2::from_diag(&array![50.0, 50.0, 0.005]);
    let (h, g, ..) = assemble_qp_increment(&xs, &u, &xs, &a, &b, &q, &r, &q, &lo, &hi);

    let d = array![0.0, 0.0, 1.0];
    let cost = |step| true_cost(&x, &shifted(&u, &d, step), &xs, &q, &r, &q, MASS, &moi, DT);
    let eps = 0.1;
    let derivative = (cost(eps) - cost(-eps)) / (2.0 * eps);
    let curvature = (cost(eps) + cost(-eps) - 2.0 * cost(0.0)) / (eps * eps);
    eprintln!("absolute thrust: true slope={derivative:.9}, QP slope={:.9}, true curvature={curvature:.9}, QP curvature={:.9}",
              g.dot(&d), d.dot(&h.dot(&d)));
    close(derivative, 8.0, 1e-8);
    close(g.dot(&d), 0.0, 1e-12);
    close(curvature, 0.01, 1e-8);
    close(d.dot(&h.dot(&d)), 0.005 + 1e-8, 1e-12);
}

#[test]
fn terminal_stage_weight_is_dropped_in_the_qp() {
    let x = initial_state();
    let moi = inertia();
    let u = vec![array![0.0, 0.0, 800.0]];
    let xs = rollout(&x, &u, MASS, &moi, DT);
    let mut reference = xs.clone();
    reference[1][9] += 1.0;
    let (a, b) = linearize_trajectory(&xs, &u, MASS, &moi, DT);
    let (lo, hi) = bounds();
    let mut q = Array2::zeros((13, 13));
    let mut qn = q.clone();
    q[[9, 9]] = 1000.0;
    qn[[9, 9]] = 250.0;
    let r = Array2::zeros((3, 3));
    let (h, g, _, _, _, stacked_q, _) =
        assemble_qp_increment(&xs, &u, &reference, &a, &b, &q, &r, &qn, &lo, &hi);
    let d = array![0.0, 0.0, 1.0];
    let cost = |step| true_cost(&x, &shifted(&u, &d, step), &reference, &q, &r, &qn, MASS, &moi, DT);
    let eps = 0.1;
    let derivative = (cost(eps) - cost(-eps)) / (2.0 * eps);
    let curvature = (cost(eps) + cost(-eps) - 2.0 * cost(0.0)) / (eps * eps);
    eprintln!("terminal velocity: true slope={derivative:.9}, 2*QP slope={:.9}, true curvature={curvature:.9}, 2*QP curvature={:.9}",
              2.0 * g.dot(&d), 2.0 * d.dot(&h.dot(&d)));
    close(stacked_q[[9, 9]], 250.0, 0.0);
    close(derivative, -6.25, 1e-8);
    close(2.0 * g.dot(&d), -1.25, 1e-6);
    close(curvature, 0.015625, 1e-8);
    close(2.0 * d.dot(&h.dot(&d)), 0.003125 + 2e-8, 1e-8);
}

#[test]
fn nonlinear_directional_derivatives_match_the_corrected_half_cost_model() {
    let mut x = initial_state();
    x[4] = 0.04;
    x[10] = 0.08;
    x[11] = -0.05;
    let moi = inertia();
    let u = vec![
        array![0.02, -0.03, 810.0],
        array![-0.01, 0.04, 760.0],
        array![0.06, -0.01, 850.0],
        array![-0.03, -0.05, 730.0],
    ];
    let xs = rollout(&x, &u, MASS, &moi, DT);
    let mut reference = xs.clone();
    for (k, target) in reference.iter_mut().enumerate().skip(1) {
        target[2] += 0.25 * k as f64;
        target[9] += 0.5;
    }
    let (a, b) = linearize_trajectory(&xs, &u, MASS, &moi, DT);
    let (lo, hi) = bounds();
    let q = 150.0 * Array2::eye(13);
    let qn = 300.0 * Array2::eye(13);
    let r = Array2::from_diag(&array![50.0, 50.0, 0.005]);
    let (_, g, _, _, su, _, _) =
        assemble_qp_increment(&xs, &u, &reference, &a, &b, &q, &r, &qn, &lo, &hi);

    for d in [
        array![0.1, -0.1, 3.0, -0.1, 0.0, 1.0, 0.05, 0.02, -2.0, 0.0, -0.1, 4.0],
        array![0.0, 0.0, 0.0, 0.0, 0.07, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0],
    ] {
        let dx = su.dot(&d);
        let mut corrected_half_slope = 0.0;
        for k in 0..u.len() {
            let residual = &xs[k + 1] - &reference[k + 1];
            let dx_k = dx.slice(s![13 * k..13 * (k + 1)]);
            corrected_half_slope += dx_k.dot(&q.dot(&residual))
                + d.slice(s![3 * k..3 * (k + 1)]).dot(&r.dot(&u[k]));
            if k + 1 == u.len() {
                corrected_half_slope += dx_k.dot(&qn.dot(&residual));
            }
        }
        let cost = |step| true_cost(&x, &shifted(&u, &d, step), &reference, &q, &r, &qn, MASS, &moi, DT);
        for eps in [1e-3, 1e-4] {
            let fd = (cost(eps) - cost(-eps)) / (2.0 * eps);
            eprintln!("nonlinear eps={eps:e}: true slope={fd:.9}, 2*QP slope={:.9}, corrected={:.9}",
                      2.0 * g.dot(&d), 2.0 * corrected_half_slope);
            close(fd, 2.0 * corrected_half_slope, 5e-4);
            assert!((fd - 2.0 * g.dot(&d)).abs() > 0.1);
        }
    }
}

#[test]
fn pgd_descent_can_be_true_cost_ascent_and_line_search_retains_nominal() {
    let x = initial_state();
    let moi = inertia();
    let u = vec![array![0.0, 0.0, 800.0]];
    let xs = rollout(&x, &u, MASS, &moi, DT);
    let mut reference = xs.clone();
    reference[1][9] += 0.4;
    let (a, b) = linearize_trajectory(&xs, &u, MASS, &moi, DT);
    // Hold the gimbals fixed to isolate thrust and make the 1-step cost quadratic.
    let lo = array![0.0, 0.0, 300.0];
    let hi = array![0.0, 0.0, 1200.0];
    let q = Array2::zeros((13, 13));
    let mut qn = q.clone();
    qn[[9, 9]] = 1000.0;
    let r = Array2::from_diag(&array![50.0, 50.0, 0.005]);
    let (h, g, dlo, dhi, ..) =
        assemble_qp_increment(&xs, &u, &reference, &a, &b, &q, &r, &qn, &lo, &hi);
    let du = solve_qp_pgd(&h, &g, &dlo, &dhi, &Array1::zeros(3), 20, 1.0);
    let cost = |step| true_cost(&x, &shifted(&u, &du, step), &reference, &q, &r, &qn, MASS, &moi, DT);
    let fd = (cost(1e-5) - cost(-1e-5)) / 2e-5;
    eprintln!("PGD thrust increment={:.9} N, model slope={:.9}, true slope={fd:.9}",
              du[2], g.dot(&du));
    assert!(du[2] > 0.0 && g.dot(&du) < 0.0 && fd > 0.0);
    assert!(QP_cost(&h, &g, &du) < 0.0);
    for exponent in 0..10 {
        assert!(cost(0.5_f64.powi(exponent)) > cost(0.0));
    }
    assert!(cost(-1e-3) < cost(0.0));
    let (retained, _) =
        nmpc_step(&x, &u, &reference, &q, &r, &qn, &lo, &hi, 1, 1.0, MASS, &moi, DT);
    assert_eq!(retained, u);
}
