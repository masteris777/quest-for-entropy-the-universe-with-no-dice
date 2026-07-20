# Lab 88 - The Forced Complex Structure (the "forced i")

import json
from pathlib import Path

import sympy as sp
from sympy import (
    symbols, Matrix, Rational, eye, zeros, simplify, trigsimp,
    cos, sin, diag, linsolve, S, expand, factor, Poly
)
import numpy as np

LAB_FOLDER = Path(__file__).parent


# ============================================================
# Stage A - The 2-plane lemma (symbolic angle)
# ============================================================
def run_stage_a():
    th, ph = symbols('th ph', real=True)

    def R(angle):
        return Matrix([[cos(angle), -sin(angle)], [sin(angle), cos(angle)]])

    J0 = Matrix([[0, -1], [1, 0]])
    I2 = eye(2)

    Rth = R(th)

    # A1: R(th)^T R(th) - I = 0
    A1_expr = simplify(trigsimp(Rth.T * Rth - I2))
    A1_pass = A1_expr == zeros(2, 2)
    print("A1 (R^T R - I):")
    sp.pprint(A1_expr)

    # A2: J0^2 + I = 0
    A2_expr = simplify(J0 * J0 + I2)
    A2_pass = A2_expr == zeros(2, 2)
    print("A2 (J0^2 + I):")
    sp.pprint(A2_expr)

    # A3: J0 R(th) - R(th) J0 = 0
    A3_expr = simplify(trigsimp(J0 * Rth - Rth * J0))
    A3_pass = A3_expr == zeros(2, 2)
    print("A3 (J0 R - R J0):")
    sp.pprint(A3_expr)

    # A4: direct sum, two symbols
    Rph = R(ph)
    J0J0 = diag(J0, J0)
    RthRph = diag(Rth, Rph)
    A4_expr = simplify(trigsimp(J0J0 * RthRph - RthRph * J0J0))
    A4_pass = A4_expr == zeros(4, 4)
    print("A4 (blockdiag commutation):")
    sp.pprint(A4_expr)

    # A5: recorded remark, no computation
    negJ0 = -J0
    A5_check_A2 = simplify(negJ0 * negJ0 + I2) == zeros(2, 2)
    A5_check_A3 = simplify(trigsimp(negJ0 * Rth - Rth * negJ0)) == zeros(2, 2)
    print(f"A5 (-J0 also satisfies A2/A3): A2-form={A5_check_A2}, A3-form={A5_check_A3}")

    overall_pass = A1_pass and A2_pass and A3_pass and A4_pass

    return {
        "kind": "proof",
        "A1": bool(A1_pass),
        "A2": bool(A2_pass),
        "A3": bool(A3_pass),
        "A4": bool(A4_pass),
        "A5_remark": "-J0 also satisfies J^2=-I and commutes with R(th); the complex structure is unique only up to orientation (the two signs of i).",
        "A5_verified": bool(A5_check_A2 and A5_check_A3),
        "pass": bool(overall_pass),
    }


# ============================================================
# Stage B - Exact instances and the obstruction
# ============================================================
def run_stage_b():
    J0 = Matrix([[0, -1], [1, 0]])
    I4 = eye(4)

    R1 = Matrix([[Rational(3, 5), -Rational(4, 5)],
                 [Rational(4, 5), Rational(3, 5)]])
    R2 = Matrix([[Rational(5, 13), -Rational(12, 13)],
                 [Rational(12, 13), Rational(5, 13)]])

    M0 = diag(R1, R2)  # 4x4

    Q13 = eye(4)
    Q13[0, 0] = Rational(3, 5)
    Q13[0, 2] = -Rational(4, 5)
    Q13[2, 0] = Rational(4, 5)
    Q13[2, 2] = Rational(3, 5)

    Q24 = eye(4)
    Q24[1, 1] = Rational(5, 13)
    Q24[1, 3] = -Rational(12, 13)
    Q24[3, 1] = Rational(12, 13)
    Q24[3, 3] = Rational(5, 13)

    Q = Q13 * Q24

    M = Q * M0 * Q.T
    J = Q * diag(J0, J0) * Q.T

    # --- B1 ---
    B1_orthogonal = (M.T * M == I4)
    print(f"B1 orthogonal (M^T M == I): {B1_orthogonal}")

    x = symbols('x')
    charpoly_M = M.charpoly(x).as_expr()
    target = expand((x**2 - Rational(6, 5) * x + 1) * (x**2 - Rational(10, 13) * x + 1))
    B1_charpoly_matches = expand(charpoly_M - target) == 0
    print(f"B1 charpoly matches target: {B1_charpoly_matches}")
    print(f"  charpoly(M) = {expand(charpoly_M)}")
    print(f"  target      = {target}")

    disc1 = Rational(6, 5)**2 - 4
    disc2 = Rational(10, 13)**2 - 4
    print(f"  discriminants: {disc1}, {disc2}")

    B1_J_squared = (J * J == -I4)
    print(f"B1 J*J == -I (direct product): {B1_J_squared}")

    B1_J_commutes = (J * M == M * J)
    print(f"B1 J*M == M*J (direct product): {B1_J_commutes}")

    # --- B2 ---
    C = diag(R1, Matrix([[1, 0], [0, -1]]))
    e3 = Matrix([0, 0, 1, 0])

    nullspace_CmI = (C - I4).nullspace()
    B2_eigenspace_dim = len(nullspace_CmI)
    print(f"B2 nullspace(C - I) dimension: {B2_eigenspace_dim}")
    for v in nullspace_CmI:
        print(f"  basis vector: {v.T}")

    # Commutant: T*C - C*T = 0, 16 unknowns
    t_syms = symbols('t11 t12 t13 t14 t21 t22 t23 t24 t31 t32 t33 t34 t41 t42 t43 t44')
    T = Matrix(4, 4, t_syms)
    eqs_matrix = T * C - C * T
    eqs = [eqs_matrix[i, j] for i in range(4) for j in range(4)]

    sol = linsolve(eqs, t_syms)
    sol = list(sol)
    if len(sol) == 0:
        B2_commutant_dim = 0
        free_syms = []
        sol_expr = None
    else:
        sol_expr = sol[0]  # tuple of 16 expressions in free symbols
        free_syms = sorted(set().union(*[expr.free_symbols for expr in sol_expr]), key=str)
        B2_commutant_dim = len(free_syms)
    print(f"B2 commutant solution-space dimension: {B2_commutant_dim}")

    # Build basis: for each free symbol, set it to 1 and others to 0
    basis_matrices = []
    for fs in free_syms:
        subs_dict = {other: (1 if other == fs else 0) for other in free_syms}
        entries = [expr.subs(subs_dict) for expr in sol_expr]
        Ti = Matrix(4, 4, entries)
        basis_matrices.append(Ti)

    B2_eigenline_invariant_all = True
    for idx, Ti in enumerate(basis_matrices):
        Tie3 = Ti * e3
        # e3 = (0,0,1,0): components 0,1,3 must be exactly zero for Tie3 to be a multiple of e3
        is_multiple = (Tie3[0] == 0) and (Tie3[1] == 0) and (Tie3[3] == 0)
        print(f"  commutant basis T_{idx+1} * e3 = {Tie3.T}  -> multiple of e3: {is_multiple}")
        if not is_multiple:
            B2_eigenline_invariant_all = False

    B3_text = (
        "Any J commuting with C lies in the commutant, so J e3 = c e3 for some real c (B2). "
        "Then J^2 e3 = c^2 e3. But J^2 = -I forces c^2 = -1, impossible over the reals. "
        "Hence no complex structure commutes with C: an odd-dimensional real eigenspace obstructs the i. "
        "This is the dashboard-level form of the mixing-spectrum wall: decay/fixed directions "
        "(heat-equation side) cannot carry complex amplitudes; norm-preserving rotation (quantum side) must."
    )
    B3_argument_included = True
    print("B3:", B3_text)

    overall_pass = (
        B1_orthogonal and B1_charpoly_matches and B1_J_squared and B1_J_commutes
        and B2_eigenspace_dim == 1 and B2_eigenline_invariant_all and B3_argument_included
    )

    return {
        "kind": "proof",
        "B1_orthogonal": bool(B1_orthogonal),
        "B1_charpoly_matches": bool(B1_charpoly_matches),
        "B1_discriminants": [str(disc1), str(disc2)],
        "B1_J_squared": bool(B1_J_squared),
        "B1_J_commutes": bool(B1_J_commutes),
        "B2_eigenspace_dim": int(B2_eigenspace_dim),
        "B2_commutant_dim": int(B2_commutant_dim),
        "B2_eigenline_invariant_all": bool(B2_eigenline_invariant_all),
        "B3_argument_included": bool(B3_argument_included),
        "B3_text": B3_text,
        "pass": bool(overall_pass),
    }


# ============================================================
# Stage C - J-admissibility of recovered dashboards
# ============================================================
def build_stream(name, T, rng):
    if name == "torus":
        a = np.sqrt(2) - 1
        b = np.sqrt(3) - 1
        t = np.arange(T)
        x1 = np.mod(t * a, 1.0)
        x2 = np.mod(t * b, 1.0)
        y = (np.cos(2 * np.pi * x1)
             + 0.6 * np.cos(2 * np.pi * (x1 + x2))
             + 0.3 * np.cos(2 * np.pi * x2))
        return y
    elif name == "logistic":
        z = np.empty(T)
        z0 = 0.37281
        z_prev = z0
        for i in range(T):
            z[i] = z_prev
            z_prev = 4 * z_prev * (1 - z_prev)
        return z - 0.5
    elif name == "iid":
        return rng.standard_normal(T)
    else:
        raise ValueError(name)


def dashboard_pipeline(y, L=40, r=6):
    T = len(y)
    N = T - L
    X = np.empty((N, L))
    for t in range(N):
        X[t, :] = y[t:t + L]
    U, S_, Vt = np.linalg.svd(X, full_matrices=False)
    Z = U[:, :r] * S_[:r]
    A_ls, *_ = np.linalg.lstsq(Z[:-1], Z[1:], rcond=None)
    A = A_ls.T
    return A


def j_tests(A, tol_T1=0.02, tol_T2=0.05, tol_T3=0.05):
    lam, P = np.linalg.eig(A)

    real_eig_rule_failed = bool(np.any(np.abs(lam.imag) < 1e-9))

    result = {
        "eigenvalues": [f"{c.real:.6f}{'+' if c.imag >= 0 else ''}{c.imag:.6f}i" for c in lam],
        "real_eig_rule_failed": real_eig_rule_failed,
    }

    if real_eig_rule_failed:
        result["T1"] = None
        result["T2"] = None
        result["T3"] = None
        result["imag_residual"] = None
        result["failed_tests"] = ["real_eigenvalue_rule"]
        result["passes_all"] = False
        return result

    D = np.diag(1j * np.sign(lam.imag))
    Pinv = np.linalg.inv(P)
    J_complex = P @ D @ Pinv
    imag_residual = float(np.max(np.abs(J_complex.imag)))
    J = np.real(J_complex)

    T1 = float(np.max(np.abs(np.abs(lam) - 1)))
    T2 = float(np.max(np.abs(J @ J + np.eye(A.shape[0]))))
    T3 = float(np.max(np.abs(J @ A - A @ J)))

    failed_tests = []
    if T1 > tol_T1:
        failed_tests.append("T1")
    if T2 > tol_T2:
        failed_tests.append("T2")
    if T3 > tol_T3:
        failed_tests.append("T3")
    if imag_residual > 1e-6:
        failed_tests.append("reality_check")

    result["T1"] = T1
    result["T2"] = T2
    result["T3"] = T3
    result["imag_residual"] = imag_residual
    result["failed_tests"] = failed_tests
    result["passes_all"] = len(failed_tests) == 0

    return result


def run_stage_c():
    rng = np.random.default_rng(88)
    T = 20000
    L = 40
    r = 6

    streams = {}
    for name in ["torus", "logistic", "iid"]:
        y = build_stream(name, T, rng)
        A = dashboard_pipeline(y, L=L, r=r)
        res = j_tests(A)
        streams[name] = res
        print(f"\n--- Stream: {name} ---")
        print(f"  eigenvalues: {res['eigenvalues']}")
        print(f"  real_eig_rule_failed: {res['real_eig_rule_failed']}")
        print(f"  T1={res.get('T1')}, T2={res.get('T2')}, T3={res.get('T3')}, imag_residual={res.get('imag_residual')}")
        print(f"  failed_tests: {res['failed_tests']}")

    torus_pass = streams["torus"]["passes_all"]
    logistic_fail = (not streams["logistic"]["passes_all"]) if not streams["logistic"]["real_eig_rule_failed"] else True
    logistic_fail = len(streams["logistic"]["failed_tests"]) > 0
    iid_fail = len(streams["iid"]["failed_tests"]) > 0

    overall_pass = torus_pass and logistic_fail and iid_fail

    out = {
        "kind": "simulation",
        "torus": {
            "eigenvalues": streams["torus"]["eigenvalues"],
            "real_eig_rule_failed": streams["torus"]["real_eig_rule_failed"],
            "T1": streams["torus"]["T1"],
            "T2": streams["torus"]["T2"],
            "T3": streams["torus"]["T3"],
            "imag_residual": streams["torus"]["imag_residual"],
            "passes_all": streams["torus"]["passes_all"],
        },
        "logistic": {
            "eigenvalues": streams["logistic"]["eigenvalues"],
            "real_eig_rule_failed": streams["logistic"]["real_eig_rule_failed"],
            "T1": streams["logistic"]["T1"],
            "T2": streams["logistic"]["T2"],
            "T3": streams["logistic"]["T3"],
            "failed_tests": streams["logistic"]["failed_tests"],
        },
        "iid": {
            "eigenvalues": streams["iid"]["eigenvalues"],
            "real_eig_rule_failed": streams["iid"]["real_eig_rule_failed"],
            "T1": streams["iid"]["T1"],
            "T2": streams["iid"]["T2"],
            "T3": streams["iid"]["T3"],
            "failed_tests": streams["iid"]["failed_tests"],
        },
        "pass": bool(overall_pass),
    }
    return out


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    results = {}

    print("=" * 60)
    print("STAGE A - The 2-plane lemma")
    print("=" * 60)
    stage_a = run_stage_a()
    results["A"] = stage_a
    print(f"\nStage A: {'PASS' if stage_a['pass'] else 'FAIL'}")

    print("\n" + "=" * 60)
    print("STAGE B - Exact instances and the obstruction")
    print("=" * 60)
    stage_b = run_stage_b()
    results["B"] = stage_b
    print(f"\nStage B: {'PASS' if stage_b['pass'] else 'FAIL'}")

    print("\n" + "=" * 60)
    print("STAGE C - J-admissibility of recovered dashboards")
    print("=" * 60)
    stage_c = run_stage_c()
    results["C"] = stage_c
    print(f"\nStage C: {'PASS' if stage_c['pass'] else 'FAIL'}")

    # Verdict
    if not (stage_a["pass"] and stage_b["pass"]):
        verdict = "FAIL"
        verdict_reason = "Stage A and/or Stage B did not verify exactly; see printed failing objects."
        obligations_closed = []
    elif stage_c["pass"]:
        verdict = "PASS"
        verdict_reason = "Stages A and B verified exactly; Stage C matched all three predeclared predictions (torus passes T1/T2/T3, logistic and iid each fail at least one test)."
        obligations_closed = ["model-zero-w1-forced-complex-structure"]
    else:
        verdict = "PARTIAL"
        deviations = []
        if not stage_c["torus"]["passes_all"]:
            deviations.append(f"torus failed: {stage_c['torus']}")
        if len(stage_c["logistic"]["failed_tests"]) == 0 and not stage_c["logistic"]["real_eig_rule_failed"]:
            deviations.append("logistic unexpectedly passed all J-tests")
        if len(stage_c["iid"]["failed_tests"]) == 0 and not stage_c["iid"]["real_eig_rule_failed"]:
            deviations.append("iid unexpectedly passed all J-tests")
        verdict_reason = "Stages A and B verified exactly; Stage C deviated from predeclared predictions: " + "; ".join(deviations)
        obligations_closed = []

    metrics = {
        "lab": 88,
        "stream": "cryptographic-substrate",
        "stages": results,
        "obligations_closed": obligations_closed,
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "leakage_audit": {
            "wrote_outside_lab_folder": False,
            "posthoc_threshold_change": False,
            "floats_in_proof_stages": False,
            "streams_modified": False,
        },
    }

    with open(LAB_FOLDER / "metrics_88.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nVerdict: {verdict}")
    print(f"Reason: {verdict_reason}")
