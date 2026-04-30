# Summary of `pdf.py`

## Purpose

`contraction_lit/pdf.py` is a standalone contraction driver for pion PDF-related test measurements. The changelog in the file shows the script grew from gauge-flow and fermion-flow cross checks into pion 2-point, local 3-point, and non-local 3-point contractions.

The script uses the `lqcd` package with the NumPy backend and the UKQCD gamma convention. Its active path currently runs a fixed small test setup and writes NumPy correlator files under `../data/test/`.

## Entry Point and Arguments

When command-line arguments are supplied, the script requires:

- `-cd, --confdir`: intended configuration/perambulator directory.
- `-e, --ens`: ensemble or configuration name.
- `-d, --diagram`: one or more diagram labels, documented as `loop`, `BD`, or `W`.

When no command-line arguments are supplied, it falls back to an internal `Args` class with a hard-coded test directory, `ens = "beta6.00"`, and `diagram = ["bdv"]`.

Important implementation detail: the parsed arguments are printed, but they do not currently control the workflow. The script always uses hard-coded settings in the main body.

## Fixed Runtime Setup

The active settings are:

- Backend: `numpy`.
- Gamma convention: `ukqcd`.
- Geometry: `geo_vec = [8, 4, 4, 4]`, so `T = 8`, `X = Y = Z = 4`.
- Test mode: `check = 1`, so only configuration `conf = 0` is processed.
- Gauge input in test mode: `../WFlow_tests_Ani/conf.0000.npy`.
- Output directory: `../data/test/`.
- Boundary condition: `U.apply_boundary_condition_minus_one()` is applied before constructing the Dirac operator.
- Dirac operator: twisted-mass clover with `m = 0.5`, `mu = 0.112994350282`, and `csw = 1.74`.
- Solver: `BiCGStab`, tolerance `1e-9`, maximum iterations `500`, twisted-mass rotation enabled.

Although Jacobi smearing parameters are defined, the active source and sink smearing calls are commented out. The point source is placed at lattice origin `[0, 0, 0, 0]` for all spin-color components.

## Helper Readers

`read_gauge_Ani(filename)` and `read_spinor_Ani(filename)` read CSV-style reference data from Ani's format. Both convert complex data from real/imaginary columns, reshape from `xyzt` order to `txyz`, and reorder the spin/direction axis from Ani's convention into the code's convention.

These helpers are only used inside disabled cross-check blocks in the current script.

## Main Workflow

1. Load or read the gauge field.
2. Apply the boundary condition to get `U_with_phase`.
3. Build the twisted-mass clover Dirac operator.
4. Construct a full point-source propagator source `srcfull`.
5. Solve point-to-all propagators:
   - `Su_ps = ut.propagator_parallelized(..., 'u')`
   - `Sd_ps = ut.propagator_parallelized(..., 'd')`
6. Use unsmeared propagators as the active sink propagators:
   - `Su_ss = Su_ps`
   - `Sd_ss = Sd_ps`
7. Compute pion 2-point and 3-point contractions.

`Sd_ps`, `Su_ss`, and `Sd_ss` are prepared but are not used by the active contraction code that follows.

## Pion 2-Point Function

The pion 2-point function is computed with:

```python
corr_2pt_pion_space = cf.meson(Su_ps, Su_ps, 5, 5)
```

Then the script sums over spatial sites for each time slice:

```python
corr_2pt_pion[t] = xp.sum(corr_2pt_pion_space[t])
```

Output:

- `../data/test/corr_2pt_pion_conf_<conf>.npy`
- Shape in the active test setup: `(8,)`

## Local Flowed 3-Point Function with `gamma_t`

In test mode, the script also computes a local flowed 3-point pion correlator with insertion `gamma_0`.

Settings:

- Flow steps: `gflow_niter = 2`.
- Flow step size: `gflow_dt = 0.125`.
- Output array shape: `(gflow_niter + 1, T, T)`, which is `(3, 8, 8)` in the active setup.

For each flow time and source-sink separation `tsep`, the script:

1. Forward-flows the point-to-all propagator.
2. Builds a sequential source with a `gamma_5` insertion.
3. Solves the down-quark sequential propagator.
4. Forward-flows the sequential propagator.
5. Contracts

```python
conj(Seq_fs) * gamma_5 * gamma_0 * Su_fs
```

6. Sums over spatial sites for insertion times `tins <= tsep`.

Output:

- `../data/test/corr_3pt_pion_conf_<conf>_g4.npy`
- Shape in the active test setup: `(3, 8, 8)`

Debug side effects:

- `Seq_it_0_tsep5.npy`
- `Seq_it_1_tsep5.npy`

These files are saved in the current working directory when `it` is 0 or 1 and `tsep` is 5.

## Non-Local PDF 3-Point Functions

The non-local PDF block is the main active PDF-related calculation.

Settings:

- Flow steps: `gflow_niter = 10`.
- Flow step size: `gflow_dt = 0.125`.
- Maximum derivative/link order: `N_der = 5`.
- Directions are mapped by

```python
mu_num2st = {
    0: ['t', '-t'],
    1: ['x', '-x'],
    2: ['y', '-y'],
    3: ['z', '-z'],
}
```

The script allocates one correlator array for each derivative/link order `d = 0, 1, 2, 3, 4, 5`.

### `d = 0`

This is the local insertion case. For each gamma direction `mu1`, the contraction is:

```python
conj(Seq_fs) * gamma_5 * gamma_mu1 * Su_fs
```

Output shape:

- `(Nflow, mu1, tsep, tins)`
- Active setup: `(11, 4, 8, 8)`

### `d = 1`

This is a one-link non-local insertion. For each `mu1` and link direction `mu2`, the script computes forward and backward terms:

- Forward term: uses `U_f.mu(+mu2)` and `Su_fs.shift(+mu2)` with a positive sign.
- Backward term: uses `U_f.mu(-mu2)` and `Su_fs.shift(-mu2)` with a negative sign.

Output shape:

- `(Nflow, mu1, mu2, fwd_bwd, tsep, tins)`
- Active setup: `(11, 4, 4, 2, 8, 8)`

### `d = 2`

This is a two-link non-local insertion. For each `mu1`, `mu2`, and `mu3`, the script computes all four forward/backward sign combinations:

- `(+mu2, +mu3)` with a positive sign.
- `(+mu2, -mu3)` with a negative sign.
- `(-mu2, +mu3)` with a negative sign.
- `(-mu2, -mu3)` with a positive sign.

Each term uses two gauge links and a twice-shifted propagator.

Output shape:

- `(Nflow, mu1, mu2, mu3, fwd_bwd_mu2, fwd_bwd_mu3, tsep, tins)`
- Active setup: `(11, 4, 4, 4, 2, 2, 8, 8)`

### `d = 3, 4, 5`

The longer non-local insertions are generated by a helper that builds the color-link chain dynamically:

- `d = 3`: three gauge links, three shifts, and sign `(-1)^(number of backward shifts)`.
- `d = 4`: four gauge links, four shifts, and sign `(-1)^(number of backward shifts)`.
- `d = 5`: five gauge links, five shifts, and sign `(-1)^(number of backward shifts)`.

The active setup shapes are:

- `d = 3`: `(11, 4, 4, 4, 4, 2, 2, 2, 8, 8)`
- `d = 4`: `(11, 4, 4, 4, 4, 4, 2, 2, 2, 2, 8, 8)`
- `d = 5`: `(11, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 8, 8)`

## Output Files

The active script writes:

- `../data/test/corr_2pt_pion_conf_<conf>.npy`
- `../data/test/corr_3pt_pion_conf_<conf>_g4.npy`
- `../data/test/corr_3pt_pion_conf_<conf>_Nder_0.npy` through `../data/test/corr_3pt_pion_conf_<conf>_Nder_5.npy`

In the current hard-coded test mode, `<conf>` is always `0`.

## Disabled or Inactive Blocks

The script contains several disabled checks guarded by `if 0:`:

- Gauge-flow comparison against Ani's reference data.
- Fermion forward-flow comparison.
- Fermion adjoint-flow comparison.
- Optional saving of flowed fields.
- A small intermediate Dirac-operator check.

These blocks document useful validation paths, but they are not run by default.

## Practical Notes

- `args.confdir`, `args.ens`, and `args.diagram` are parsed but not used to select input files or enable/disable diagrams.
- `check = 1` forces the script into test mode; production configuration reading is disabled unless this value is edited.
- Several values are hard-coded in the main block, including geometry, flow parameters, solver parameters, and output paths.
- `Cg5`, `beta`, `gf_tau`, `Sd_ps`, `Su_ss`, and `Sd_ss` are currently unused or only partially used.
- The calculation is expensive because the sequential propagator is recomputed for every `tsep` and flow-time block, and the PDF contractions loop over all direction combinations up to `N_der = 5`.
