// Physicochemical tables matching the Python reference in
// publications/shader-based-homology/experiments/common.py.

export const DNA_ALPHABET = "ACGT";
export const PROTEIN_ALPHABET = "ACDEFGHIKLMNPQRSTVWY";

// Kyte-Doolittle hydropathy, scaled to [0, 1].
const RAW_KD = {
  I: 4.5, V: 4.2, L: 3.8, F: 2.8, C: 2.5, M: 1.9, A: 1.8,
  G: -0.4, T: -0.7, S: -0.8, W: -0.9, Y: -1.3, P: -1.6,
  H: -3.2, E: -3.5, Q: -3.5, D: -3.5, N: -3.5, K: -3.9, R: -4.5,
};
const _kdVals = Object.values(RAW_KD);
const _kdMin = Math.min(..._kdVals);
const _kdMax = Math.max(..._kdVals);
export const HYDROPATHY = Object.fromEntries(
  Object.entries(RAW_KD).map(([k, v]) => [k, (v - _kdMin) / (_kdMax - _kdMin)])
);

// van der Waals side-chain volumes (Å^3), scaled to [0, 1].
const RAW_VDW = {
  G: 48.0, A: 67.0, S: 73.0, C: 86.0, D: 91.0, P: 90.0,
  N: 96.0, T: 93.0, E: 109.0, V: 105.0, Q: 114.0, H: 118.0,
  M: 124.0, I: 124.0, L: 124.0, K: 135.0, R: 148.0, F: 135.0,
  Y: 141.0, W: 163.0,
};
const _vVals = Object.values(RAW_VDW);
const _vMin = Math.min(..._vVals);
const _vMax = Math.max(..._vVals);
export const VOLUME = Object.fromEntries(
  Object.entries(RAW_VDW).map(([k, v]) => [k, (v - _vMin) / (_vMax - _vMin)])
);

// Net charge at neutral pH, shifted into [0, 1].
const RAW_CHARGE = {
  K: 1.0, R: 1.0, H: 0.5,
  D: -1.0, E: -1.0,
  A: 0.0, C: 0.0, F: 0.0, G: 0.0, I: 0.0, L: 0.0,
  M: 0.0, N: 0.0, P: 0.0, Q: 0.0, S: 0.0, T: 0.0,
  V: 0.0, W: 0.0, Y: 0.0,
};
export const CHARGE = Object.fromEntries(
  Object.entries(RAW_CHARGE).map(([k, v]) => [k, (v + 1.0) / 2.0])
);
