"""Leaf optics: turn an incident PPFD into absorbed photons and driving electrons.

WHY THIS STEP EXISTS
--------------------
A quantum sensor reports PPFD -- photons per second per square metre in the
400-700 nm band, counted equally regardless of wavelength.  A leaf does not
treat them equally.  It reflects and transmits a wavelength-dependent share
(green photons escape far more readily than red ones), and of the photons it
does absorb, the fraction that actually drives photosystem electron transport
also varies with wavelength (the McCree curve: red ~1.0, blue ~0.7, far-red
falling away).  Two fixtures delivering identical PPFD can therefore drive
noticeably different assimilation.

This module collapses that spectral detail into two scalars, computed once
from the fixture spectrum and the leaf's optical properties:

    f_abs   absorbed photons per incident PPFD photon        [-]
    f_ipar  driving electrons per incident PPFD photon       [-]

`photosynthesis.py` then takes PPFD_absorbed = f_abs * PPFD_incident as the
light reaching the photosystems, and IPAR = f_ipar * PPFD_incident as the
electron-transport driver.

WHAT CHANGED FROM THE MATLAB
----------------------------
The legacy `parabs_batch.m` pipeline treats the column of `relative_pfd.txt`
as a spectral IRRADIANCE in W m-2 nm-1 and converts it to photons with
E = hc/lambda.  But that file is a relative PHOTON flux distribution, as its
name says -- the columns integrate to ~1 in photon units.  Passing photons
through a W-to-photon conversion inflates the result by a factor of ~4.2, so
the MATLAB reports an absorbed PPFD about three times the incident PPFD,
which is not physically possible.

This module normalises the spectrum in photon units over the PPFD band and
weights it, so f_abs and f_ipar are bounded by 1 by construction.

DATA FILES  (../data/)
----------------------
relative_pfd.txt      516 x 9.  Column 1 = wavelength [nm], columns 2-9 =
                      relative photon flux distributions for eight fixtures.
                      Column 9 (the default) is the broad, red-weighted
                      white spectrum; columns 2-8 are narrow-band red/blue
                      LED mixes.
lettuce_spectrum.txt  17 x 3.  Wavelength [nm], relative quantum yield [-],
                      leaf absorptance [-].
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "data")

# Integration grid.  Absorption is integrated over 350-750 nm because leaves
# do absorb outside the PAR band, but the spectrum is NORMALISED over
# 400-700 nm, because that is the band a quantum sensor integrates and
# therefore the band the reported PPFD refers to.
_GRID = np.arange(350.0, 751.0, 1.0)
_PPFD_BAND = (_GRID >= 400.0) & (_GRID <= 700.0)


@dataclass
class LeafOptics:
    """Spectral weighting factors for one fixture / leaf combination."""

    f_abs: float
    """Absorbed photons per incident PPFD photon [-]."""

    f_ipar: float
    """Quantum-yield-weighted absorbed photons per incident PPFD photon [-].
    Always <= f_abs."""

    spectrum_column: int
    """1-based MATLAB column of relative_pfd.txt that was used."""

    peak_nm: float
    """Wavelength of peak photon flux [nm], for sanity-checking."""

    @classmethod
    def from_files(cls, spectrum_column: int = 9,
                   pfd_path: str | None = None,
                   leaf_path: str | None = None) -> "LeafOptics":
        """Build the weighting factors from the two spectral data files.

        Parameters
        ----------
        spectrum_column : int
            1-based column index into relative_pfd.txt, MATLAB convention
            (column 1 is wavelength, so the first spectrum is column 2).
            Default 9 = the broad white spectrum, matching the fixture in the
            grow tent and the CEAC MATLAB masterfile's `ri(:,9)`.
        """
        pfd_path = pfd_path or os.path.join(_DATA_DIR, "relative_pfd.txt")
        leaf_path = leaf_path or os.path.join(_DATA_DIR, "lettuce_spectrum.txt")

        ri = np.loadtxt(pfd_path)      # [wavelength, spectrum_1..spectrum_8]
        leaf = np.loadtxt(leaf_path)   # [wavelength, quantum_yield, absorptance]

        # Resample everything onto the common 1 nm grid.  Outside the tabulated
        # range each curve is taken as zero, which is right for the fixture
        # (no emission) and conservative for the leaf (no absorption).
        phi = np.interp(_GRID, ri[:, 0], ri[:, spectrum_column - 1],
                        left=0.0, right=0.0)          # relative photon flux
        qy = np.interp(_GRID, leaf[:, 0], leaf[:, 1], left=0.0, right=0.0)
        absorptance = np.interp(_GRID, leaf[:, 0], leaf[:, 2],
                                left=0.0, right=0.0)

        # Normalise so the spectrum carries exactly one photon inside the
        # 400-700 nm PPFD band.  After this, integrating phi * <anything>
        # over the full grid gives that quantity "per unit reported PPFD".
        ppfd_norm = np.trapezoid(phi[_PPFD_BAND], _GRID[_PPFD_BAND])
        if ppfd_norm <= 0:
            raise ValueError(
                f"Spectrum column {spectrum_column} carries no photons in "
                "400-700 nm; check the column index.")
        phi = phi / ppfd_norm

        f_abs = float(np.trapezoid(phi * absorptance, _GRID))
        f_ipar = float(np.trapezoid(phi * absorptance * qy, _GRID))

        return cls(f_abs=f_abs, f_ipar=f_ipar,
                   spectrum_column=spectrum_column,
                   peak_nm=float(_GRID[np.argmax(phi)]))

    def absorbed_ppfd(self, ppfd_incident):
        """Absorbed photon flux [umol m-2_leaf s-1] from incident PPFD."""
        return self.f_abs * np.asarray(ppfd_incident, dtype=float)

    def ipar(self, ppfd_incident):
        """Electron-transport driving flux [umol m-2_leaf s-1]."""
        return self.f_ipar * np.asarray(ppfd_incident, dtype=float)

    def __str__(self) -> str:
        return (f"LeafOptics(column={self.spectrum_column}, "
                f"peak={self.peak_nm:.0f} nm, "
                f"f_abs={self.f_abs:.3f}, f_ipar={self.f_ipar:.3f})")


# A module-level default so callers that do not care about the fixture can
# just import and go.  Built lazily to keep import cheap and to avoid dying
# at import time if the data directory has moved.
_DEFAULT: LeafOptics | None = None


def default_optics() -> LeafOptics:
    """Leaf optics for the grow-tent white LED panel (relative_pfd column 9)."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = LeafOptics.from_files(spectrum_column=9)
    return _DEFAULT


def dli_to_mean_ppfd(dli_mol_m2_d: float, photoperiod_h: float) -> float:
    """Mean PPFD over the light period [umol m-2 s-1] from a daily light integral.

    DLI is the time integral of PPFD over the whole day, but all of it is
    delivered inside the photoperiod, so dividing by 24 h (as the CEAC MATLAB
    masterfile does) understates the instantaneous flux the leaves actually
    see by the ratio 24/photoperiod -- a factor of 1.5 at a 16 h photoperiod.
    Because the light-response curve is saturating, that is not a wash: it
    puts the leaf on a steeper part of the curve and inflates the marginal
    value of extra light.
    """
    if photoperiod_h <= 0:
        return 0.0
    return dli_mol_m2_d * 1e6 / (photoperiod_h * 3600.0)


def mean_ppfd_to_dli(ppfd_umol_m2_s: float, photoperiod_h: float) -> float:
    """Daily light integral [mol m-2 d-1] from mean PPFD over the light period."""
    return ppfd_umol_m2_s * photoperiod_h * 3600.0 / 1e6
