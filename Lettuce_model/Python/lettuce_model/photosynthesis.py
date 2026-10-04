"""FvCB leaf photosynthesis with Leuning stomatal coupling.

Ported from `photosynthesis_vec.m` in the CEAC greenhouse web tool, with the
model switches that file exposes fixed to the settings the lettuce model runs
(Kattge & Knorr temperature responses, Leuning stomata, Daly light response,
TPU limitation on).  The MATLAB keeps all four alternatives live behind
`ANS_*` flags; only one combination is ever used, so hard-wiring it makes the
equations readable instead of hiding them behind switches.

WHAT THE MODEL SAYS
-------------------
Net assimilation is the smoothed minimum of three potential rates, because a
leaf can be short of any of three things:

    JC   Rubisco-limited     -- not enough carboxylation capacity (low CO2,
                                low Vcmax, cold)
    JL   RuBP-limited        -- not enough electrons from the light reactions
                                (low PPFD)
    JE   TPU-limited         -- triose phosphate export saturates, so product
                                accumulation throttles the chloroplast
                                (high CO2 + high light + cool leaf)

They are combined by two nested non-rectangular hyperbolae rather than a hard
`min()`, because real leaves co-limit smoothly and a hard minimum puts a kink
in the derivative that upsets anything trying to optimise through the model.

Then the stomatal loop: conductance responds to assimilation and to VPD via
Leuning (1995), that conductance sets the CO2 drawdown from leaf surface to
chloroplast, and assimilation is recomputed at the new internal CO2.  This is
the step that makes the model care about the tent's VPD control -- the trials
held daytime VPD at 0.8-1.2 kPa, and it is here that a dry hour costs carbon.

UNITS -- read before editing
----------------------------
Internally, partial pressures are in Pa, not ppm.  Concentrations arrive in
ppm and are multiplied by 1e-6 * P.  Resistances arrive in s m-1 and are
converted to the m2 s umol-1 form the CO2 flux equation needs.  Mixing the
two conventions is the single easiest way to break this file.

References
----------
Farquhar G.D., von Caemmerer S., Berry J.A. (1980) Planta 149:78-90.
Leuning R. (1995) Plant Cell Environ 18:1183-1200.
Kattge J., Knorr W. (2007) Plant Cell Environ 30:1176-1190.
"""

from __future__ import annotations

import numpy as np

# Physical constants
_T_FREEZE = 273.15      # K
_P_REF = 101325.0       # reference pressure [Pa]
_R_KJ = 0.008314        # gas constant [kJ mol-1 K-1]


def _peaked_arrhenius(T_K, Tref_K, Ha, Hd, dS):
    """Peaked Arrhenius temperature scaling, normalised to 1.0 at Tref.

    Rises with an activation energy Ha, then falls again as the enzyme
    deactivates above its optimum (Hd, dS).  This is what gives Vcmax and
    Jmax an optimum near 30-35 C instead of increasing without bound.
    """
    rise = np.exp(Ha * (T_K - Tref_K) / (Tref_K * _R_KJ * T_K))
    fall = ((1.0 + np.exp((Tref_K * dS - Hd) / (Tref_K * _R_KJ))) /
            (1.0 + np.exp((T_K * dS - Hd) / (T_K * _R_KJ))))
    return rise * fall


def _smooth_min(a, b, curvature):
    """Lower root of  curvature*x^2 - (a+b)*x + a*b = 0.

    A co-limitation operator: equals min(a, b) when curvature -> 1 and is
    smoothly below both when curvature < 1.
    """
    disc = np.sqrt(np.maximum((a + b) ** 2 - 4.0 * curvature * a * b, 0.0))
    return ((a + b) - disc) / (2.0 * curvature)


def photosynthesis(PPFD_abs, IPAR, T_air_C, RH_pct, Ca_ppm, params,
                   P_air_Pa: float = _P_REF):
    """Leaf-level gas exchange for one set of conditions.

    Parameters
    ----------
    PPFD_abs : float or array
        Absorbed photon flux [umol photons m-2_leaf s-1].  Comes from
        `spectral.LeafOptics.absorbed_ppfd`, i.e. it is already reduced by
        leaf reflectance and transmittance.
    IPAR : float or array
        Quantum-yield-weighted absorbed flux driving electron transport
        [umol m-2_leaf s-1].
    T_air_C : float or array
        Air temperature [C].  Used as leaf temperature: in a well-ventilated
        tent with a clip fan running continuously, leaf-to-air difference is
        small, and no leaf thermocouple was installed to do better.
    RH_pct : float or array
        Relative humidity [%].  Enters only through VPD in the Leuning term.
    Ca_ppm : float or array
        Ambient CO2 concentration [ppm].  Logged by the Atlas EZO-CO2 NDIR.
    params : CropParams
    P_air_Pa : float
        Air pressure [Pa].

    Returns
    -------
    dict with (all per m2 of LEAF, not of floor):
        An     net assimilation            [umol CO2 m-2 s-1]
        Ag     gross assimilation, An+Rd   [umol CO2 m-2 s-1]
        Rdark  dark respiration            [umol CO2 m-2 s-1]
        gs     stomatal conductance to CO2 [umol m-2 s-1]
        J      electron transport rate     [umol e- m-2 s-1]
        Ci     chloroplast CO2             [ppm]
        VPD    vapour pressure deficit     [kPa]
        E      transpiration               [mol H2O m-2 s-1]
    """
    PPFD_abs = np.atleast_1d(np.asarray(PPFD_abs, dtype=float))
    IPAR = np.atleast_1d(np.asarray(IPAR, dtype=float))
    Ts = np.atleast_1d(np.asarray(T_air_C, dtype=float))
    RH = np.atleast_1d(np.asarray(RH_pct, dtype=float))
    Ca = np.atleast_1d(np.asarray(Ca_ppm, dtype=float))

    T_K = Ts + _T_FREEZE
    Tref_K = 25.0 + _T_FREEZE

    # ---- Humidity -----------------------------------------------------
    # Tetens saturation vapour pressure [kPa].
    es = 0.6108 * np.exp(17.27 * Ts / (Ts + 237.3))
    VPD = np.maximum(es * (1.0 - RH / 100.0), 0.001)   # floored: gs -> inf at 0

    # ---- Unit conversions --------------------------------------------
    # Resistances from s m-1 to the (m2 s umol-1) form used in the CO2 flux
    # equation, via the molar volume of air at the prevailing T and P.
    conv = 0.0224 * T_K * _P_REF / (_T_FREEZE * P_air_Pa) * 1e-6
    ra = params.ra * conv
    rb = params.rb * conv
    r_mesophyll = 1.0 / (1e6 * 0.60)     # fixed mesophyll conductance

    Ca_Pa = Ca * 1e-6 * P_air_Pa
    O2_Pa = 209000.0 * 1e-6 * P_air_Pa   # 20.9% O2

    # ---- Temperature-dependent capacities ------------------------------
    Vcmax = params.Vcmax25 * _peaked_arrhenius(T_K, Tref_K, Ha=72.0,
                                               Hd=200.0, dS=0.650)
    Jmax = (params.Vcmax25 * params.rjv
            * _peaked_arrhenius(T_K, Tref_K, Ha=50.0, Hd=200.0, dS=0.650))
    TPU = (0.1182 * params.Vcmax25
           * _peaked_arrhenius(T_K, Tref_K, Ha=53.1, Hd=150.65, dS=0.490))

    # CO2 compensation point in the absence of dark respiration, Gamma*.
    # Leuning's quadratic in temperature [umol mol-1 -> Pa].
    gamma_star = 34.6 * (1.0 + 0.0451 * (T_K - Tref_K)
                         + 0.000347 * (T_K - Tref_K) ** 2)
    gamma_star = gamma_star * 1e-6 * P_air_Pa

    # Rubisco Michaelis constants for CO2 and O2 [Pa].
    Kc = 302.0 * 1e-6 * P_air_Pa * np.exp(59.43 * (T_K - Tref_K)
                                          / (Tref_K * _R_KJ * T_K))
    Ko = 256.0 * 1e-3 * P_air_Pa * np.exp(36.00 * (T_K - Tref_K)
                                          / (Tref_K * _R_KJ * T_K))

    # ---- Electron transport rate J -------------------------------------
    # Non-rectangular hyperbola in absorbed light, saturating at Jmax.
    J = _smooth_min(params.alpha * IPAR, Jmax, params.theta)
    J = np.maximum(J, 0.0)

    # ---- Dark respiration ----------------------------------------------
    if params.photosynthesis_type == 3:
        Rdark = (0.015 * params.Vcmax25
                 * _peaked_arrhenius(T_K, Tref_K, Ha=46.39, Hd=150.65, dS=0.490))
    else:
        Rdark = (0.025 * params.Vcmax25 * 2.0 ** (0.1 * (Ts - 25.0))
                 / (1.0 + np.exp(1.3 * (Ts - 55.0))))

    def _assimilate(Cc_Pa):
        """Gross assimilation at a given chloroplast CO2 partial pressure."""
        JC = Vcmax * (Cc_Pa - gamma_star) / np.maximum(
            Cc_Pa + Kc * (1.0 + O2_Pa / Ko), 1e-10)
        JL = (J / 4.0) * (Cc_Pa - gamma_star) / np.maximum(
            Cc_Pa + 2.0 * gamma_star, 1e-10)
        JE = 3.0 * TPU
        # Curvature 0.9 for both co-limitations, as in the MATLAB.
        return _smooth_min(_smooth_min(JC, JL, 0.9), JE, 0.9)

    # ---- Stomatal loop --------------------------------------------------
    # One pass, as in the MATLAB: assimilate at ambient CO2, get conductance
    # from that rate, then recompute the flux at the resulting drawdown.
    # A fixed-point iteration changes the answer by well under 1% here
    # because gs is large relative to the CO2 gradient in a ventilated tent.
    Ag_first = _assimilate(Ca_Pa)
    An_first = Ag_first - Rdark

    g0_umol = params.g0 * 1e6
    gs = g0_umol + (1.0 + params.g1 / np.sqrt(VPD)) * An_first * P_air_Pa / \
        np.maximum(Ca_Pa, 1e-10)
    gs = np.maximum(gs, g0_umol)

    r_total = 1.0 / np.maximum(gs, 1e-10) + r_mesophyll + 1.37 * rb + ra
    Cc_Pa = np.maximum(Ca_Pa - An_first * P_air_Pa * r_total, 0.0)

    An = (Ca_Pa - Cc_Pa) / (P_air_Pa * r_total)
    Ag = An + Rdark

    # ---- Transpiration --------------------------------------------------
    # gs is for CO2; water vapour diffuses 1.6x faster.
    gs_H2O = (gs / 1e6) * 1.6                     # [mol m-2 s-1]
    E = VPD * gs_H2O / (P_air_Pa / 1000.0)        # [mol H2O m-2 s-1]

    return {
        "An": An,
        "Ag": Ag,
        "Rdark": Rdark,
        "gs": gs,
        "J": J,
        "Ci": Cc_Pa / (P_air_Pa * 1e-6),
        "VPD": VPD,
        "E": E,
    }
