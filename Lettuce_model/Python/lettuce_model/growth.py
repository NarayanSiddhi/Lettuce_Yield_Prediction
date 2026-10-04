"""Daily carbon balance and biomass accumulation for hydroponic lettuce.

THE MODEL IN ONE PARAGRAPH
--------------------------
Two state variables carry the crop.  `W_struct` is structural dry matter --
cell walls, the stuff that makes a leaf -- and it is what determines both leaf
area and harvestable weight.  `W_buffer` is the non-structural pool of
soluble sugars and starch.  Photosynthesis pays into the buffer; maintenance
respiration draws from it around the clock; and structural growth draws from
it at a rate that saturates as the buffer fills.  Splitting the two is what
lets the model do something a single-pool model cannot: keep growing for a
day or two after the light drops, and stall when the buffer empties even
though the leaves are still there.

    W_buffer  <--- photosynthesis (light period only)
              ---> maintenance respiration (24 h)
              ---> structural growth + its synthesis overhead

    W_struct  ---> leaf area  ---> light interception ---> photosynthesis
              ---> harvestable fresh weight

The feedback from W_struct back into photosynthesis through leaf area is what
produces the sigmoid: exponential while the canopy is open and every new leaf
intercepts new light, then linear once the canopy closes and interception
saturates at 1.

DAILY TIME STEP -- and why that is defensible here
--------------------------------------------------
The step is one day.  Photosynthesis is evaluated once, at the mean conditions
over the light period, rather than integrated hour by hour.  Because the
light-response curve is concave, evaluating at the mean of a varying PPFD
overestimates the integral of a varying one (Jensen's inequality).  In these
trials that bias is small, because the LED panel is a step function: PPFD is
essentially constant for 16 h and zero for 8 h, which is exactly the case
where mean-over-photoperiod is nearly exact.  Do not reuse this shortcut with
sunlight drivers without checking it.

Van Henten (1994) Agricultural Systems 45:55-72 is the source for the growth
and respiration structure.  A copy is in ../../Literature/.

MAPPING TO THE LEGACY MATLAB
-----------------------------
`C:\\CEAC_proposal_modeling\\Lettuce_model\\simulateLettuceGrowthSingleDay.m`
    Ws  ->  W_buffer   (its comment says "shoot dry biomass"; the equations
                        use it as the assimilate pool)
    Wg  ->  W_struct   (its comment says "root dry biomass"; the equations
                        use it for leaf area and for fresh weight)

Three equations were corrected on the way across, each flagged CORRECTION in
the code below.  They are listed in ../../Lettuce_model/README.txt so the
differences are auditable rather than silent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .params import CropParams
from .photosynthesis import photosynthesis
from .spectral import LeafOptics, default_optics, dli_to_mean_ppfd

# Molar masses
_MW_CO2 = 44.0     # g mol-1
_MW_CH2O = 30.0    # g mol-1


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------
def initial_state(params: CropParams) -> dict:
    """State at transplant, derived from the OBSERVED transplant fresh weight.

    Anchoring to the measurement rather than to a literature seedling weight
    matters more than it looks: growth is close to exponential for the first
    ~12 days, so a 30% error in starting mass is still a 30% error two weeks
    later.  Both trials transplanted at the same 4.7 g median net fresh
    weight, so both start identically -- any later divergence is the model
    responding to the drivers, not to different initial conditions.
    """
    # Shoot fresh weight per plant -> shoot dry matter per m2 of bench.
    shoot_dm_per_m2 = (params.initial_fresh_weight_g
                       * params.planting_density
                       * params.dry_matter_fraction
                       / max(params.usable_fraction, 1e-6))

    # Split into buffer and structure, then scale structure up from shoot-only
    # to whole-plant, since W_struct includes roots (fraction c_tau).
    buffer = params.initial_buffer_fraction * shoot_dm_per_m2
    struct_shoot = shoot_dm_per_m2 - buffer
    struct_total = struct_shoot / (1.0 - params.c_tau)

    return {
        "W_struct": float(struct_total),
        "W_buffer": float(buffer),
        "GDD": 0.0,
    }


# ---------------------------------------------------------------------------
# Derived canopy quantities
# ---------------------------------------------------------------------------
def structural_mass_at_target(params: CropParams) -> float:
    """Structural dry matter per m2 when the crop reaches its target weight.

    Used only as the maturity scale over which specific leaf area declines.
    """
    shoot_dm = (params.target_harvest_weight_g * params.planting_density
                * params.dry_matter_fraction / max(params.usable_fraction, 1e-6))
    return shoot_dm / (1.0 - params.c_tau)


def leaf_area_index(params: CropParams, W_struct: float) -> float:
    """Leaf area index [m2_leaf m-2_floor].

    Only the shoot share of structural dry matter carries leaf area, hence the
    (1 - c_tau) factor.

    Two forms are available (see `params.lar_model`):

    'constant'   LAI = c_lar * (1 - c_tau) * W_struct.  Van Henten's original.

    'declining'  Specific leaf area falls linearly with accumulated mass, from
                 sla_max at transplant to sla_min at the target head weight,
                 because leaves thicken as the head fills.  Integrating
                 dLAI = SLA(W) dW from 0 to W gives the closed form below.
                 Past the target mass the canopy keeps thickening at sla_min.

    CORRECTION vs. MATLAB: `simulateLettuceGrowthSingleDay.m` computes
    `f_light = 1 - exp(-0.7 * c_lar * Wg)`, dropping the (1 - c_tau) shoot
    fraction and using an extinction coefficient of 0.7 where Van Henten
    specifies 0.9.
    """
    W = max(float(W_struct), 0.0)
    shoot_frac = 1.0 - params.c_tau

    if params.lar_model == "constant":
        return params.c_lar * shoot_frac * W

    W_h = max(structural_mass_at_target(params), 1e-9)
    W_eff = min(W, W_h)
    lai = shoot_frac * (params.sla_max * W_eff
                        - (params.sla_max - params.sla_min)
                        * W_eff * W_eff / (2.0 * W_h))
    if W > W_h:
        lai += shoot_frac * params.sla_min * (W - W_h)
    return max(lai, 0.0)


def light_interception(params: CropParams, W_struct: float) -> float:
    """Fraction of incident PPFD intercepted by the canopy [-].

    Beer-Lambert.  Saturates near 1 at LAI ~4-5, which for these parameters
    happens around W_struct ~ 60-80 g DM m-2, i.e. roughly DAT 14-18.  That
    saturation is the mechanism behind the growth curve straightening out from
    exponential to linear part way through the cycle.
    """
    return 1.0 - np.exp(-params.c_K * leaf_area_index(params, W_struct))


def fresh_weight_per_plant(params: CropParams, W_struct: float,
                           W_buffer: float) -> float:
    """Harvestable shoot fresh weight per plant [g FW plant-1].

    Both pools count: the buffer is real dry matter sitting in the leaves.

    CORRECTION vs. MATLAB: `Biomass_fresh = Wg / dry_matter_fraction *
    usable_fraction` uses only one pool and does not remove the root share, so
    it reports whole-plant structure as if it were marketable shoot.
    """
    shoot_dm = (1.0 - params.c_tau) * (max(W_struct, 0.0) + max(W_buffer, 0.0))
    fresh_per_m2 = shoot_dm / params.dry_matter_fraction * params.usable_fraction
    return fresh_per_m2 / params.planting_density


def ec_growth_factor(params: CropParams, ec_mS_cm: float) -> float:
    """Optional empirical EC modifier on structural growth [-].

    Two limbs, after Samarakoon et al. (2020) for NFT lettuce:
      below the optimum, a Michaelis-Menten rise in nutrient availability;
      above it, a linear osmotic/transport penalty.

    Returns 1.0 unless `params.ec_response` is True.  Read the note in
    params.py before switching it on -- it moves EC signal out of the residual
    that this project's image analysis is meant to explain.
    """
    if not params.ec_response or not np.isfinite(ec_mS_cm):
        return 1.0
    ec = max(float(ec_mS_cm), 0.0)
    k = params.ec_half_saturation_mS_cm
    opt = params.ec_optimum_mS_cm
    rise = (ec / (k + ec)) / (opt / (k + opt))          # 1.0 at the optimum
    if ec <= opt:
        return float(np.clip(rise, 0.0, 1.0))
    return float(np.clip(1.0 - params.ec_supra_slope_per_mS * (ec - opt),
                         0.0, 1.0))


# ---------------------------------------------------------------------------
# One day
# ---------------------------------------------------------------------------
def simulate_day(state: dict, params: CropParams, *,
                 T_air_C: float, T_air_light_C: float,
                 RH_pct: float, RH_light_pct: float,
                 CO2_ppm: float, PPFD_umol_m2_s: float,
                 P_air_Pa: float = 101325.0,
                 EC_mS_cm: float = np.nan,
                 optics: LeafOptics | None = None) -> dict:
    """Advance the crop one day and return the new state plus diagnostics.

    All fluxes below are per square metre of BENCH FLOOR unless the name says
    otherwise.  The conversion from leaf area to floor area happens once, in
    the `f_light` multiplication -- that is the big-leaf assumption, and it is
    the main structural approximation in the model.
    """
    optics = optics or default_optics()
    W_struct = state["W_struct"]
    W_buffer = state["W_buffer"]

    # -- 1. Which temperature drives photosynthesis? --------------------
    # Assimilation only happens while the lights are on, so the photoperiod
    # mean is the physical choice.  'daily' reproduces the CEAC web tool.
    if params.temp_basis == "photoperiod" and np.isfinite(T_air_light_C):
        T_photo, RH_photo = T_air_light_C, RH_light_pct
    else:
        T_photo, RH_photo = T_air_C, RH_pct

    # -- 2. Light reaching the photosystems -----------------------------
    # PPFD_umol_m2_s is what a quantum sensor at canopy height would read.
    # The optics step removes reflected and transmitted photons and weights
    # the rest by their photosynthetic effectiveness.
    ppfd_abs = optics.f_abs * PPFD_umol_m2_s
    ipar = optics.f_ipar * PPFD_umol_m2_s

    # -- 3. Leaf gas exchange -------------------------------------------
    gas = photosynthesis(ppfd_abs, ipar, T_photo, RH_photo, CO2_ppm,
                         params, P_air_Pa=P_air_Pa)
    Ag_leaf = float(gas["Ag"][0])        # gross, [umol CO2 m-2_leaf s-1]
    An_leaf = float(gas["An"][0])
    E_leaf = float(gas["E"][0])

    # -- 4. Scale leaf to canopy, and instant to daily ------------------
    f_light = light_interception(params, W_struct)

    # Gross canopy assimilation over the light period, expressed as the CH2O
    # it is worth to the buffer:
    #   umol CO2 m-2_leaf s-1
    #     * 44e-6 g CO2 umol-1        -> g CO2 m-2_leaf s-1
    #     * f_light                   -> g CO2 m-2_floor s-1
    #     * canopy_efficiency         -> big-leaf upscaling correction
    #     * photoperiod_s             -> g CO2 m-2_floor d-1
    #     * c_alpha (= 30/44)         -> g CH2O m-2_floor d-1
    P_day = (max(Ag_leaf, 0.0) * _MW_CO2 * 1e-6
             * f_light * params.canopy_efficiency
             * params.photoperiod_s * params.c_alpha)

    # Maintenance respiration runs 24 h on the 24-h mean temperature, and is
    # charged against BOTH pools' worth of tissue -- structure only, since the
    # buffer is substrate rather than machinery that needs upkeep.
    resp_coeff = (params.c_resp_sht * (1.0 - params.c_tau)
                  + params.c_resp_rt * params.c_tau)
    R_day = (resp_coeff * W_struct
             * params.Q10_resp ** ((T_air_C - 25.0) / 10.0)
             * 86400.0)

    # -- 5. Structural growth -------------------------------------------
    # Van Henten's saturating rule: growth is proportional to the existing
    # structure (each gram of leaf builds more leaf) and is throttled by how
    # full the buffer is.
    #
    # CORRECTION vs. MATLAB: the MATLAB writes
    #     r_gr = c_gr_max * Ws/(Wg+Ws) * fT ; dWg = r_gr * Ws
    # which makes growth proportional to the BUFFER rather than to the
    # structure, and uses a plain fraction where Van Henten has a saturation
    # function.  With c_gamma = 1 the two denominators coincide, but the pool
    # the rate multiplies does not.
    f_growth_T = params.Q10_growth ** ((T_air_C - 20.0) / 10.0)
    r_gr = (params.c_gr_max
            * W_buffer / max(params.c_gamma * W_struct + W_buffer, 1e-9)
            * f_growth_T * 86400.0)                 # [d-1]
    r_gr *= ec_growth_factor(params, EC_mS_cm)      # 1.0 unless enabled

    dW_struct = r_gr * W_struct

    # Growth cannot outrun the buffer: laying down 1 g of structure costs
    # 1/c_beta g of CH2O (the extra is synthesis respiration).
    cost_per_g = 1.0 / params.c_beta
    available = max(W_buffer + P_day - R_day, 0.0)
    dW_struct = min(dW_struct, available / cost_per_g)
    dW_struct = max(dW_struct, 0.0)

    # -- 6. Update the pools ---------------------------------------------
    W_buffer_new = W_buffer + P_day - R_day - dW_struct * cost_per_g
    W_struct_new = W_struct + dW_struct

    # A buffer cannot go negative.  If respiration exceeds supply the crop
    # would in reality remobilise structure; over a 28-day lettuce cycle in a
    # heated tent that never happens, so clamping is safe and any clamping
    # event is reported so it cannot pass unnoticed.
    buffer_clamped = W_buffer_new < 0.0
    W_buffer_new = max(W_buffer_new, 0.0)

    GDD_new = state["GDD"] + max(0.0, T_air_C - params.GDD_base_C)

    fw = fresh_weight_per_plant(params, W_struct_new, W_buffer_new)

    return {
        # state
        "W_struct": W_struct_new,
        "W_buffer": W_buffer_new,
        "GDD": GDD_new,
        # diagnostics
        "fresh_weight_g": fw,
        "LAI": leaf_area_index(params, W_struct_new),
        "f_light": f_light,
        "Ag_leaf_umol_m2_s": Ag_leaf,
        "An_leaf_umol_m2_s": An_leaf,
        "P_gross_gCH2O_m2_d": P_day,
        "R_maint_gCH2O_m2_d": R_day,
        "dW_struct_g_m2_d": dW_struct,
        "r_gr_per_d": r_gr,
        "transpiration_mol_m2_s_leaf": E_leaf,
        "VPD_kPa": float(gas["VPD"][0]),
        "Ci_ppm": float(gas["Ci"][0]),
        "buffer_clamped": bool(buffer_clamped),
    }


# ---------------------------------------------------------------------------
# A whole trial
# ---------------------------------------------------------------------------
def simulate_trial(drivers: pd.DataFrame, params: CropParams,
                   optics: LeafOptics | None = None,
                   state: dict | None = None) -> pd.DataFrame:
    """Run the model over a trial's daily driver table.

    Parameters
    ----------
    drivers : DataFrame
        Indexed by DAT, with the columns written by `prepare_drivers.py`.
    params : CropParams
    optics : LeafOptics, optional
    state : dict, optional
        Starting state; defaults to `initial_state(params)`.

    Returns
    -------
    DataFrame indexed by DAT.  Row DAT=0 is the state AT transplant (nothing
    has been integrated yet), so `fresh_weight_g` there equals
    `params.initial_fresh_weight_g` by construction.  Row DAT=n is the state
    at the end of day n.
    """
    optics = optics or default_optics()
    state = dict(state or initial_state(params))

    rows = []
    # Day 0 = the transplant state itself, before any integration.
    rows.append({
        "DAT": int(drivers.index[0]),
        "W_struct": state["W_struct"],
        "W_buffer": state["W_buffer"],
        "GDD": state["GDD"],
        "fresh_weight_g": fresh_weight_per_plant(
            params, state["W_struct"], state["W_buffer"]),
        "LAI": leaf_area_index(params, state["W_struct"]),
        "f_light": light_interception(params, state["W_struct"]),
    })

    for dat, row in drivers.iloc[1:].iterrows():
        out = simulate_day(
            state, params,
            T_air_C=float(row["T_air_C"]),
            T_air_light_C=float(row.get("T_air_light_C", np.nan)),
            RH_pct=float(row["RH_pct"]),
            RH_light_pct=float(row.get("RH_light_pct", np.nan)),
            CO2_ppm=float(row["CO2_ppm"]),
            PPFD_umol_m2_s=float(row["PPFD_umol_m2_s"]),
            P_air_Pa=float(row.get("P_air_Pa", 101325.0)),
            EC_mS_cm=float(row.get("EC_mS_cm", np.nan)),
            optics=optics,
        )
        state = {k: out[k] for k in ("W_struct", "W_buffer", "GDD")}
        out["DAT"] = int(dat)
        rows.append(out)

    result = pd.DataFrame(rows).set_index("DAT")
    result.attrs["canopy_efficiency"] = params.canopy_efficiency
    return result
