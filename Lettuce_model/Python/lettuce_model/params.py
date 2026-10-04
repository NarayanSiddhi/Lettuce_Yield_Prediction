"""Parameter set for the grow-tent lettuce crop model.

Every parameter below carries its units and its provenance.  Three kinds of
parameter appear here and they should be treated very differently:

  (a) PHYSIOLOGY constants taken from the literature (Van Henten 1994;
      Farquhar et al. 1980; Kattge & Knorr 2007).  Do not tune these to make
      a curve fit -- if the model misses, the miss is information.

  (b) PRODUCTION-SYSTEM descriptors of THIS grow tent (plant number, floor
      area, photoperiod, dry-matter fraction).  These are measurements.  If a
      better measurement exists, replace the number and say so.

  (c) ONE calibration coefficient, `canopy_efficiency`, which absorbs
      everything the single-leaf FvCB upscaling cannot represent (vertical
      light gradients, diffuse redistribution inside the tent, reflection off
      the mylar walls, cultivar vigour).  This is the only knob that
      `calibrate.py` is allowed to move.  Keeping the fit to a single scalar
      is deliberate: with eight fresh-weight checkpoints per trial, a model
      with several free parameters would fit the noise.

Variable naming
---------------
The legacy MATLAB in `C:\\CEAC_proposal_modeling\\Lettuce_model` calls the two
state variables `Ws` and `Wg` and labels them "shoot" and "root".  Those labels
do not match how the equations use them (fresh weight and leaf area are both
computed from `Wg`, which the comment calls root).  This module uses the
Van Henten names instead, which the equations actually follow:

    W_struct  (X_sdw)   structural dry weight  [g DM m-2 floor]
    W_buffer  (X_nsdw)  non-structural dry weight, the assimilate pool
                        that photosynthesis fills and growth drains
                                               [g CH2O m-2 floor]

See `../MATLAB/README.txt` and `growth.py` for the full mapping.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass
class CropParams:
    """Complete parameter set for one lettuce production system."""

    # =====================================================================
    # (a) PHOTOSYNTHESIS -- FvCB leaf biochemistry
    #     Farquhar, von Caemmerer & Berry (1980) Planta 149:78-90.
    #     Temperature responses after Kattge & Knorr (2007).
    #     These are the values the CEAC web tool runs lettuce with.
    # =====================================================================
    Vcmax25: float = 70.0
    """Maximum Rubisco carboxylation rate at 25 C [umol CO2 m-2_leaf s-1].
    Sets the CO2- and light-saturated ceiling on assimilation."""

    rjv: float = 1.8
    """Jmax/Vcmax ratio [-].  Jmax25 = rjv * Vcmax25; controls how quickly
    photosynthesis saturates with light relative to with CO2."""

    g1: float = 104.0
    """Leuning (1995) stomatal slope [-].  Larger g1 => stomata stay open
    wider at a given VPD, so assimilation is less VPD-limited.  This is what
    makes the model respond to the tent's VPD control at all."""

    g0: float = 0.0567
    """Residual (cuticular) stomatal conductance to CO2 [mol m-2 s-1].
    Floor on conductance when the Leuning term goes to zero."""

    alpha: float = 0.2
    """Quantum yield of electron transport [mol e- (mol absorbed photon)-1].
    Initial slope of the J vs. PPFD light-response curve."""

    theta: float = 0.7
    """Curvature of the non-rectangular hyperbola for J [-].  theta -> 1 gives
    a sharp Blackman-style knee, theta -> 0 a smooth rectangular hyperbola."""

    photosynthesis_type: int = 3
    """3 = C3 (lettuce), 4 = C4.  Only 3 is exercised here."""

    ra: float = 50.0
    """Aerodynamic resistance above the canopy [s m-1].  Low in a tent with an
    oscillating clip fan running continuously."""

    rb: float = 50.0
    """Leaf boundary-layer resistance [s m-1]."""

    # =====================================================================
    # (a) GROWTH AND RESPIRATION -- Van Henten (1994) Agricultural Systems
    #     45:55-72, "Validation of a dynamic lettuce growth model for
    #     greenhouse climate control".  A copy is in ../../Literature/.
    # =====================================================================
    c_alpha: float = 0.68
    """CO2 -> CH2O conversion [g CH2O (g CO2)-1].  = 30/44, the ratio of
    molar masses.  Van Henten's c_alpha = 0.68."""

    c_beta: float = 0.8
    """Growth (synthesis) efficiency [g structural DM (g CH2O)-1].  The
    complement, (1-c_beta)/c_beta, is the growth-respiration overhead charged
    to the buffer for every gram of structure laid down."""

    c_gr_max: float = 5.0e-6
    """Saturation structural growth rate at 20 C [s-1].  Van Henten p.5:
    "From measurements of Van Holsteijn (1981) c_gr,max was estimated as
    5 x 10^-6 s-1."  NOTE the legacy MATLAB masterfile uses 1e-6, five times
    smaller, which is why a 250 g head there takes ~70 d instead of ~30 d."""

    c_gamma: float = 1.0
    """Saturation constant of the growth-rate response to buffer filling [-].
    r_gr = c_gr_max * W_buffer / (c_gamma*W_struct + W_buffer) * fT."""

    c_resp_sht: float = 3.47e-7
    """Shoot maintenance respiration coefficient at 25 C [g CH2O (g DM)-1 s-1]."""

    c_resp_rt: float = 1.16e-7
    """Root maintenance respiration coefficient at 25 C [g CH2O (g DM)-1 s-1].
    Roots respire ~3x less per gram than shoots."""

    c_tau: float = 0.15
    """Root fraction of structural dry weight [-].  Constant partitioning; a
    reasonable simplification for lettuce, which does not shift allocation
    much over a 28-day cycle."""

    Q10_resp: float = 2.0
    """Q10 of maintenance respiration [-].  Respiration doubles per +10 C."""

    Q10_growth: float = 1.6
    """Q10 of the structural growth rate [-]."""

    # ---- Canopy light interception (Beer-Lambert) ----
    lar_model: str = "declining"
    """How leaf area is derived from structural dry matter:
        'declining' specific leaf area falls linearly from sla_max at
                    transplant to sla_min at harvest mass (DEFAULT)
        'constant'  a single leaf area ratio, c_lar, at all masses
                    (Van Henten's original form)"""

    sla_max: float = 0.030
    """Specific leaf area of young lettuce leaves [m2_leaf (g shoot DM)-1].
    = 300 cm2 g-1.  Seedling leaves are thin and expand fast."""

    sla_min: float = 0.015
    """Specific leaf area at harvest maturity [m2_leaf (g shoot DM)-1].
    = 150 cm2 g-1.  Leaves thicken and accumulate dry matter as the head
    fills, so the same gram of dry matter buys progressively less area."""

    c_lar: float = 0.075
    """Constant structural leaf area ratio [m2_leaf (g structural DM)-1],
    used only when lar_model == 'constant'.

    This is Van Henten's published value and it is NOT the default here, for
    a reason worth recording.  With c_lar = 0.075 the canopy reaches LAI 9.2
    at a 220 g head and intercepts 98% of incident light by DAT 20.  Neither
    is true of this crop: lettuce at a 200-250 g head carries LAI 4-5, and
    the DAT 20 camera frames show wide gaps between plants with the white
    gutters plainly visible.  The consequence in the model is that growth
    goes linear around DAT 20 whereas the measured crop stayed close to
    exponential to harvest -- the model saturates on light capture that the
    real canopy had not yet achieved.

    The 'declining' default instead builds LAI from a measured lettuce
    specific leaf area (sla_max/sla_min above, 150-300 cm2 g-1, the usual
    published range), which puts LAI at ~4.6 for a 227 g head.  That is a
    correction on physical grounds, not a fit: neither SLA value was tuned
    against the fresh-weight data."""

    c_K: float = 0.9
    """Canopy extinction coefficient [-].  Van Henten uses 0.9 for lettuce's
    near-horizontal leaf display, higher than the ~0.7 typical of erectophile
    canopies.  f_light = 1 - exp(-c_K * LAI)."""

    # =====================================================================
    # (c) THE ONE CALIBRATION COEFFICIENT
    # =====================================================================
    canopy_efficiency: float = 1.0
    """Multiplier on gross canopy assimilation [-].

    A single-leaf FvCB rate multiplied by an interception fraction
    systematically under-predicts whole-canopy assimilation, because leaves
    lower in the canopy sit on the steep part of their light-response curve
    and the mylar tent walls return light that a big-leaf model treats as
    lost.  Heuvelink (2005) reports canopy photosynthesis 2-3x above naive
    single-leaf upscaling in deep canopies; a 30 cm lettuce canopy is not
    deep, so values well above ~2 should be read as the model compensating
    for something else and investigated rather than accepted.

    Fitted per trial by `calibrate.calibrate_canopy_efficiency`."""

    # =====================================================================
    # (b) PRODUCTION SYSTEM -- measurements of THIS grow tent
    #     Sources: thesis Ch.3 (Tables 3.1-3.3) and the camera captures.
    # =====================================================================
    n_plants: int = 30
    """Plants in the NFT unit: 5 gutters x 6 net-pot positions.
    Confirmed by the Trial 2 wide-angle capture on 2026-01-20."""

    growing_area_m2: float = 1.49
    """Canopy floor area of the NFT bench [m2].

    ESTIMATE, not a measurement -- 1.22 x 1.22 m, the footprint of the
    standard 4 ft x 4 ft grow tent the system appears to occupy.  It sets
    planting_density, which scales biomass per m2 to biomass per plant, so a
    10% error here is a 10% error in predicted head weight.  MEASURE THE
    BENCH and replace this number before quoting absolute yields."""

    dry_matter_fraction: float = 0.045
    """Shoot dry matter fraction [g DM (g FW)-1].  Hydroponic leaf lettuce
    typically runs 4-5%; 4.5% is the midpoint.  Not measured in these trials
    (only fresh weight was recorded), so this is the largest single
    uncertainty in converting modelled dry matter to the observed fresh
    weight.  The CEAC web tool uses 0.04."""

    usable_fraction: float = 1.0
    """Fraction of shoot fresh weight counted as yield [-].

    1.0 here on purpose: the trials weighed the whole shoot and then
    subtracted a flat 25 g for saturated rockwool and net pot, they did not
    trim to a marketable head.  The CEAC web tool uses 0.85 because it models
    a marketable head.  Do not mix the two conventions."""

    photoperiod_h: float = 16.0
    """Light period [h d-1].  05:00-21:00 America/Los_Angeles, Mycodo Daily
    Trigger on GPIO 22, identical in both trials."""

    target_harvest_weight_g: float = 227.0
    """Marketable target fresh weight per plant [g].  The steering target the
    trials were run against (thesis Ch.3)."""

    # ---- Initial state ----
    initial_fresh_weight_g: float = 4.7
    """Observed median net fresh weight per plant at transplant (DAT 0) [g].
    Same value in both trials.  `growth.initial_state` converts this to
    structural dry matter, so the model starts where the crop actually did
    rather than at a literature seedling weight."""

    initial_buffer_fraction: float = 0.25
    """Non-structural share of shoot dry matter at transplant [-].
    Young, rapidly expanding lettuce carries a large soluble pool; 0.20-0.30
    is the usual range.  Only affects the first few days."""

    GDD_base_C: float = 4.0
    """Base temperature for growing degree days [C].  Reported, not used to
    drive growth -- the carbon balance already carries the temperature
    response through Q10 and the FvCB temperature functions."""

    # =====================================================================
    # OPTIONAL: empirical EC response  (OFF by default -- read this)
    # =====================================================================
    ec_response: bool = False
    """Enable the empirical EC growth modifier below.

    Left OFF by default, deliberately.  The mechanistic core has no nutrient
    module: it assumes nutrients are non-limiting and predicts growth from
    light, temperature, CO2 and humidity alone.  That is exactly what makes
    it useful for this project -- the model-minus-observed residual is then a
    clean target for the image + EC analysis to explain.  Switching this on
    moves part of the EC signal INTO the model and out of the residual, which
    is a different (and much weaker) experiment.  Turn it on only to test how
    much of the trial difference a published EC response alone can account
    for."""

    ec_optimum_mS_cm: float = 1.8
    """EC at which lettuce fresh weight plateaus in NFT [mS cm-1].
    Samarakoon et al. (2020): fresh weight rose to ~1.8 mS cm-1 and then
    flattened, with disorder incidence rising above it."""

    ec_half_saturation_mS_cm: float = 0.6
    """Half-saturation EC of the sub-optimal limb [mS cm-1].  Michaelis-Menten
    shape: f_EC = EC / (k + EC), normalised to 1.0 at ec_optimum."""

    ec_supra_slope_per_mS: float = 0.06
    """Relative growth lost per mS cm-1 above the optimum [-].  Osmotic and
    transport cost; ~6% per mS cm-1 is at the mild end of the published range
    for lettuce and is a placeholder, not a calibrated value."""

    # ---- Numerics ----
    temp_basis: str = "daily"
    """Which temperature/humidity average drives photosynthesis:
        'daily'       24-h mean (what the CEAC MATLAB masterfile does)
        'photoperiod' mean over the 16 h the lights are on

    'photoperiod' is the more physical choice -- assimilation only happens
    when the lights are on, so it should see the temperature that prevailed
    then.  'daily' is the default only so that results reproduce the web tool
    out of the box.  Maintenance respiration always uses the 24-h mean,
    because it runs around the clock."""

    def with_(self, **kwargs) -> "CropParams":
        """Return a copy with fields replaced.  `p.with_(canopy_efficiency=2)`."""
        return replace(self, **kwargs)

    # ---- Derived quantities -------------------------------------------
    @property
    def planting_density(self) -> float:
        """Plants per square metre of bench [plants m-2]."""
        return self.n_plants / self.growing_area_m2

    @property
    def photoperiod_s(self) -> float:
        """Light period [s d-1]."""
        return self.photoperiod_h * 3600.0


def lettuce_params(**overrides) -> CropParams:
    """Baseline grow-tent lettuce parameter set, optionally overridden."""
    return CropParams(**overrides)


# ---------------------------------------------------------------------------
# Per-trial parameter sets.
#
# The physiology is identical between trials, by design: the experiment held
# hardware, climate setpoints, photoperiod and control logic constant and
# varied only the EC steering strategy and the timing of the light-intensity
# increase.  Both of those enter through the DRIVER table, not through
# parameters, so the two entries below differ only in the calibrated
# canopy_efficiency (filled in by calibrate.py) and the label.
# ---------------------------------------------------------------------------
TRIAL_PARAMS = {
    1: lettuce_params(),
    2: lettuce_params(),
}
