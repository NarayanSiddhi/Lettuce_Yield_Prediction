function p = lettuceParams(varargin)
%LETTUCEPARAMS  Parameter set for the grow-tent lettuce crop model.
%
%   p = lettuceParams()                       baseline parameters
%   p = lettuceParams('canopyEfficiency',1.4) with overrides
%
% Every parameter carries its units and its provenance.  THREE KINDS of
% parameter live here and they must be treated differently:
%
%   (a) PHYSIOLOGY constants from the literature (Van Henten 1994; Farquhar
%       et al. 1980; Kattge & Knorr 2007).  Do not tune these to improve a
%       fit.  If the model misses, the miss is information.
%
%   (b) PRODUCTION-SYSTEM descriptors of THIS grow tent (plant number, bench
%       area, photoperiod, dry matter fraction).  These are measurements.
%       Replace them when a better measurement exists, and say so.
%
%   (c) ONE calibration coefficient, canopyEfficiency, which absorbs
%       everything single-leaf upscaling cannot represent.  It is the only
%       knob calibrateCanopyEfficiency.m is allowed to move.  With eight
%       fresh-weight checkpoints per trial, a model with several free
%       parameters would fit the noise rather than the crop.
%
% STATE VARIABLE NAMING
%   The legacy CEAC MATLAB (C:\CEAC_proposal_modeling\Lettuce_model) calls
%   its two states Ws and Wg and labels them "shoot" and "root".  Those
%   labels do not match how its equations use them -- fresh weight and leaf
%   area are both computed from Wg, which the comment calls root.  This
%   implementation uses the Van Henten names the equations actually follow:
%
%       Wstruct  (X_sdw)   structural dry weight   [g DM m^-2 floor]
%       Wbuffer  (X_nsdw)  non-structural pool     [g CH2O m^-2 floor]
%
%   See README.txt for the full mapping and the list of corrections.
%
% See also SIMULATELETTUCEDAY, SIMULATELETTUCETRIAL, RUN_LETTUCE_TRIAL.

% =====================================================================
% (a) PHOTOSYNTHESIS -- FvCB leaf biochemistry
%     Farquhar, von Caemmerer & Berry (1980) Planta 149:78-90.
%     Temperature responses after Kattge & Knorr (2007).
%     Values as used by the CEAC greenhouse web tool for lettuce.
% =====================================================================
p.Vcmax25   = 70.0;    % max Rubisco carboxylation at 25 C [umol CO2 m-2_leaf s-1]
p.rjv       = 1.8;     % Jmax/Vcmax ratio [-]; sets how fast A saturates with light
p.g1        = 104.0;   % Leuning stomatal slope [-]; larger => less VPD-limited
p.g0        = 0.0567;  % residual (cuticular) conductance to CO2 [mol m-2 s-1]
p.alpha     = 0.2;     % quantum yield of electron transport [mol e- / mol photon]
p.theta     = 0.7;     % curvature of the J light-response hyperbola [-]
p.CT        = 3;       % 3 = C3 (lettuce), 4 = C4
p.ra        = 50.0;    % aerodynamic resistance above canopy [s m-1]
p.rb        = 50.0;    % leaf boundary-layer resistance [s m-1]

% =====================================================================
% (a) GROWTH AND RESPIRATION -- Van Henten (1994) Agric. Systems 45:55-72
%     "Validation of a dynamic lettuce growth model for greenhouse climate
%      control".  A copy is in ..\..\Literature\.
% =====================================================================
p.cAlpha    = 0.68;    % CO2 -> CH2O conversion [g CH2O / g CO2] (= 30/44)
p.cBeta     = 0.8;     % synthesis efficiency [g structural DM / g CH2O];
                       % (1-cBeta)/cBeta is the growth-respiration overhead
p.cGrMax    = 5.0e-6;  % saturation structural growth rate at 20 C [s-1].
                       % Van Henten p.5, from Van Holsteijn (1981).  NOTE the
                       % legacy CEAC masterfile uses 1e-6, five times smaller,
                       % which is why a 250 g head takes ~70 d there.
p.cGamma    = 1.0;     % saturation constant of the growth response [-]
p.cRespSht  = 3.47e-7; % shoot maintenance respiration at 25 C [g CH2O /g DM /s]
p.cRespRt   = 1.16e-7; % root  maintenance respiration at 25 C [g CH2O /g DM /s]
p.cTau      = 0.15;    % root fraction of structural dry weight [-]
p.Q10resp   = 2.0;     % Q10 of maintenance respiration [-]
p.Q10growth = 1.6;     % Q10 of the structural growth rate [-]

% ---- Canopy light interception (Beer-Lambert) ------------------------
p.larModel  = 'declining';  % 'declining' (default) or 'constant'
p.slaMax    = 0.030;   % SLA of young leaves      [m2_leaf / g shoot DM] = 300 cm2/g
p.slaMin    = 0.015;   % SLA at harvest maturity  [m2_leaf / g shoot DM] = 150 cm2/g
p.cLar      = 0.075;   % constant leaf area ratio [m2_leaf / g structural DM],
                       % used only when larModel == 'constant'.
%
% cLar = 0.075 is Van Henten's published value and it is NOT the default
% here, for a reason worth recording.  With it the canopy reaches LAI 9.2 at
% a 220 g head and intercepts 98% of incident light by DAT 20.  Neither is
% true of this crop: lettuce at a 200-250 g head carries LAI 4-5, and the
% DAT 20 camera frames show wide gaps between plants with the white gutters
% plainly visible.  In the model that premature saturation turns growth
% linear around DAT 20, whereas the measured crop stayed close to
% exponential to harvest.  The 'declining' default instead builds LAI from a
% measured lettuce specific leaf area (150-300 cm2/g, the usual published
% range), giving LAI ~4.6 at 227 g.  Neither SLA value was tuned against the
% fresh-weight data -- this is a correction on physical grounds, not a fit.

p.cK        = 0.9;     % canopy extinction coefficient [-]; Van Henten's value
                       % for lettuce's near-horizontal leaf display

% =====================================================================
% (c) THE ONE CALIBRATION COEFFICIENT
% =====================================================================
p.canopyEfficiency = 1.0;
% Multiplier on gross canopy assimilation [-].  A single-leaf FvCB rate times
% an interception fraction under-predicts whole-canopy assimilation, because
% lower leaves sit on the steep part of their light-response curve and the
% mylar tent walls return light a big-leaf model treats as lost.  Heuvelink
% (2005) reports 2-3x for deep canopies; a 30 cm lettuce canopy is not deep,
% so a fitted value well above ~2 should be investigated, not accepted.
% Fitted per trial by calibrateCanopyEfficiency.m.

% =====================================================================
% (b) PRODUCTION SYSTEM -- measurements of THIS grow tent
%     Sources: thesis Ch.3 (Tables 3.1-3.3) and the camera captures.
% =====================================================================
p.nPlants   = 30;      % 5 NFT gutters x 6 net-pot positions.  Confirmed by
                       % the Trial 2 wide-angle frame of 2026-01-20.
p.growingAreaM2 = 1.49;
% ESTIMATE, not a measurement -- 1.22 x 1.22 m, the footprint of the standard
% 4 ft x 4 ft grow tent the system appears to occupy.  It sets planting
% density, which converts biomass per m2 to biomass per plant, so a 10% error
% here is a 10% error in predicted head weight.  MEASURE THE BENCH and
% replace this number before quoting absolute yields.

p.dryMatterFraction = 0.045;
% Shoot dry matter fraction [g DM / g FW].  Hydroponic leaf lettuce runs
% 4-5%; 4.5% is the midpoint.  NOT measured in these trials (only fresh
% weight was recorded), so this is the largest single uncertainty in
% converting modelled dry matter to observed fresh weight.  CEAC uses 0.04.

p.usableFraction = 1.0;
% Fraction of shoot fresh weight counted as yield [-].  1.0 on purpose: the
% trials weighed the whole shoot and subtracted a flat 25 g for saturated
% rockwool and net pot, they did not trim to a marketable head.  CEAC uses
% 0.85 because it models a marketable head.  Do not mix the conventions.

p.photoperiodH = 16.0;     % 05:00-21:00 America/Los_Angeles, both trials
p.targetHarvestWeightG = 227.0;  % the steering target (thesis Ch.3)

% ---- Initial state ---------------------------------------------------
p.initialFreshWeightG = 4.7;
% Observed median net fresh weight per plant at transplant (DAT 0), the same
% in both trials.  initialState() converts this to structural dry matter, so
% the model starts where the crop actually did rather than at a literature
% seedling weight.  Growth is near-exponential for ~12 d, so a 30% error in
% starting mass is still a 30% error two weeks later.

p.initialBufferFraction = 0.25;
% Non-structural share of shoot dry matter at transplant [-].  Young lettuce
% carries a large soluble pool; 0.20-0.30 is the usual range.

p.GDDbaseC = 4.0;
% Base temperature for growing degree days [C].  Reported, not used to drive
% growth -- the carbon balance already carries temperature through Q10 and
% the FvCB temperature functions.

% =====================================================================
% OPTIONAL: empirical EC response  (OFF by default -- read this)
% =====================================================================
p.ecResponse = false;
% Left OFF deliberately.  The mechanistic core has no nutrient module: it
% assumes nutrients are non-limiting and predicts growth from light,
% temperature, CO2 and humidity alone.  That is exactly what makes it useful
% here -- the model-minus-observed residual is then a clean target for the
% image + EC analysis to explain.  Switching this on moves part of the EC
% signal INTO the model and out of the residual, which is a different and
% much weaker experiment.  Use it only to test how much of the trial
% difference a published EC response alone can account for.
p.ecOptimumMScm        = 1.8;   % FW plateaus here in NFT (Samarakoon 2020)
p.ecHalfSaturationMScm = 0.6;   % half-saturation of the sub-optimal limb
p.ecSupraSlopePerMS    = 0.06;  % relative growth lost per mS/cm above optimum
                                % (placeholder, not calibrated)

% ---- Numerics --------------------------------------------------------
p.tempBasis = 'daily';
% Which average drives photosynthesis:
%   'daily'       24-h mean (what the CEAC MATLAB masterfile does)
%   'photoperiod' mean over the 16 h the lights are on
% 'photoperiod' is the more physical choice -- assimilation only happens
% while the lights are on.  'daily' is the default only so results reproduce
% the web tool out of the box.  Maintenance respiration ALWAYS uses the 24-h
% mean, because it runs around the clock.

% ---- Apply name/value overrides -------------------------------------
for k = 1:2:numel(varargin)
    name = varargin{k};
    if ~isfield(p, name)
        error('lettuceParams:unknownParameter', ...
              'Unknown parameter "%s".', name);
    end
    p.(name) = varargin{k+1};
end

% ---- Derived ---------------------------------------------------------
p.plantingDensity = p.nPlants / p.growingAreaM2;   % [plants m-2]
p.photoperiodS    = p.photoperiodH * 3600;         % [s d-1]
end
