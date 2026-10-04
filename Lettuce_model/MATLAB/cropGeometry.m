function varargout = cropGeometry(action, p, varargin)
%CROPGEOMETRY  Canopy and biomass conversions for the lettuce crop model.
%
%   s   = cropGeometry('initialState',  p)
%   LAI = cropGeometry('LAI',           p, Wstruct)
%   f   = cropGeometry('lightFraction', p, Wstruct)
%   FW  = cropGeometry('freshWeight',   p, Wstruct, Wbuffer)
%   fEC = cropGeometry('ecFactor',      p, ECmScm)
%   Wh  = cropGeometry('structAtTarget', p)
%
% Collected here rather than scattered as separate files so the whole
% dry-matter <-> leaf-area <-> fresh-weight chain reads in one place.  Every
% one of these conversions is a place where an error would be invisible in
% the output but wrong by tens of grams at harvest.

switch lower(action)

% ----------------------------------------------------------------------
case 'initialstate'
    % State at transplant, derived from the OBSERVED transplant fresh weight.
    %
    % Anchoring to the measurement rather than a literature seedling weight
    % matters more than it looks: growth is near-exponential for the first
    % ~12 days, so a 30% error in starting mass is still a 30% error two
    % weeks later.  Both trials transplanted at the same 4.7 g median net
    % fresh weight, so both start identically -- any later divergence is the
    % model responding to the drivers, not to different initial conditions.

    % Shoot fresh weight per plant -> shoot dry matter per m2 of bench.
    shootDMperM2 = p.initialFreshWeightG * p.plantingDensity * ...
                   p.dryMatterFraction / max(p.usableFraction, 1e-6);

    % Split into buffer and structure, then scale structure up from
    % shoot-only to whole-plant, since Wstruct includes roots (fraction cTau).
    buffer      = p.initialBufferFraction * shootDMperM2;
    structShoot = shootDMperM2 - buffer;

    s.Wstruct = structShoot / (1 - p.cTau);
    s.Wbuffer = buffer;
    s.GDD     = 0;
    varargout{1} = s;

% ----------------------------------------------------------------------
case 'structattarget'
    % Structural dry matter per m2 at the target head weight.  Used only as
    % the maturity scale over which specific leaf area declines.
    shootDM = p.targetHarvestWeightG * p.plantingDensity * ...
              p.dryMatterFraction / max(p.usableFraction, 1e-6);
    varargout{1} = shootDM / (1 - p.cTau);

% ----------------------------------------------------------------------
case 'lai'
    % Leaf area index [m2_leaf m-2_floor].  Only the shoot share of
    % structural dry matter carries leaf area, hence the (1 - cTau) factor.
    %
    %  'constant'   LAI = cLar * (1-cTau) * Wstruct.  Van Henten's original.
    %  'declining'  Specific leaf area falls linearly with accumulated mass,
    %               from slaMax at transplant to slaMin at the target head
    %               weight, because leaves thicken as the head fills.
    %               Integrating dLAI = SLA(W) dW from 0 to W gives the closed
    %               form below.  Past the target mass the canopy keeps
    %               thickening at slaMin.
    %
    % CORRECTION vs. the legacy simulateLettuceGrowthSingleDay.m, which uses
    %   f_light = 1 - exp(-0.7 * c_lar * Wg)
    % dropping the (1-cTau) shoot fraction and using an extinction
    % coefficient of 0.7 where Van Henten specifies 0.9.
    W         = max(varargin{1}, 0);
    shootFrac = 1 - p.cTau;

    if strcmpi(p.larModel, 'constant')
        varargout{1} = p.cLar * shootFrac * W;
        return
    end

    Wh   = cropGeometry('structAtTarget', p);
    Weff = min(W, Wh);
    lai  = shootFrac * (p.slaMax * Weff - ...
                        (p.slaMax - p.slaMin) * Weff.^2 / (2*Wh));
    lai  = lai + shootFrac * p.slaMin * max(W - Wh, 0);
    varargout{1} = max(lai, 0);

% ----------------------------------------------------------------------
case 'lightfraction'
    % Fraction of incident PPFD intercepted by the canopy [-], Beer-Lambert.
    %
    % Saturates near 1 at LAI ~4-5, which for these parameters happens late
    % in the cycle.  That saturation is the mechanism that bends the growth
    % curve from exponential to linear -- get it wrong and the shape of the
    % whole trajectory is wrong, which is exactly what happens with the
    % legacy cLar (see lettuceParams.m).
    LAI = cropGeometry('LAI', p, varargin{1});
    varargout{1} = 1 - exp(-p.cK * LAI);

% ----------------------------------------------------------------------
case 'freshweight'
    % Harvestable shoot fresh weight per plant [g FW plant-1].
    %
    % BOTH pools count: the buffer is real dry matter sitting in the leaves.
    %
    % CORRECTION vs. the legacy MATLAB, whose
    %   Biomass_fresh = Wg / dry_matter_fraction * usable_fraction
    % uses one pool only and does not remove the root share, so it reports
    % whole-plant structure as if it were marketable shoot.
    Wstruct = max(varargin{1}, 0);
    Wbuffer = max(varargin{2}, 0);
    shootDM     = (1 - p.cTau) * (Wstruct + Wbuffer);
    freshPerM2  = shootDM / p.dryMatterFraction * p.usableFraction;
    varargout{1} = freshPerM2 / p.plantingDensity;

% ----------------------------------------------------------------------
case 'ecfactor'
    % Optional empirical EC modifier on structural growth [-].
    %
    % Two limbs, after Samarakoon et al. (2020) for NFT lettuce: below the
    % optimum a Michaelis-Menten rise in nutrient availability, above it a
    % linear osmotic/transport penalty.
    %
    % Returns 1.0 unless p.ecResponse is true.  Read the note in
    % lettuceParams.m before switching it on -- it moves EC signal out of the
    % residual this project's image analysis is meant to explain.
    EC = varargin{1};
    if ~p.ecResponse || ~isfinite(EC)
        varargout{1} = 1.0;
        return
    end
    EC  = max(EC, 0);
    k   = p.ecHalfSaturationMScm;
    opt = p.ecOptimumMScm;
    if EC <= opt
        rise = (EC/(k+EC)) / (opt/(k+opt));          % 1.0 at the optimum
        varargout{1} = min(max(rise, 0), 1);
    else
        varargout{1} = min(max(1 - p.ecSupraSlopePerMS*(EC-opt), 0), 1);
    end

% ----------------------------------------------------------------------
otherwise
    error('cropGeometry:unknownAction', 'Unknown action "%s".', action);
end
end
