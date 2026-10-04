function out = photosynthesisFvCB(PPFDabs, IPAR, TairC, RHpct, CaPpm, p, PairPa)
%PHOTOSYNTHESISFVCB  Leaf gas exchange: Farquhar-von Caemmerer-Berry + Leuning.
%
%   out = photosynthesisFvCB(PPFDabs, IPAR, TairC, RHpct, CaPpm, p)
%   out = photosynthesisFvCB(..., PairPa)
%
% Ported from photosynthesis_vec.m in the CEAC greenhouse web tool, with the
% four model switches that file exposes (ANS_TEMP, ANSG, ANSMOD, ANSTPU,
% ANSGS) fixed to the combination the lettuce model actually runs: Kattge &
% Knorr temperature responses, Leuning stomata, Daly light response, TPU
% limitation on.  Only one combination is ever used, so hard-wiring it makes
% the equations readable instead of hiding them behind flags.
%
% WHAT THE MODEL SAYS
%   Net assimilation is the smoothed minimum of three potential rates,
%   because a leaf can be short of any of three things:
%
%     JC  Rubisco-limited  -- not enough carboxylation capacity
%                             (low CO2, low Vcmax, cold)
%     JL  RuBP-limited     -- not enough electrons from the light reactions
%                             (low PPFD)
%     JE  TPU-limited      -- triose phosphate export saturates, so product
%                             accumulation throttles the chloroplast
%                             (high CO2 + high light + cool leaf)
%
%   They are combined by two nested non-rectangular hyperbolae rather than a
%   hard min(), because real leaves co-limit smoothly and a hard minimum puts
%   a kink in the derivative that upsets anything optimising through it.
%
%   Then the stomatal step: conductance responds to assimilation and to VPD
%   via Leuning (1995), that conductance sets the CO2 drawdown from leaf
%   surface to chloroplast, and assimilation is recomputed at the new
%   internal CO2.  This is what makes the model care about the tent's VPD
%   control -- the trials held daytime VPD at 0.8-1.2 kPa, and it is here
%   that a dry hour costs carbon.
%
% UNITS -- read before editing
%   Internally, partial pressures are in Pa, not ppm.  Concentrations arrive
%   in ppm and are multiplied by 1e-6*P.  Resistances arrive in s m-1 and are
%   converted to the m2 s umol-1 form the CO2 flux equation needs.  Mixing
%   the two conventions is the easiest way to break this file.
%
% INPUTS   (all per m^2 of LEAF)
%   PPFDabs  absorbed photon flux            [umol photons m-2 s-1]
%   IPAR     quantum-yield-weighted flux     [umol m-2 s-1]
%   TairC    air (= leaf) temperature        [C]
%   RHpct    relative humidity               [%]
%   CaPpm    ambient CO2                     [ppm]
%   p        parameter struct from lettuceParams
%   PairPa   air pressure [Pa], default 101325
%
% OUTPUT struct
%   An     net assimilation            [umol CO2 m-2 s-1]
%   Ag     gross assimilation, An+Rd   [umol CO2 m-2 s-1]
%   Rdark  dark respiration            [umol CO2 m-2 s-1]
%   gs     stomatal conductance to CO2 [umol m-2 s-1]
%   J      electron transport rate     [umol e- m-2 s-1]
%   Ci     chloroplast CO2             [ppm]
%   VPD    vapour pressure deficit     [kPa]
%   E      transpiration               [mol H2O m-2 s-1]
%
% REFERENCES
%   Farquhar G.D., von Caemmerer S., Berry J.A. (1980) Planta 149:78-90.
%   Leuning R. (1995) Plant Cell Environ 18:1183-1200.
%   Kattge J., Knorr W. (2007) Plant Cell Environ 30:1176-1190.

if nargin < 7 || isempty(PairPa), PairPa = 101325; end

Tf   = 273.15;      % K
Pref = 101325;      % reference pressure [Pa]
Rkj  = 0.008314;    % gas constant [kJ mol-1 K-1]

Ts    = TairC(:);
Tk    = Ts + Tf;
TrefK = 25 + Tf;

% ---- Humidity ---------------------------------------------------------
es  = 0.6108 .* exp(17.27 .* Ts ./ (Ts + 237.3));       % Tetens, [kPa]
VPD = max(es .* (1 - RHpct(:)/100), 0.001);             % floored: gs -> Inf at 0

% ---- Unit conversions -------------------------------------------------
% Resistances from s m-1 to (m2 s umol-1) via the molar volume of air.
conv = 0.0224 .* Tk .* Pref ./ (Tf .* PairPa) .* 1e-6;
ra   = p.ra .* conv;
rb   = p.rb .* conv;
rMes = 1 / (1e6 * 0.60);              % fixed mesophyll conductance

CaPa = CaPpm(:) .* 1e-6 .* PairPa;
O2Pa = 209000  .* 1e-6 .* PairPa;     % 20.9% O2

% ---- Temperature-dependent capacities ---------------------------------
% Peaked Arrhenius: rises with activation energy Ha, then falls as the
% enzyme deactivates above its optimum (Hd, dS).  This is what gives Vcmax
% and Jmax an optimum near 30-35 C instead of rising without bound.
kT = @(Ha,Hd,dS) exp(Ha .* (Tk - TrefK) ./ (TrefK .* Rkj .* Tk)) .* ...
                 (1 + exp((TrefK.*dS - Hd) ./ (TrefK .* Rkj))) ./ ...
                 (1 + exp((Tk   .*dS - Hd) ./ (Tk    .* Rkj)));

Vcmax = p.Vcmax25          .* kT(72.0, 200.00, 0.650);
Jmax  = p.Vcmax25 * p.rjv  .* kT(50.0, 200.00, 0.650);
TPU   = 0.1182 * p.Vcmax25 .* kT(53.1, 150.65, 0.490);

% CO2 compensation point without dark respiration, Gamma* [Pa].
gammaStar = 34.6 .* (1 + 0.0451.*(Tk - TrefK) + 0.000347.*(Tk - TrefK).^2);
gammaStar = gammaStar .* 1e-6 .* PairPa;

% Rubisco Michaelis constants for CO2 and O2 [Pa].
Kc = 302.0 .* 1e-6 .* PairPa .* exp(59.43 .* (Tk - TrefK) ./ (TrefK .* Rkj .* Tk));
Ko = 256.0 .* 1e-3 .* PairPa .* exp(36.00 .* (Tk - TrefK) ./ (TrefK .* Rkj .* Tk));

% ---- Electron transport rate J ----------------------------------------
% Non-rectangular hyperbola in absorbed light, saturating at Jmax.
J = smoothMin(p.alpha .* IPAR(:), Jmax, p.theta);
J = max(J, 0);

% ---- Dark respiration --------------------------------------------------
if p.CT == 3
    Rdark = 0.015 * p.Vcmax25 .* kT(46.39, 150.65, 0.490);
else
    Rdark = 0.025 * p.Vcmax25 .* 2.0.^(0.1*(Ts-25)) ./ (1 + exp(1.3*(Ts-55)));
end

% ---- Assimilation at ambient CO2 ---------------------------------------
AgFirst = assimilate(CaPa);
AnFirst = AgFirst - Rdark;

% ---- Stomatal loop -----------------------------------------------------
% One pass, as in the legacy MATLAB: assimilate at ambient CO2, get
% conductance from that rate, recompute the flux at the resulting drawdown.
% A fixed-point iteration moves the answer well under 1% here, because gs is
% large relative to the CO2 gradient in a ventilated tent.
g0umol = p.g0 * 1e6;
gs = g0umol + (1 + p.g1 ./ sqrt(VPD)) .* AnFirst .* PairPa ./ max(CaPa, 1e-10);
gs = max(gs, g0umol);

rTotal = 1 ./ max(gs, 1e-10) + rMes + 1.37 .* rb + ra;
CcPa   = max(CaPa - AnFirst .* PairPa .* rTotal, 0);

An = (CaPa - CcPa) ./ (PairPa .* rTotal);
Ag = An + Rdark;

% ---- Transpiration ------------------------------------------------------
% gs is for CO2; water vapour diffuses 1.6x faster.
gsH2O = (gs / 1e6) * 1.6;                     % [mol m-2 s-1]
E     = VPD .* gsH2O ./ (PairPa / 1000);      % [mol H2O m-2 s-1]

out = struct('An', An, 'Ag', Ag, 'Rdark', Rdark, 'gs', gs, 'J', J, ...
             'Ci', CcPa ./ (PairPa .* 1e-6), 'VPD', VPD, 'E', E);

% ======================================================================
    function A = assimilate(Cc)
        % Gross assimilation at a given chloroplast CO2 partial pressure.
        JC = Vcmax .* (Cc - gammaStar) ./ max(Cc + Kc .* (1 + O2Pa ./ Ko), 1e-10);
        JL = (J/4) .* (Cc - gammaStar) ./ max(Cc + 2*gammaStar, 1e-10);
        JE = 3 * TPU;
        % Curvature 0.9 for both co-limitations, as in the legacy MATLAB.
        A  = smoothMin(smoothMin(JC, JL, 0.9), JE, 0.9);
    end
end

function y = smoothMin(a, b, curvature)
%SMOOTHMIN  Lower root of  curvature*x^2 - (a+b)*x + a*b = 0.
%   A co-limitation operator: equals min(a,b) as curvature -> 1, and lies
%   smoothly below both when curvature < 1.
disc = sqrt(max((a + b).^2 - 4 .* curvature .* a .* b, 0));
y    = ((a + b) - disc) ./ (2 .* curvature);
end
