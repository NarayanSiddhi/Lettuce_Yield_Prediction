function sim = simulateLettuceTrial(drivers, p, optics, state0)
%SIMULATELETTUCETRIAL  Run the crop model over a trial's daily driver table.
%
%   sim = simulateLettuceTrial(drivers, p)
%   sim = simulateLettuceTrial(drivers, p, optics, state0)
%
% INPUTS
%   drivers  table from loadDrivers(): one row per day after transplant, with
%            columns DAT, T_air_C, T_air_light_C, RH_pct, RH_light_pct,
%            CO2_ppm, P_air_Pa, EC_mS_cm, PPFD_umol_m2_s, DLI_mol_m2_d
%   p        parameter struct from lettuceParams
%   optics   struct from leafOptics (default: leafOptics())
%   state0   initial state (default: cropGeometry('initialState', p))
%
% OUTPUT
%   sim      table indexed by DAT.  Row DAT = 0 is the state AT transplant
%            (nothing has been integrated yet), so freshWeightG there equals
%            p.initialFreshWeightG by construction.  Row DAT = n is the state
%            at the end of day n.

if nargin < 3 || isempty(optics), optics = leafOptics(); end
if nargin < 4 || isempty(state0), state0 = cropGeometry('initialState', p); end

nDays = height(drivers);
state = state0;

DAT             = zeros(nDays,1);
Wstruct         = nan(nDays,1);
Wbuffer         = nan(nDays,1);
GDD             = nan(nDays,1);
freshWeightG    = nan(nDays,1);
LAI             = nan(nDays,1);
fLight          = nan(nDays,1);
PgrossGCH2OM2D  = nan(nDays,1);
RmaintGCH2OM2D  = nan(nDays,1);
AgLeafUmolM2S   = nan(nDays,1);
rGrPerD         = nan(nDays,1);
VPDkPa          = nan(nDays,1);
bufferClamped   = false(nDays,1);

% ---- Day 0: the transplant state itself, before any integration -------
DAT(1)          = drivers.DAT(1);
Wstruct(1)      = state.Wstruct;
Wbuffer(1)      = state.Wbuffer;
GDD(1)          = state.GDD;
freshWeightG(1) = cropGeometry('freshWeight', p, state.Wstruct, state.Wbuffer);
LAI(1)          = cropGeometry('LAI', p, state.Wstruct);
fLight(1)       = cropGeometry('lightFraction', p, state.Wstruct);

% ---- Integrate ---------------------------------------------------------
for i = 2:nDays
    drv = struct( ...
        'TairC',       drivers.T_air_C(i), ...
        'TairLightC',  getCol(drivers, 'T_air_light_C',  i, NaN), ...
        'RHpct',       drivers.RH_pct(i), ...
        'RHlightPct',  getCol(drivers, 'RH_light_pct',   i, NaN), ...
        'CO2ppm',      drivers.CO2_ppm(i), ...
        'PPFDumolM2S', drivers.PPFD_umol_m2_s(i), ...
        'PairPa',      getCol(drivers, 'P_air_Pa',       i, 101325), ...
        'ECmScm',      getCol(drivers, 'EC_mS_cm',       i, NaN));

    out   = simulateLettuceDay(state, p, drv, optics);
    state = struct('Wstruct', out.Wstruct, 'Wbuffer', out.Wbuffer, ...
                   'GDD', out.GDD);

    DAT(i)              = drivers.DAT(i);
    Wstruct(i)          = out.Wstruct;
    Wbuffer(i)          = out.Wbuffer;
    GDD(i)              = out.GDD;
    freshWeightG(i)     = out.freshWeightG;
    LAI(i)              = out.LAI;
    fLight(i)           = out.fLight;
    PgrossGCH2OM2D(i)   = out.PgrossGCH2OM2D;
    RmaintGCH2OM2D(i)   = out.RmaintGCH2OM2D;
    AgLeafUmolM2S(i)    = out.AgLeafUmolM2S;
    rGrPerD(i)          = out.rGrPerD;
    VPDkPa(i)           = out.VPDkPa;
    bufferClamped(i)    = out.bufferClamped;
end

sim = table(DAT, Wstruct, Wbuffer, GDD, freshWeightG, LAI, fLight, ...
            PgrossGCH2OM2D, RmaintGCH2OM2D, AgLeafUmolM2S, rGrPerD, ...
            VPDkPa, bufferClamped);
end

function v = getCol(tbl, name, i, fallback)
%GETCOL  Read an optional column, falling back when absent or NaN.
if ismember(name, tbl.Properties.VariableNames)
    v = tbl.(name)(i);
    if ~isfinite(v), v = fallback; end
else
    v = fallback;
end
end
