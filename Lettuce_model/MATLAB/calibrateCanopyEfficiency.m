function [ce, info] = calibrateCanopyEfficiency(drivers, observed, p, optics, bounds)
%CALIBRATECANOPYEFFICIENCY  Fit the single free coefficient of the crop model.
%
%   [ce, info] = calibrateCanopyEfficiency(drivers, observed, p)
%   [ce, info] = calibrateCanopyEfficiency(drivers, observed, p, optics, bounds)
%
% WHY ONLY ONE FREE PARAMETER
%   There are eight fresh-weight checkpoints per trial, and they are not
%   eight independent observations -- they are eight views of the same ten
%   plants, four days apart, so the effective sample size is smaller still.
%   A model with three or four free parameters will fit that beautifully and
%   predict nothing.  So exactly one coefficient moves, canopyEfficiency, and
%   everything else stays at its literature or measured value.  If the fit is
%   bad, that is a result ABOUT THE MODEL, which is more useful than a good
%   fit obtained by tuning.
%
% WHAT IS FITTED
%   Least squares in LOG fresh weight, not in grams.  Growth spans 4.7 g to
%   220 g over the cycle, so a linear-space fit is decided almost entirely by
%   the last two checkpoints and ignores whether the establishment phase was
%   right.  Log space weights proportional error equally at every checkpoint,
%   which is the right question here: the downstream use is early detection
%   of plants departing from trajectory, and that needs the early points.
%
%   DAT 0 is excluded: the model is initialised FROM it, so including it
%   would reward the fit for an identity.
%
% OUTPUTS
%   ce    fitted canopy efficiency [-]
%   info  struct with objective value, checkpoint count, and an atBound flag.
%         atBound = true means the model cannot reach the observed biomass by
%         scaling canopy assimilation alone -- read the value as a limit, not
%         an estimate, and look for the real cause.

if nargin < 4 || isempty(optics), optics = leafOptics(); end
if nargin < 5 || isempty(bounds)
    % Deliberately generous upper bound, so an implausible fit shows up as a
    % large number rather than being silently clipped at a respectable one.
    bounds = [0.2, 6.0];
end

keep = observed.DAT > 0;
dats = observed.DAT(keep);
obs  = observed.observed_median_g(keep);

objective = @(ce) logMSE(ce, drivers, p, optics, dats, obs);

opts   = optimset('TolX', 1e-4, 'Display', 'off');
[ce, fval] = fminbnd(objective, bounds(1), bounds(2), opts);

info = struct( ...
    'canopyEfficiency', ce, ...
    'objectiveLogMSE',  fval, ...
    'nCheckpoints',     numel(dats), ...
    'bounds',           bounds, ...
    'atBound',          abs(ce - bounds(1)) < 1e-3 || abs(ce - bounds(2)) < 1e-3);
end

function v = logMSE(ce, drivers, p, optics, dats, obs)
p.canopyEfficiency = ce;
sim  = simulateLettuceTrial(drivers, p, optics);
pred = max(interpAtDAT(sim, dats), 1e-6);
v    = mean((log(pred) - log(obs)).^2);
end

function y = interpAtDAT(sim, dats)
%INTERPATDAT  Model fresh weight on the observation days (exact row lookup).
y = nan(numel(dats),1);
for k = 1:numel(dats)
    idx = find(sim.DAT == dats(k), 1);
    if isempty(idx)
        error('calibrateCanopyEfficiency:missingDay', ...
              'Driver table has no DAT = %d.', dats(k));
    end
    y(k) = sim.freshWeightG(idx);
end
end
