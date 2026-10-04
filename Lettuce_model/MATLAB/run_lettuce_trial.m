%RUN_LETTUCE_TRIAL  Run the lettuce crop model over the two grow-tent trials.
%
% Edit the OPTIONS block below, then press Run.  Outputs land in .\outputs\:
%     trialN_simulation.csv     full daily state and fluxes
%     trialN_fit.csv            checkpoint comparison, including the residual
%     growth_trajectories.png   model vs observed vs steering band
%     carbon_balance.png        the fluxes behind the trajectory
%
% PREREQUISITE
%   The daily driver tables in ..\data\ are written by ..\Python\
%   prepare_drivers.py from the raw Mycodo exports.  Run that once:
%       cd ..\Python && python prepare_drivers.py
%   Parsing the raw exports in one place only means the MATLAB and Python
%   models are driven by identical numbers, so any difference between their
%   outputs is a difference in the MODEL rather than in the data prep.
%
% WHAT THE OUTPUT IS FOR
%   The model sees only light, temperature, CO2 and humidity.  It has no
%   nutrient module.  Whatever of the observed growth it does NOT reproduce
%   -- the residual column in trialN_fit.csv -- is the part attributable to
%   EC steering, cultivar variation, or anything else outside the model.
%   That residual is precisely what the canopy-image analysis in this project
%   is being asked to explain, so it is the deliverable, not a nuisance.

clear; close all; clc

%% ===================== OPTIONS =====================================
TRIALS            = [1 2];        % which trials to run
CALIBRATE         = true;         % fit canopyEfficiency to observed biomass
CANOPY_EFFICIENCY = [];           % [] = use CALIBRATE; or force a value
TEMP_BASIS        = 'daily';      % 'daily' or 'photoperiod'
EC_RESPONSE       = false;        % enable the empirical EC modifier
SPECTRUM_COLUMN   = 9;            % 9 = broad white LED panel
GROWING_AREA_M2   = [];           % [] = default 1.49; override if measured
DRY_MATTER_FRAC   = [];           % [] = default 0.045
MAKE_PLOTS        = true;
%% ====================================================================

here   = fileparts(mfilename('fullpath'));
outDir = fullfile(here, 'outputs');
if ~isfolder(outDir), mkdir(outDir); end

optics = leafOptics(SPECTRUM_COLUMN);
fprintf('Leaf optics: column %d, peak %.0f nm, fAbs = %.3f, fIpar = %.3f\n\n', ...
        optics.spectrumColumn, optics.peakNm, optics.fAbs, optics.fIpar);

results = struct([]);

for k = 1:numel(TRIALS)
    trial = TRIALS(k);
    meta  = loadTrialData('meta', trial);

    fprintf('%s\n%s  (%s -> %s)\n%s\n', repmat('=',1,72), meta.label, ...
            meta.transplant, meta.harvest, repmat('=',1,72));
    fprintf('  EC strategy : %s\n', meta.ecStrategy);
    fprintf('  Light       : %s\n', meta.light);

    drivers  = loadTrialData('drivers',  trial);
    observed = loadTrialData('observed', trial);

    args = {'tempBasis', TEMP_BASIS, 'ecResponse', EC_RESPONSE};
    if ~isempty(GROWING_AREA_M2)
        args = [args, {'growingAreaM2', GROWING_AREA_M2}];  %#ok<AGROW>
    end
    if ~isempty(DRY_MATTER_FRAC)
        args = [args, {'dryMatterFraction', DRY_MATTER_FRAC}]; %#ok<AGROW>
    end
    p = lettuceParams(args{:});

    fprintf('\n  Planting density : %.1f plants/m2 (%d plants on %.2f m2)\n', ...
            p.plantingDensity, p.nPlants, p.growingAreaM2);
    fprintf('  Dry matter frac  : %.3f\n', p.dryMatterFraction);
    fprintf('  Photosynthesis T : %s mean\n', p.tempBasis);

    % ---- calibration ------------------------------------------------
    if ~isempty(CANOPY_EFFICIENCY)
        p.canopyEfficiency = CANOPY_EFFICIENCY;
        calSource = 'set in the OPTIONS block';
        calInfo   = struct('canopyEfficiency', CANOPY_EFFICIENCY, 'atBound', false);
    elseif ~CALIBRATE
        calSource = 'uncalibrated default (1.0)';
        calInfo   = struct('canopyEfficiency', p.canopyEfficiency, 'atBound', false);
    else
        [ce, calInfo] = calibrateCanopyEfficiency(drivers, observed, p, optics);
        p.canopyEfficiency = ce;
        calSource = 'least squares in log fresh weight';
        if calInfo.atBound
            warning(['The fit sat on a search bound. The model cannot reach ' ...
                     'the observed biomass by scaling canopy assimilation ' ...
                     'alone -- treat the value as a limit, not an estimate.']);
        end
    end
    fprintf('  Canopy efficiency: %.3f (%s)\n', p.canopyEfficiency, calSource);

    % ---- simulate -----------------------------------------------------
    sim    = simulateLettuceTrial(drivers, p, optics);
    report = fitReport(sim, observed);
    m      = fitMetrics(report);

    fprintf('\n  Model vs observed median fresh weight [g/plant]:\n');
    disp(report);
    fprintf('  MAE %.1f g | RMSE %.1f g | MAPE %.1f%% | R2 %.3f | bias %+.1f g\n', ...
            m.MAE_g, m.RMSE_g, m.MAPE_pct, m.R2, m.bias_g);
    fprintf('  Harvest (DAT 28): observed %.1f g, modelled %.1f g\n', ...
            m.finalObserved_g, m.finalModelled_g);

    if any(sim.bufferClamped)
        fprintf(['  NOTE: assimilate buffer hit zero on %d day(s) -- the ' ...
                 'crop was carbon-limited then.\n'], sum(sim.bufferClamped));
    end

    writetable(sim,    fullfile(outDir, sprintf('trial%d_simulation.csv', trial)));
    writetable(report, fullfile(outDir, sprintf('trial%d_fit.csv', trial)));

    r = struct('trial', trial, 'meta', meta, 'p', p, 'sim', sim, ...
               'report', report, 'metrics', m, 'observed', observed, ...
               'drivers', drivers, 'calibration', calInfo);
    if isempty(results), results = r; else, results(end+1) = r; end %#ok<SAGROW>
    fprintf('\n');
end

%% ---- cross-trial comparison ------------------------------------------
if numel(results) == 2
    fprintf('%s\nCROSS-TRIAL COMPARISON\n%s\n', repmat('=',1,72), repmat('=',1,72));
    fprintf('  Observed harvest difference : %+.1f g (adaptive minus fixed)\n', ...
        results(1).metrics.finalObserved_g - results(2).metrics.finalObserved_g);
    fprintf('  Modelled harvest difference : %+.1f g\n', ...
        results(1).metrics.finalModelled_g - results(2).metrics.finalModelled_g);
    fprintf(['\n  The model sees only light, temperature, CO2 and humidity.\n' ...
             '  Whatever of the observed difference it does NOT reproduce is\n' ...
             '  the part attributable to EC steering, cultivar variation, or\n' ...
             '  anything else outside the model -- and that residual is what\n' ...
             '  the canopy-image analysis is being asked to explain.\n\n']);
end

%% ---- figures ----------------------------------------------------------
if MAKE_PLOTS
    band = loadTrialData('band');

    % Figure 1: growth trajectories
    f1 = figure('Position', [80 80 560*numel(results) 460], 'Color', 'w');
    for k = 1:numel(results)
        r = results(k);
        subplot(1, numel(results), k); hold on; grid on
        fill([band.DAT; flipud(band.DAT)], ...
             [band.target_low_g; flipud(band.target_high_g)], ...
             [0.85 0.85 0.85], 'EdgeColor', 'none', ...
             'DisplayName', 'steering target band');
        plot(band.DAT, band.target_mid_g, '--', 'Color', [0.45 0.45 0.45], ...
             'LineWidth', 1.2, 'DisplayName', 'target mid-line (expolinear)');
        t = 0:28;
        plot(t, 227*exp(-exp(3.2 - 0.20*t)), ':', 'Color', [0.55 0.55 0.55], ...
             'LineWidth', 1.1, 'DisplayName', 'Gompertz as written in Ch.3');
        plot(r.sim.DAT, r.sim.freshWeightG, '-', 'Color', [0.12 0.47 0.71], ...
             'LineWidth', 2, 'DisplayName', 'crop model');
        plot(r.observed.DAT, r.observed.observed_median_g, 'o', ...
             'MarkerFaceColor', [0.84 0.15 0.16], 'MarkerEdgeColor', 'none', ...
             'MarkerSize', 7, 'DisplayName', 'observed median');
        if r.trial == 1
            xline(12, '-.', 'PPFD 200 \rightarrow 445', 'Color', [1 0.5 0.05], ...
                  'LabelOrientation', 'horizontal', 'HandleVisibility', 'off');
        end
        title(sprintf('%s\ncanopy efficiency = %.2f, MAE = %.1f g', ...
              r.meta.label, r.p.canopyEfficiency, r.metrics.MAE_g), ...
              'FontSize', 9, 'Interpreter', 'none');
        xlabel('Days after transplant'); ylabel('Fresh weight [g plant^{-1}]');
        xticks(band.DAT); xlim([0 28]); ylim([0 260]);
        if k == 1, legend('Location', 'northwest', 'FontSize', 8); end
    end
    exportgraphics(f1, fullfile(outDir, 'growth_trajectories.png'), ...
                   'Resolution', 160);

    % Figure 2: what drove it
    f2 = figure('Position', [80 80 1000 700], 'Color', 'w');
    panels = {'DLI supplied', 'Canopy light interception', ...
              'Carbon balance', 'Nutrient solution EC (measured)'};
    for k = 1:numel(results)
        r = results(k);
        subplot(2,2,1); hold on; grid on
        plot(r.drivers.DAT, r.drivers.DLI_mol_m2_d, 'DisplayName', r.meta.short);
        subplot(2,2,2); hold on; grid on
        plot(r.sim.DAT, r.sim.fLight, 'DisplayName', r.meta.short);
        subplot(2,2,3); hold on; grid on
        plot(r.sim.DAT, r.sim.PgrossGCH2OM2D, 'DisplayName', [r.meta.short ' gross P']);
        plot(r.sim.DAT, r.sim.RmaintGCH2OM2D, '--', 'DisplayName', [r.meta.short ' maint R']);
        subplot(2,2,4); hold on; grid on
        if ismember('EC_mS_cm', r.drivers.Properties.VariableNames)
            plot(r.drivers.DAT, r.drivers.EC_mS_cm, 'DisplayName', r.meta.short);
        end
    end
    ylabels = {'DLI [mol m^{-2} d^{-1}]', 'intercepted fraction [-]', ...
               'g CH_2O m^{-2} d^{-1}', 'EC [mS cm^{-1}]'};
    for k = 1:4
        subplot(2,2,k); title(panels{k}); xlabel('Days after transplant');
        ylabel(ylabels{k}); legend('Location', 'best', 'FontSize', 8);
    end
    exportgraphics(f2, fullfile(outDir, 'carbon_balance.png'), 'Resolution', 160);

    fprintf('Wrote figures to %s\n', outDir);
end


%% ======================= local functions ==============================
function report = fitReport(sim, observed)
%FITREPORT  Checkpoint-by-checkpoint comparison of model against observation.
%
% The residual_g column is the quantity this whole project is built around:
% what a mechanistic model driven by light, temperature, CO2 and humidity
% CANNOT explain, and therefore what canopy image features and EC history are
% being asked to predict.
n = height(observed);
DAT = observed.DAT;
observed_g = observed.observed_median_g;
modelled_g = nan(n,1);
for i = 1:n
    idx = find(sim.DAT == DAT(i), 1);
    if ~isempty(idx), modelled_g(i) = sim.freshWeightG(idx); end
end
residual_g   = observed_g - modelled_g;
residual_pct = 100 * residual_g ./ max(observed_g, 1e-9);
target_mid_g = observed.target_mid_g;
report = table(DAT, observed_g, round(modelled_g,1), round(residual_g,1), ...
               round(residual_pct,1), target_mid_g, ...
    'VariableNames', {'DAT','observed_g','modelled_g','residual_g', ...
                      'residual_pct','target_mid_g'});
end

function m = fitMetrics(report)
%FITMETRICS  Summary error statistics, excluding the DAT 0 identity.
r     = report(report.DAT > 0, :);
resid = r.residual_g;
obs   = r.observed_g;
pred  = r.modelled_g;
ssRes = sum(resid.^2);
ssTot = sum((obs - mean(obs)).^2);
m = struct( ...
    'n',               height(r), ...
    'MAE_g',           mean(abs(resid)), ...
    'RMSE_g',          sqrt(mean(resid.^2)), ...
    'MAPE_pct',        mean(abs(resid ./ max(obs,1e-9))) * 100, ...
    'R2',              1 - ssRes/ssTot, ...
    'bias_g',          mean(resid), ...
    'finalObserved_g', obs(end), ...
    'finalModelled_g', pred(end));
end
