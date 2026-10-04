function varargout = loadTrialData(what, trial, dataDir)
%LOADTRIALDATA  Read the prepared driver tables, observations and metadata.
%
%   drivers  = loadTrialData('drivers',  trial)
%   observed = loadTrialData('observed', trial)   % trial = [] for both
%   meta     = loadTrialData('meta',     trial)
%   band     = loadTrialData('band')
%
% The CSVs are written by ..\Python\prepare_drivers.py, which parses the raw
% Mycodo exports.  Keeping the raw-export parsing in exactly one place (and
% in Python, where the .xlsx readers are better) means the MATLAB and Python
% models are guaranteed to be driven by identical numbers -- so any
% difference between their outputs is a difference in the MODEL, which is the
% only thing worth comparing.
%
% If the CSVs are missing, run once from a shell:
%     cd ..\Python && python prepare_drivers.py

if nargin < 2, trial = []; end
if nargin < 3 || isempty(dataDir)
    dataDir = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'data');
end

switch lower(what)

case 'drivers'
    f = fullfile(dataDir, sprintf('trial%d_daily_drivers.csv', trial));
    assertFile(f);
    varargout{1} = readtable(f);

case 'observed'
    % Observed median net fresh weight and the steering target band.
    % Weights are NET of the 25 g substrate + net-pot correction the thesis
    % applies (Section 3.5.4), so they are directly comparable with the
    % model's shoot fresh weight.
    f = fullfile(dataDir, 'observed_fresh_weight.csv');
    assertFile(f);
    obs = readtable(f);
    if ~isempty(trial)
        obs = obs(obs.trial == trial, :);
    end
    varargout{1} = obs;

case 'band'
    % The steering target band, verbatim from the operator's
    % Lettuce_FW_EC_Tracker sheet.  Identical for both trials.
    %
    % NOTE, because it matters when citing: thesis Ch.3 describes this as a
    % Gompertz curve W = 227*exp(-exp(3.2 - 0.20t)) with a flat +/-10% band.
    % The tabulated band is neither.  Its mid-line is EXPOLINEAR -- growing
    % by a constant factor of 2.52 every 4 days to DAT 12 (RGR 0.2312 d-1),
    % then by a constant 10.68 g d-1 from DAT 16 -- and its tolerance NARROWS
    % with maturity: +/-15% for DAT 0-12, +/-10% for DAT 16-24, +/-7% at
    % harvest.  Both limbs are exact to the sheet's last digit.  The Gompertz
    % curve as written gives 83.5 g at DAT 16 against the sheet's 98.8 g, and
    % 207.3 g at DAT 28 against 227.0 g.
    DAT           = [0; 4; 8; 12; 16; 20; 24; 28];
    target_low_g  = [3.0; 7.5; 18.9; 47.6;  88.9; 127.4; 165.8; 211.1];
    target_mid_g  = [3.5; 8.8; 22.2; 56.0;  98.8; 141.5; 184.3; 227.0];
    target_high_g = [4.0; 10.1; 25.5; 64.4; 108.6; 155.7; 202.7; 242.9];
    tolerance_pct = round(100*(target_high_g - target_mid_g)./target_mid_g, 1);
    varargout{1}  = table(DAT, target_low_g, target_mid_g, target_high_g, ...
                          tolerance_pct);

case 'meta'
    switch trial
    case 1
        m.label      = 'Trial 1 - corrective adaptive EC steering';
        m.short      = 'adaptive';
        m.transplant = '2025-10-14';
        m.harvest    = '2025-11-11';
        m.ecStrategy = ['1.2-1.4 mS/cm at transplant, stepped up at 4-day ' ...
                        'checkpoints when median FW fell below the target ' ...
                        'band (+0.2 below, -0.1 after two checkpoints ' ...
                        'above); measured daily mean reached 2.25 mS/cm'];
        m.light      = ['~200 umol m-2 s-1 for DAT 0-11, raised to ~445 at ' ...
                        'DAT 12 and held there'];
        m.observedFinalG = 219.8;
    case 2
        m.label      = 'Trial 2 - preventive fixed EC';
        m.short      = 'fixed';
        m.transplant = '2026-01-19';
        m.harvest    = '2026-02-16';
        m.ecStrategy = ['held at 1.4-1.6 mS/cm for the whole cycle; growth ' ...
                        'checkpoints used for evaluation only, never for ' ...
                        'intervention'];
        m.light      = '~445 umol m-2 s-1 constant from transplant';
        m.observedFinalG = 179.8;
    otherwise
        error('loadTrialData:badTrial', 'Trial must be 1 or 2.');
    end
    varargout{1} = m;

otherwise
    error('loadTrialData:unknown', 'Unknown request "%s".', what);
end
end

function assertFile(f)
if ~isfile(f)
    error('loadTrialData:missingFile', ...
        ['%s not found.\n' ...
         'Generate the driver tables first:\n' ...
         '    cd ..\\Python && python prepare_drivers.py'], f);
end
end
