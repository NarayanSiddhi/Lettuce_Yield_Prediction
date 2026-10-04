function optics = leafOptics(spectrumColumn, dataDir)
%LEAFOPTICS  Spectral weighting factors: incident PPFD -> absorbed photons.
%
%   optics = leafOptics()      broad white LED (relative_pfd column 9)
%   optics = leafOptics(3)     a narrow-band red/blue mix
%
% WHY THIS STEP EXISTS
%   A quantum sensor reports PPFD -- photons per second per square metre in
%   the 400-700 nm band, counted equally regardless of wavelength.  A leaf
%   does not treat them equally.  It reflects and transmits a wavelength-
%   dependent share (green photons escape far more readily than red), and of
%   the photons it does absorb, the fraction that drives photosystem electron
%   transport also varies with wavelength (the McCree curve: red ~1.0, blue
%   ~0.7, far-red falling away).  Two fixtures at identical PPFD can drive
%   noticeably different assimilation.
%
%   This function collapses that into two scalars, computed once:
%       fAbs    absorbed photons per incident PPFD photon        [-]
%       fIpar   driving electrons per incident PPFD photon       [-]
%
% WHAT CHANGED FROM THE LEGACY parabs_batch.m
%   The legacy pipeline treats a column of relative_pfd.txt as a spectral
%   IRRADIANCE in W m-2 nm-1 and converts it to photons with E = hc/lambda.
%   But that file is a relative PHOTON flux distribution, as its name says --
%   its columns integrate to ~1 in photon units.  Passing photons through a
%   W-to-photon conversion inflates the result by a factor of ~4.2, so the
%   legacy code reports an absorbed PPFD roughly three times the incident
%   PPFD, which is not physically possible.
%
%   Here the spectrum is normalised in PHOTON units over the PPFD band, so
%   fAbs and fIpar are bounded by 1 by construction.
%
% DATA FILES (..\data\)
%   relative_pfd.txt      516 x 9.  Col 1 = wavelength [nm], cols 2-9 =
%                         relative photon flux distributions for eight
%                         fixtures.  Column 9 (default) is the broad,
%                         red-weighted white spectrum; 2-8 are narrow-band
%                         red/blue LED mixes.
%   lettuce_spectrum.txt  17 x 3.  Wavelength [nm], relative quantum yield,
%                         leaf absorptance.
%
% OUTPUT
%   optics.fAbs, .fIpar, .spectrumColumn, .peakNm

if nargin < 1 || isempty(spectrumColumn), spectrumColumn = 9; end
if nargin < 2 || isempty(dataDir)
    dataDir = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'data');
end

ri   = load(fullfile(dataDir, 'relative_pfd.txt'));      % [nm, spectra...]
leaf = load(fullfile(dataDir, 'lettuce_spectrum.txt'));  % [nm, QY, absorptance]

% Integration grid.  Absorption is integrated over 350-750 nm because leaves
% do absorb outside the PAR band, but the spectrum is NORMALISED over
% 400-700 nm, because that is the band a quantum sensor integrates and hence
% the band the reported PPFD refers to.
L        = (350:750)';
ppfdBand = L >= 400 & L <= 700;

phi = interp1(ri(:,1),   ri(:,spectrumColumn), L, 'linear', 0);  % rel. photon flux
qy  = interp1(leaf(:,1), leaf(:,2),            L, 'linear', 0);  % rel. quantum yield
ab  = interp1(leaf(:,1), leaf(:,3),            L, 'linear', 0);  % absorptance

% Normalise so the spectrum carries exactly one photon inside 400-700 nm.
% After this, integrating phi .* <anything> over the full grid gives that
% quantity "per unit reported PPFD".
ppfdNorm = trapz(L(ppfdBand), phi(ppfdBand));
if ppfdNorm <= 0
    error('leafOptics:emptySpectrum', ...
          'Spectrum column %d carries no photons in 400-700 nm.', spectrumColumn);
end
phi = phi / ppfdNorm;

optics.fAbs           = trapz(L, phi .* ab);
optics.fIpar          = trapz(L, phi .* ab .* qy);
optics.spectrumColumn = spectrumColumn;
[~, iPeak]            = max(phi);
optics.peakNm         = L(iPeak);
end
