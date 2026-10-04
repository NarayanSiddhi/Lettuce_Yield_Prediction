function out = simulateLettuceDay(state, p, drv, optics)
%SIMULATELETTUCEDAY  Advance the lettuce crop one day.
%
%   out = simulateLettuceDay(state, p, drv, optics)
%
% THE MODEL IN ONE PARAGRAPH
%   Two state variables carry the crop.  Wstruct is structural dry matter --
%   cell walls, the stuff that makes a leaf -- and it determines both leaf
%   area and harvestable weight.  Wbuffer is the non-structural pool of
%   soluble sugars and starch.  Photosynthesis pays into the buffer;
%   maintenance respiration draws from it around the clock; and structural
%   growth draws from it at a rate that saturates as the buffer fills.
%   Splitting the two is what lets the model do something a single-pool model
%   cannot: keep growing for a day or two after the light drops, and stall
%   when the buffer empties even though the leaves are still there.
%
%     Wbuffer  <--- photosynthesis (light period only)
%              ---> maintenance respiration (24 h)
%              ---> structural growth + its synthesis overhead
%
%     Wstruct  ---> leaf area ---> light interception ---> photosynthesis
%              ---> harvestable fresh weight
%
%   The feedback from Wstruct back into photosynthesis through leaf area is
%   what produces the sigmoid: exponential while the canopy is open and every
%   new leaf intercepts new light, then linear once interception saturates.
%
% DAILY TIME STEP -- and why it is defensible here
%   Photosynthesis is evaluated once per day, at the mean conditions over the
%   light period, rather than integrated hour by hour.  Because the
%   light-response curve is concave, evaluating at the mean of a varying PPFD
%   overestimates the integral of a varying one (Jensen's inequality).  In
%   these trials that bias is small, because the LED panel is a step
%   function: PPFD is essentially constant for 16 h and zero for 8 h, exactly
%   the case where mean-over-photoperiod is nearly exact.  Do NOT reuse this
%   shortcut with sunlight drivers without checking it.
%
% INPUTS
%   state   struct with Wstruct, Wbuffer, GDD
%   p       parameter struct from lettuceParams
%   drv     struct of the day's drivers:
%             TairC, TairLightC, RHpct, RHlightPct, CO2ppm,
%             PPFDumolM2S, PairPa, ECmScm
%   optics  struct from leafOptics
%
% OUTPUT  struct with the new state plus daily diagnostics.  All fluxes are
%   per square metre of BENCH FLOOR unless the name says otherwise.
%
% Van Henten (1994) Agricultural Systems 45:55-72 is the source for the
% growth and respiration structure.  A copy is in ..\..\Literature\.

MW_CO2 = 44.0;   % g mol-1

Wstruct = state.Wstruct;
Wbuffer = state.Wbuffer;

% -- 1. Which temperature drives photosynthesis? ------------------------
% Assimilation only happens while the lights are on, so the photoperiod mean
% is the physical choice.  'daily' reproduces the CEAC web tool.
if strcmpi(p.tempBasis, 'photoperiod') && isfinite(drv.TairLightC)
    Tphoto  = drv.TairLightC;
    RHphoto = drv.RHlightPct;
else
    Tphoto  = drv.TairC;
    RHphoto = drv.RHpct;
end

% -- 2. Light reaching the photosystems ---------------------------------
% drv.PPFDumolM2S is what a quantum sensor at canopy height would read.  The
% optics step removes reflected and transmitted photons and weights the rest
% by their photosynthetic effectiveness.
ppfdAbs = optics.fAbs  * drv.PPFDumolM2S;
ipar    = optics.fIpar * drv.PPFDumolM2S;

% -- 3. Leaf gas exchange ------------------------------------------------
gas    = photosynthesisFvCB(ppfdAbs, ipar, Tphoto, RHphoto, drv.CO2ppm, ...
                            p, drv.PairPa);
AgLeaf = gas.Ag(1);       % gross, [umol CO2 m-2_leaf s-1]
AnLeaf = gas.An(1);
Eleaf  = gas.E(1);

% -- 4. Scale leaf to canopy, and instant to daily ----------------------
fLight = cropGeometry('lightFraction', p, Wstruct);

% Gross canopy assimilation over the light period, expressed as the CH2O it
% is worth to the buffer:
%   umol CO2 m-2_leaf s-1
%     * 44e-6 g CO2 umol-1        -> g CO2 m-2_leaf s-1
%     * fLight                    -> g CO2 m-2_floor s-1
%     * canopyEfficiency          -> big-leaf upscaling correction
%     * photoperiodS              -> g CO2 m-2_floor d-1
%     * cAlpha (= 30/44)          -> g CH2O m-2_floor d-1
Pday = max(AgLeaf, 0) * MW_CO2 * 1e-6 * fLight * p.canopyEfficiency * ...
       p.photoperiodS * p.cAlpha;

% Maintenance respiration runs 24 h on the 24-h mean temperature, charged
% against structure only -- the buffer is substrate, not machinery needing
% upkeep.
respCoeff = p.cRespSht * (1 - p.cTau) + p.cRespRt * p.cTau;
Rday = respCoeff * Wstruct * p.Q10resp^((drv.TairC - 25)/10) * 86400;

% -- 5. Structural growth -------------------------------------------------
% Van Henten's saturating rule: growth is proportional to the existing
% structure (each gram of leaf builds more leaf) and is throttled by how full
% the buffer is.
%
% CORRECTION vs. the legacy MATLAB, which writes
%     r_gr = c_gr_max * Ws/(Wg+Ws) * fT ;   dWg = r_gr * Ws
% making growth proportional to the BUFFER rather than to the structure, and
% using a plain fraction where Van Henten has a saturation function.  With
% cGamma = 1 the two denominators coincide, but the pool the rate multiplies
% does not.
fGrowthT = p.Q10growth^((drv.TairC - 20)/10);
rGr = p.cGrMax * Wbuffer / max(p.cGamma*Wstruct + Wbuffer, 1e-9) * ...
      fGrowthT * 86400;                                    % [d-1]
rGr = rGr * cropGeometry('ecFactor', p, drv.ECmScm);       % 1.0 unless enabled

dWstruct = rGr * Wstruct;

% Growth cannot outrun the buffer: laying down 1 g of structure costs
% 1/cBeta g of CH2O (the extra is synthesis respiration).
costPerG  = 1 / p.cBeta;
available = max(Wbuffer + Pday - Rday, 0);
dWstruct  = max(min(dWstruct, available / costPerG), 0);

% -- 6. Update the pools ---------------------------------------------------
WbufferNew = Wbuffer + Pday - Rday - dWstruct * costPerG;
WstructNew = Wstruct + dWstruct;

% A buffer cannot go negative.  If respiration exceeded supply the crop would
% in reality remobilise structure; over a 28-day lettuce cycle in a heated
% tent that never happens, so clamping is safe -- and any clamping event is
% reported so it cannot pass unnoticed.
bufferClamped = WbufferNew < 0;
WbufferNew    = max(WbufferNew, 0);

out.Wstruct = WstructNew;
out.Wbuffer = WbufferNew;
out.GDD     = state.GDD + max(0, drv.TairC - p.GDDbaseC);

out.freshWeightG      = cropGeometry('freshWeight', p, WstructNew, WbufferNew);
out.LAI               = cropGeometry('LAI', p, WstructNew);
out.fLight            = fLight;
out.AgLeafUmolM2S     = AgLeaf;
out.AnLeafUmolM2S     = AnLeaf;
out.PgrossGCH2OM2D    = Pday;
out.RmaintGCH2OM2D    = Rday;
out.dWstructGM2D      = dWstruct;
out.rGrPerD           = rGr;
out.transpirationMolM2S = Eleaf;
out.VPDkPa            = gas.VPD(1);
out.CiPpm             = gas.Ci(1);
out.bufferClamped     = bufferClamped;
end
