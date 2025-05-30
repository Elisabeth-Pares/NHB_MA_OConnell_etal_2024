%%=====================================================================================================================================
% “Regressing Away Common Neural Choice Signals does not make them Artifacts. Comment on Frӧmer et al (2024 , Nature Human Behaviour)”
% Authors: Redmond G. O’Connell, Elaine A. Corbett, Elisabeth Parés-Pujolràs, Daniel Feuerriegel & Simon P. Kelly
%%=====================================================================================================================================
% Simulation of CPP-like signal based on a more realistic drift diffusion process 
%(Supplementary Fig. 1)

clear all

N =5000; % number of trials

% parameters:
% Urgency:
Z = 0.3; % starting level at evidence onset 
U = 1; % slope (buildup rate) of dynamic urgency at the motor level (prop. to bound /sec)
Sz = 0.4; % start point variability at the motor level applied independently to both sides - full width of uniform distribution
Su = 0.17 % between-trial var in urgency rate

d = [1 2]; % drift rate 
s = 0.5; % gaussian noise /sec at accumulator level
evonT = 0.2; % time at which sensory evidence starts impacting on the sensory accumulator
%accT = 0.2; %  Accumulation onset time. Let's say same as evidence encoding onset.
mT = 0.1; %  motor time
st = 0.1; %  motor time variability
Terz = 0.1; % variability applied to sensory encoding delays

% Write label for figure according to variability settings 
if st > Terz
    ratioLab = 'Stim Var < Motor Var'; 
elseif st < Terz
    ratioLab = 'Stim Var > Motor Var';
elseif st == Terz
    ratioLab = 'Stim Var = Motor Var';
end

postAccDurM = 0.1; % mean duration of post decision accumulation
postAccDurS = 0.2; % range of uniform dist of post-decision accumulation duration

CPPrampdownDur = 0.1; % in Sec; CPP will linearly ramp back down after the accumulation stops

ITIs = [1.5 2 2.5]; % in sec. Maybe make it the same as the time from stimulus offset to next evidence onset in natalie's data? These are probably variable

bound = 1; % boundary height - same for both effectors
dt = 0.002; % time step
t = [0:dt:1.6]; % time base - let's make it a typical stimulus duration
    
% ERP filter - lowpass - to make sure it is smoothed in same way as real data (later when making full block of data)
fs=500;
LPFcutoff= 8.05;       % Low Pass Filter cutoff in Hz
% Get FIR filter weights B for Hamming-windowed sinc filter:
wc = LPFcutoff/fs*2*pi;
LPK=[];
L=103;
for i=1:L
    n = i-ceil(L/2);
    if n==0, LPK(i)=wc/pi;
    else, LPK(i) = sin(wc*n)/(pi*n); end
end
LPK = LPK.*hamming(L)';
fftlen = round(fs/17*5);

% initialise behav and waveforms we're saving on each single trial
rt = zeros(N,1);
ch = zeros(N,1);
cpp1 = zeros(N,length(t));

% Make a vector of trigger values - let's say 1 and 2 for low and high coherence
trialdriftCond = []; for c=1:length(d), trialdriftCond = [trialdriftCond c*ones(1,round(N/length(d)))]; end
trialdriftCond = trialdriftCond(randperm(N)); % shuffle trial order

for n=1:N
    
    %%%% next three lines determine when the accumulation process starts on
    %%%% each trial - original model allowed for accumulation to start
    %%%% before the evidence is encoded in the brain but here we make the
    %%%% simplifying assumption that they onset together
    evidence = zeros(1,length(t));
    evidence(t>=(evonT+(rand-.5)*Terz)) = 1;
    startAccT =     t(find(evidence,1,'first'));
    
    % Make linear urgency functions and add start point variability
    u1 = Z + (U+randn*Su)*t; u1 = u1 + (rand-.5)*Sz;
    u2 = Z + (U+randn*Su)*t; u2 = u2 + (rand-.5)*Sz; 

    dtt = d(trialdriftCond(n)); % drift rate for this trial

    noise = [zeros(1,length(find(t<startAccT))) randn(1,length(find(t>=startAccT)))*s*sqrt(dt)];
    ev=evidence;
    ev(find(t<startAccT))=0; % set to zero all of the evidence before accT
    cumdifev = cumsum(ev*dtt*dt + noise); % This is the cumulative differential evidence, just as in a 1d DDM. 

    % Now combine the urgency with cumulative evidence for two racing Decision Variables
    DV1 = u1 + max([zeros(size(cumdifev));cumdifev]); % Add the half-wave-rectified Cumulative differetial evidence
    DV2 = u2 + max([zeros(size(cumdifev));-cumdifev]);

    % terminate the decision process on bound crossing, record threshold-crossing samplepoint:
    t1i = find(DV1>bound,1); % finding the sample point of threshold crossing of each, then will pick the earlier as the winner
    t2i = find(DV2>bound,1);
    if isempty(t1i) & isempty(t2i), rt(n,1) = nan; ch(n,1)=2; % miss
    elseif ~isempty(t1i) & isempty(t2i), dti=t1i; ch(n,1)=1; 
    elseif isempty(t1i) & ~isempty(t2i), dti=t2i; ch(n,1)=0;
    else, if t1i<=t2i, ch(n,1)=1; else, ch(n,1)=0; end % CHoice: 1=correct, 0=error
        dti = min(t1i,t2i); end % calling it decision time index dti

    % now record rt in sec after adding motor time, with variability
    rt(n,1) = t(dti)+mT+(rand-.5)*st;  
    
    % now make the CPP peak and go down linearly after a certain amount of post-dec accum time for this trial:
    pdaccDurtt = postAccDurM + postAccDurM*randn; % this is in Sec. Sorry I'm flitting between sec to sample points!
    accStopi = dti + round(pdaccDurtt/dt);
    while accStopi < 0%EPP edited to avoid neg index errors
            pdaccDurtt = postAccDurM + postAccDurM*randn; % this is in Sec. Sorry I'm flitting between sec to sample points!
            accStopi = dti + round(pdaccDurtt/dt);
    end
    cumdifev = abs(cumdifev);

    if accStopi<length(t)
        cumdifev(accStopi+1:end) = max(0, cumdifev(accStopi)-[1:(length(t)-accStopi)]*cumdifev(accStopi)*(dt/CPPrampdownDur));
    end
    
    % Save the decision signal waveforms for this trial:
    cpp1(n,:) = cumdifev; % CPP on the scalp is assumed to reflect the absolute value of the cumulative differential evidence
 
end

% good accuracies?
for c=1:length(d)
    acc(c) = length(find(trialdriftCond==c & ch'==1))/length(find(trialdriftCond==c & ch'<2));
end

% grand
Trig_times = [];
% now let's make a continuous 'EEG' trace from this
scalpCPP = []; stimTi=[]; respTi=[]; % we're going to record the sample point at which a stimulus onset, and at which a response was made
k=0;
type = []; latency = []; EEG = []; 
for n=1:N
    scalpCPP = [scalpCPP zeros(1,round(ITIs(randi(length(ITIs)))/dt))];
    stimTi=[stimTi length(scalpCPP)];
        %%%% put triggers into .set format
    k=k+1;
    latency(1,k)=length(scalpCPP);
    type{1,k} = char('stim_on');
    
    if ch(n)<2, respTi=[respTi stimTi(end)+round(rt(n)/dt)]; 
        k=k+1; 
        latency(1,k) = [stimTi(end)+round(rt(n)/dt)];
        type{1,k} =  char('resp_on');
       
    end
    scalpCPP = [scalpCPP cpp1(n,:)]; 
end

EEGnoise = 1; % how big to make this? We're still in CPP units. This is pretty big.
scalpCPP = scalpCPP+randn(1,length(scalpCPP))*EEGnoise;
% filter like the real ERPs:
scalpCPP = conv(scalpCPP,LPK,'same');
EEG.data = double(scalpCPP);

% adding trigger info to .set format
Temp = table(char(type'), latency', 'VariableNames',{'type','latency'});
EEG.event = table2struct(Temp);
EEG.srate = fs;
EEG.pnts = length(EEG.data);  %%%% NOT SURE IF THIS IS RIGHT
EEG.nbchan = 1;

%Loop through trials to assign rt values
k=0;
for n=1:N
   %Stim
   k = k+1;
   [EEG.event(k).is_slow] = deal(rt(n)>=median(rt)); % add an indicator variable for rt median split, as in Fromer et al
   [EEG.event(k).condition] = deal(trialdriftCond(n)); % add an indicator variable for rt median split, as in Fromer et al

   k = k+1;
   [EEG.event(k).is_slow] = deal(rt(n)>=median(rt)); % add an indicator variable for rt median split, as in Fromer et al
   [EEG.event(k).condition] = deal(trialdriftCond(n)); % add an indicator variable for rt median split, as in Fromer et al
end
figure; hist(rt,[-.2:.01:2]); xlim([-0.2,2])

%% Sanity-check: let's derive regular ERPs, pretending all we have are triggers
% Time base for stimulus-locked waveforms:
ts = [-500:499]; % in sample points
tt = ts*dt;      % in sec

% Time base for response-locked waveforms:
trs = [-500:499]; % in sample points
tr = trs*dt;      % in sec
ylims = [-0.05 1]; % ylimits for plotting waveforms
clear erp erpr avERP avERPr
check=0;

for n=1:length(stimTi)
    erp(n,:) = scalpCPP(stimTi(n)+ts);
    nextRespi = respTi(find(respTi>stimTi(n),1));
    if nextRespi-stimTi(n)<2/dt % legit rt if less than 2 sec
        check = check+1;
        if nextRespi+trs(end) > length(scalpCPP)
            erpr(n,:) = NaN;
        else
        erpr(n,:) = scalpCPP(nextRespi+trs);
        end
    else
        erpr(n,:) = nan(1,length(trs));
    end
end
for c=1:length(d)
    avERP(c,:) = nanmean(erp(find(trialdriftCond==c),:));
    avERPr(c,:) = nanmean(erpr(find(trialdriftCond==c),:));
    
end

avERP_S(1,:) = nanmean(erp(rt>=median(rt),:));
avERP_S(2,:) = nanmean(erp(rt<median(rt),:));
avERP_R(1,:) = nanmean(erpr(rt>=median(rt),:));
avERP_R(2,:) = nanmean(erpr(rt<median(rt),:));

%% RUN UNFOLD TOOLBOX
cfg = [];
%Reproducing analysis in Fromer et al. Fig. S6
cfg.formula = {'y ~ 1+cat(is_slow)', 'y ~ 1+cat(is_slow)'}; 
cfg.eventtypes = {'stim_on','resp_on'};
cfg.timelimits = [-1 1]; 
cfg.channel = 1;

plt.times = []; 
plt.timelimits = [-1 0.999]; 
plt.times = [-1:1/EEG.srate:0.999]; 

EEG = uf_designmat(EEG,cfg);
EEG = uf_timeexpandDesignmat(EEG,cfg);
EEG = uf_glmfit(EEG,'channel',1);
ufresult = uf_condense(EEG);

%Only allow to vary with response, reproducing main analysis
%al. 
cfg.formula = {'y ~ 1', 'y ~ 1+cat(is_slow)'}; 
EEG = uf_designmat(EEG,cfg);
EEG = uf_timeexpandDesignmat(EEG,cfg);
EEG = uf_glmfit(EEG,'channel',1);
ufresult1 = uf_condense(EEG);

%For residuals analysis, RT agnostic regression
cfg.formula = {'y ~ 1', 'y ~ 1'};  
EEG = uf_designmat(EEG,cfg);
EEG = uf_timeexpandDesignmat(EEG,cfg);
EEG = uf_glmfit(EEG,'channel',1);
ufresult2 = uf_condense(EEG);

[orig,corr_S,corr_R,corrS_S, corr_RES] = deal(EEG.data);
ts = round(plt.times*EEG.srate);

%%
clear erprcRAMPS_R; clear erprcRAMP_R; clear erprcRAMP_S;
for n=1:N
    
    % For corrected resp ERPs, remove S-component by subtracting from each trial:
    if EEG.event(n*2).latency+ts(end) > length(corrS_S)
        erprcRAMPS_R(n,:) = NaN;
    else
        corrS_S(EEG.event(n*2-1).latency+ts) = corrS_S(EEG.event(n*2-1).latency+ts) - ufresult1.beta(:,:,1);
        erprcRAMPS_R(n,:) = corrS_S(EEG.event(n*2).latency+ts); %Removed stimulus
    end
    
    % For corrected resp ERPs, remove S-component by subtracting from each trial:
    if EEG.event(n*2).latency+ts(end) > length(corr_S)
        erprcRAMP_R(n,:) = NaN;
    else
        corr_S(EEG.event(n*2-1).latency+ts) = corr_S(EEG.event(n*2-1).latency+ts) - ufresult.beta(:,:,1);
        if EEG.event(n*2-1).is_slow == 1 %For slow trials, remove "slow" betas too
            corr_S(EEG.event(n*2-1).latency+ts) = corr_S(EEG.event(n*2-1).latency+ts) - ufresult.beta(:,:,2);
        end
        erprcRAMP_R(n,:) = corr_S(EEG.event(n*2).latency+ts); %Removed stimulus
    end
    
    % For corrected stimulus ERPs, remove R-component by subtracting from each trial:
    if EEG.event(n*2).latency+ts(end) > length(corr_R)
        erprcRAMP_S(n,:) = NaN;
    else
        corr_R(EEG.event(n*2).latency+ts) = corr_R(EEG.event(n*2).latency+ts) - ufresult.beta(:,:,3);
        if EEG.event(n*2).is_slow == 1  %For slow trials, remove "slow" betas too
            corr_R(EEG.event(n*2).latency+ts) = corr_R(EEG.event(n*2).latency+ts) - ufresult.beta(:,:,4);
        end
        erprcRAMP_S(n,:) = corr_R(EEG.event(n*2-1).latency+ts); %Removed resp
    end
    
    %For residuals analysis, remove both S and R activity from simulated
    %ramp
    if EEG.event(n*2).latency+ts(end) > length(corr_RES)
        RAMP_RES_S(n,:) = NaN;
        RAMP_RES_R(n,:)= NaN;
    else
        corr_RES(EEG.event(n*2).latency+ts) = corr_RES(EEG.event(n*2).latency+ts) - ufresult2.beta(:,:,2);
        corr_RES(EEG.event(n*2-1).latency+ts) = corr_RES(EEG.event(n*2-1).latency+ts) - ufresult2.beta(:,:,1);
        RAMP_RES_S(n,:) = corr_RES(EEG.event(n*2-1).latency+ts); 
        RAMP_RES_R(n,:) = corr_RES(EEG.event(n*2).latency+ts); 
    end
    
end

fastIdx = find(rt < median(rt));
slowIdx = find(rt >= median(rt));
%%
colors = [180 108 110 ; 105 9 11]/255;
xlS = [-0.5 1]; % x axis limits for Stimulus-locked
xlR = [-0.75 0.5];  % x axis limits for Response-locked

FS = 11; % fontsize
xlex = [-0.1 1.1];
yl=[-0.1 0.6];

f = figure; t= tiledlayout(3,2);
set(0,'DefaultLegendAutoUpdate','off')
set(gcf,'DefaultLineLineWidth',2);

nexttile; plot(tt,avERP_S(2,:,:)','-','Color',colors(1,:)); hold on; ylim(ylims); xlim([-0.1 1]); ylabel('Raw ERP'); 
plot(tr,avERP_S(1,:,:)','-','Color',colors(2,:)); hold on; ylim(ylims); xlim([-0.5 0.2]);  
title('Stimulus')
l = legend('Fast RT','Slow RT','location', 'ne','box', 'off', 'interpreter', 'latex');
l.ItemTokenSize = [10 8];
xticklabels('');

xlim(xlS); ylim(yl); 
plot([0 0],yl,'-k','LineWidth',1);    hold on; % eSated S-compt

nexttile; plot(tr,avERP_R(2,:,:)','-','Color',colors(1,:)); hold on; ylim(ylims); xlim([-0.5 0.2]);  legend('Hard','Easy');
plot(tr,avERP_R(1,:,:)','-','Color',colors(2,:)); hold on; ylim(ylims); xlim([-0.5 0.2]);  
l=legend('Fast RT','Slow RT','location', 'nw','box', 'off', 'interpreter', 'latex');
l.ItemTokenSize = [10 8]; xticklabels('');
title('Response')
set(gca,'YTickLabel','');
plot([0 0],yl,'-k','LineWidth',1);    hold on; 
xlim(xlR); ylim(yl); 

% Only stim 
nexttile;
plot(plt.times,ufresult1.beta(:,:,1),'-'); hold on;
plot([0 0],yl,'k','LineWidth',1);
xlim(xlS); ylim(yl); hold on;xticklabels('');
l = legend(['$$\hat{S}$$', ''], 'interpreter', 'latex', 'location', 'ne', 'box', 'off')
l.ItemTokenSize = [10 8];xticklabels('');
ylabel('Unfold ERP');xticklabels('');
text(-0.3,yl(2)*1.25,'                           S ~ 1, R ~ 1 + RT')

nexttile;
plot(plt.times,nanmean(erprcRAMPS_R(fastIdx,:)),'-','Color',colors(1,:)); hold on;
plot(plt.times,nanmean(erprcRAMPS_R(slowIdx,:)),'-','Color',colors(2,:));
plot([0 0],yl,'-k','LineWidth',1);    hold on; % estimated S-compt
xlim(xlR);
ylim(yl); hold on;
set(gca,'YTickLabel','');
title(['$$\hat{S}$$ removed', ''], 'interpreter', 'latex')
l = legend({'Fast RT', 'Slow RT'}, 'interpreter', 'latex', 'location', 'nw', 'box', 'off')
l.ItemTokenSize = [10 8];xticklabels('');

% S & R
nexttile;
plot(plt.times,nanmean(erprcRAMP_S(fastIdx,:)),'-','Color',colors(1,:)); hold on;
plot(plt.times,nanmean(erprcRAMP_S(slowIdx,:)),'-','Color',colors(2,:));
plot([0 0],yl,'k','LineWidth',1); 
xlim(xlS); ylim(yl); hold on;
plot(plt.times,nanmean(RAMP_RES_S(fastIdx,:)),':','Color',[.8 .8 .8]); hold on;
plot(plt.times,nanmean(RAMP_RES_S(fastIdx,:)),':','Color',colors(1,:)); hold on;
plot(plt.times,nanmean(RAMP_RES_S(slowIdx,:)),':','Color',colors(2,:));
title(['$$\hat{R}$$ removed', ''], 'interpreter', 'latex')
l = legend({'Fast RT', 'Slow RT', '','Residuals'}, 'interpreter', 'latex', 'location', 'ne', 'box', 'off')
l.ItemTokenSize = [10 8];
ylabel('Unfold ERP')
text(-0.3,yl(2)*1.25,'                        S ~ 1 + RT, R ~ 1 + RT')

nexttile;
plot(plt.times,nanmean(erprcRAMP_R(fastIdx,:)),'-','Color',colors(1,:)); hold on;
plot(plt.times,nanmean(erprcRAMP_R(slowIdx,:)),'-','Color',colors(2,:));

plot(plt.times,nanmean(RAMP_RES_R(fastIdx,:)),':','Color',[.8 .8 .8]); hold on;
plot(plt.times,nanmean(RAMP_RES_R(fastIdx,:)),':','Color',colors(1,:)); hold on;
plot(plt.times,nanmean(RAMP_RES_R(slowIdx,:)),':','Color',colors(2,:));

plot([0 0],yl,'-k','LineWidth',1);    hold on; 
xlim(xlR);
ylim(yl); hold on;
set(gca,'YTickLabel','');
title(['$$\hat{S}$$ removed', ''], 'interpreter', 'latex')
l = legend({'Fast RT', 'Slow RT', 'Residuals'}, 'interpreter', 'latex', 'location', 'nw', 'box', 'off')
l.ItemTokenSize = [10 8];

sgtitle(t, ratioLab)
xlabel(t, 'Time (s)')

f.Units = 'centimeters';
f.OuterPosition = [0 0 12 16.5];
exportgraphics(f, [exp.figurepath, '/FigureS1.tiff'], 'Resolution', 600);