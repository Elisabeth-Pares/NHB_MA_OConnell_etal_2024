%%=====================================================================================================================================
% “Regressing Away Common Neural Choice Signals does not make them Artifacts. Comment on Frӧmer et al (2024 , Nature Human Behaviour)”
% Authors: Redmond G. O’Connell, Elaine A. Corbett, Elisabeth Parés-Pujolràs, Daniel Feuerriegel & Simon P. Kelly
%%=====================================================================================================================================

% Simulation of simple signal shapes to demonstrate Unfold works well when the data meet its assumptions, but not so much otherwise.

clear all; close all;
load EEG2x2 % to make sure structure is configured to work, I'm starting from one of the tutorial simulations
% has 200 Hz sample rate, and 120,000 samples (10 minutes) - fine for our purposes!

% We're going to replace the EEG data and the event list. Let's start by generating the RT distribution which can be
% entered as the event list in all cases.

% Let's say a stimulus comes up every
SOA = 1.8; % ...sec, and so we'll fit this many trials in 10 min:
numtr = 332; % number of trials
% generate RTs:
dm = 1.2%1.2; % mean drift rate /sec
ds = .5; %0.5 std of drift rate /sec
bcr = 0.8; % assume a collapsing bound at this rate /sec, let's say bound starts at 1 and collapses toward zero at 2 sec. Almost ruling out misses.
% Let's generate a bunch of drift rates and record them:
d = dm+ds*randn(numtr,1);
RT = 1./(d+bcr); % if you want to look at distribution:
figure; hist(RT,[-.2:.01:3])
while max(RT)>1.5 | min(RT)<0 % let's contain the RTs to within 1.5 sec (should be the case with these settings on most runs)
    d(find(RT>1.5 | RT<0)) = dm+ds*randn(length(find(RT>1.5 | RT<0)),1); % sample fresh drift rates that are hopefully less outlying
    RT(find(RT>1.5 | RT<0)) = 1./(d(find(RT>1.5 | RT<0))+bcr);
end
% Also record amplitude at resp, 'a,' of the ballistic accumulator:
a = d.*RT; % very easy equations since we're starting ramp from (0,0). To plot amplitude of DV at RT, across RT: figure; plot(RT,a,'o')

% d and a are our ground truth for the ramp model. We use the same RT distribution without loss of generality, for the simulated data with overlapping S and R signals

%% now let's make two continuous EEG signals, in both of which we insert the same trigger event for each S and R.
% in EEGSR, we insert at each event a timescale-invariant S-locked or R-locked signal, as assumed in the Unfold model.
% e.g. just a box car of different durations, or maybe a triangular wave:
durS = 0.6; durR = 0.6; % same duration and shape for S and R - don't see any need to make them different
S = [1:durS/2*EEG.srate fliplr(1:durS/2*EEG.srate-1)]./(durS/1*EEG.srate);  % The stiulus-locked signal
R = [1:durR/2*EEG.srate fliplr(1:durR/2*EEG.srate-1)]./(durR/1*EEG.srate); tR = [-(durR/2*EEG.srate-1):(durR/2*EEG.srate-1)]; % timebase to allow us to put R-locked resp centred on R
% in EEGRAMP, we interpose a ramp signal from S to R of each trial - we'll do that in the loop below.

EEG.event=[]; EEG.data=zeros(1,120000); % initialise
[EEGSR, EEGRAMP] = deal(EEG);  % blank slate for both

k=0;
for n=1:numtr
    % stimulus:
    k=k+1;
    [EEGSR.event(k).latency, EEGRAMP.event(k).latency] = deal(EEG.srate*SOA*n);  % pop the stimulus events in at every SOA
    [EEGSR.event(k).type, EEGRAMP.event(k).type] = deal('stim_on');
    % add S-locked signal to EEGSR at stimulus time:
    EEGSR.data(EEGSR.event(k).latency+[1:length(S)]) = EEGSR.data(EEGSR.event(k).latency+[1:length(S)]) + S;
    % add RAMP between S and R to EEGRAMP:
    EEGRAMP.data(EEGRAMP.event(k).latency+[1:round(EEG.srate*RT(n))]) = [1:round(EEG.srate*RT(n))]/round(EEG.srate*RT(n))*a(n); % ramps to the bound level at that RT
    [EEGSR.event(k).is_slow, EEGRAMP.event(k).is_slow] = deal(RT(n)>=median(RT)); % add an indicator variable for RT median split, as in Fromer et al
    
    % resp:
    k=k+1;
    [EEGSR.event(k).latency, EEGRAMP.event(k).latency] = deal(EEGSR.event(k-1).latency + round(EEG.srate*RT(n)));
    [EEGSR.event(k).type, EEGRAMP.event(k).type] = deal('resp');
    % add R-locked signal to EEGSR at the resp times:
    EEGSR.data(EEGSR.event(k).latency+tR) = EEGSR.data(EEGSR.event(k).latency+tR) + R;
    
    [EEGSR.event(k).is_slow, EEGRAMP.event(k).is_slow] = deal(RT(n)>=median(RT)); % add an indicator variable for RT median split, as in Fromer et al
end

cfgDesign = [];
cfgDesign.eventtypes = {'stim_on','resp'};
cfgDesign.formula = {'y~1','y~1+cat(is_slow)'};  

% For Timexpanding everything:
cfgTimeexpand = [];
cfgTimeexpand.timelimits = [-1.5 1.5]; % Fromer et al GitHub page says that analysis of Steinemann was the same as the Boldt data. The Boldt data code appears to use asymmetric S/R windows but the long part was 1.5 sec
plt.times = [-1.5:1/EEG.srate:1.499];

% now run EEGSR through unfold:
EEGSR = uf_designmat(EEGSR,cfgDesign);
EEGSR = uf_timeexpandDesignmat(EEGSR,cfgTimeexpand);
EEGSR = uf_glmfit(EEGSR,'channel',1);  % Fit the GLM
ufresultSR = uf_condense(EEGSR);

% and EEGRAMP - same code:
EEGRAMP = uf_designmat(EEGRAMP,cfgDesign);
EEGRAMP = uf_timeexpandDesignmat(EEGRAMP,cfgTimeexpand);
EEGRAMP = uf_glmfit(EEGRAMP,'channel',1);  % Fit the GLM
ufresultRAMP = uf_condense(EEGRAMP);

% ... and for looking into their Supplementary Fig 7, do another round on the ramp case, but "RT-agnostic" as they put it
cfgDesign.formula = {'y~1','y~1'};  
EEGRAMP = uf_designmat(EEGRAMP,cfgDesign);
EEGRAMP = uf_timeexpandDesignmat(EEGRAMP,cfgTimeexpand);
EEGRAMP = uf_glmfit(EEGRAMP,'channel',1);  % Fit the GLM
ufresultRAMP_RTag = uf_condense(EEGRAMP);

% Replicate Fig. S6 for ramping signal (EPP)
cfgDesign.eventtypes = {'stim_on','resp'};
cfgDesign.formula = {'y~1+cat(is_slow)','y~1+cat(is_slow)'}; 
EEGRAMP = uf_designmat(EEGRAMP,cfgDesign);
EEGRAMP = uf_timeexpandDesignmat(EEGRAMP,cfgTimeexpand);
EEGRAMP = uf_glmfit(EEGRAMP,'channel',1);  % Fit the GLM
ufresultRAMP_SR_RT = uf_condense(EEGRAMP);

% Replicate Fig. S6 for time-invariant SR signal (EPP) -
cfgDesign.eventtypes = {'stim_on','resp'};
cfgDesign.formula = {'y~1+cat(is_slow)','y~1+cat(is_slow)'};  
EEGSR = uf_designmat(EEGSR,cfgDesign);
EEGSR = uf_timeexpandDesignmat(EEGSR,cfgTimeexpand);
EEGSR = uf_glmfit(EEGSR,'channel',1);  % Fit the GLM
ufresultSR_SR_RT = uf_condense(EEGSR);


%% And now let's derive fast/slow ERP traces, both original, and "corrected" by Unfold:
% For corrected, we remove S-locked component.
% For Supl fig 7, we correct further by removing also the R compts!
[origSR, corrSR, corrSR_S,corrSR_R] = deal(EEGSR.data);
[origRAMP, corrRAMP, corrRAMP2, corrRAMP_S, corrRAMP_R] = deal(EEGRAMP.data); % corrRAMP2 will be the one corrected by removing S AND R

ts = round(plt.times*EEG.srate);
[erpSR, erprSR, erpcSR, erprcSR, erpRAMP, erprRAMP, erpcRAMP, erprcRAMP, erpcRAMP2, erprcRAMP2] = deal(nan(numtr,length(ts))); % 'r' stands for Response-locked, 'c' stands for corrected

for n=1:numtr
    % Uncorrected:
    erpSR(n,:) = origSR(EEGSR.event(n*2-1).latency+ts); % (S are the odd numbered events)
    erprSR(n,:) = origSR(EEGSR.event(n*2).latency+ts);
    erpRAMP(n,:) = origRAMP(EEGRAMP.event(n*2-1).latency+ts);
    erprRAMP(n,:) = origRAMP(EEGRAMP.event(n*2).latency+ts);
    % For corrected, remove S-locked by subtracting from each trial:
    corrSR(EEGSR.event(n*2-1).latency+ts) = corrSR(EEGSR.event(n*2-1).latency+ts) - ufresultSR.beta(:,:,1);
    erprcSR(n,:) = corrSR(EEGSR.event(n*2).latency+ts);
    corrRAMP(EEGRAMP.event(n*2-1).latency+ts) = corrRAMP(EEGRAMP.event(n*2-1).latency+ts) - ufresultRAMP.beta(:,:,1);
    erprcRAMP(n,:) = corrRAMP(EEGRAMP.event(n*2).latency+ts);
    
    % For fig S7 corrected, remove S-locked AND R-locked:
    corrRAMP2(EEGRAMP.event(n*2-1).latency+ts) = corrRAMP2(EEGRAMP.event(n*2-1).latency+ts) - ufresultRAMP_RTag.beta(:,:,1); % subtract S
    corrRAMP2(EEGRAMP.event(n*2).latency+ts) = corrRAMP2(EEGRAMP.event(n*2).latency+ts) - ufresultRAMP_RTag.beta(:,:,2);   % subtract R
    erpcRAMP2(n,:) = corrRAMP2(EEGRAMP.event(n*2-1).latency+ts);   % S-locked
    erprcRAMP2(n,:) = corrRAMP2(EEGRAMP.event(n*2).latency+ts);   % R-locked
    
    % For fig S6 corrected, remove S-locked AND R-locked in RAMP panels (RIGHT):
    % For corrected resp ERPs, remove S-component by subtracting from each trial:
    corrRAMP_S(EEGRAMP.event(n*2-1).latency+ts) = corrRAMP_S(EEGSR.event(n*2-1).latency+ts) - ufresultRAMP_SR_RT.beta(:,:,1);
    if EEGRAMP.event(n*2-1).is_slow == 1 %For slow trials, remove "slow" betas too
        corrRAMP_S(EEGRAMP.event(n*2-1).latency+ts) = corrRAMP_S(EEGSR.event(n*2-1).latency+ts) - ufresultRAMP_SR_RT.beta(:,:,2);
    end
    erprcRAMP_R(n,:) = corrRAMP_S(EEGRAMP.event(n*2).latency+ts); %Removed stimulus
    
    % For corrected stimulus ERPs, remove R-component by subtracting from each trial:
    corrRAMP_R(EEGRAMP.event(n*2).latency+ts) = corrRAMP_R(EEGRAMP.event(n*2).latency+ts) - ufresultRAMP_SR_RT.beta(:,:,3);
    if EEGRAMP.event(n*2).is_slow == 1 %For slow trials, remove "slow" betas too
        corrRAMP_R(EEGRAMP.event(n*2).latency+ts) = corrRAMP_R(EEGRAMP.event(n*2).latency+ts) - ufresultRAMP_SR_RT.beta(:,:,4);
    end
    erprcRAMP_S(n,:) = corrRAMP_R(EEGRAMP.event(n*2-1).latency+ts); %Removed resp
    
    % For fig S6 corrected, remove S-locked AND R-locked in SR panels (LEFT):
    % For corrected resp ERPs, remove S-component by subtracting from each trial:
    corrSR_S(EEGSR.event(n*2-1).latency+ts) = corrSR_S(EEGSR.event(n*2-1).latency+ts) - ufresultSR_SR_RT.beta(:,:,1);
    if EEGSR.event(n*2-1).is_slow == 1 %For slow trials, remove "slow" betas too
        corrSR_S(EEGSR.event(n*2-1).latency+ts) = corrSR_S(EEGSR.event(n*2-1).latency+ts) - ufresultSR_SR_RT.beta(:,:,2);
    end
    erprcSR_R(n,:) = corrSR_S(EEGSR.event(n*2).latency+ts); %Removed stimulus
    
    % For corrected stimulus ERPs, remove R-component by subtracting from each trial:
    corrSR_R(EEGSR.event(n*2).latency+ts) = corrSR_R(EEGSR.event(n*2).latency+ts) - ufresultSR_SR_RT.beta(:,:,3);
    if EEGSR.event(n*2).is_slow == 1 %For slow trials, remove "slow" betas too
        corrSR_R(EEGSR.event(n*2).latency+ts) = corrSR_R(EEGSR.event(n*2).latency+ts) - ufresultSR_SR_RT.beta(:,:,4);
    end
    erprcSR_S(n,:) = corrSR_R(EEGSR.event(n*2-1).latency+ts); %Removed resp
    
end

%% plotting
% first create figure and set plot dimensions etc:
plotTimes = [-1.5:1/EEG.srate:1.5];

set(0,'DefaultLegendAutoUpdate','off')
figure; set(gcf,'DefaultLineLineWidth',2);

xlS = [-0.5 1]; % x axis limits for Stimulus-locked
xlR = [-0.72 0.1];  % x axis limits for Response-locked

FS = 9; % fontsize
xlex = [-0.1 1.1];
yl=[-0.5 1];

set(0,'DefaultLegendAutoUpdate','off')
f = figure;  set(gcf,'DefaultLineLineWidth',2);
t = tiledlayout(4,4, 'TileSpacing','compact')
colors = [180 108 110 ; 105 9 11]/255;
dblue = hex2rgb("0072BD");
lblue = hex2rgb("4DBEEE");

% Example trials up top
% find suitable trials around the 10th, 90th percentile:
q19 = quantile(RT,[.1 .9]);
[mm,tr(1)]  = min(abs(RT-q19(1))); [mm,tr(2)]  = min(abs(RT-q19(2)));
for n=1:2
    if n == 1; nexttile([1,2]); end
    tex = [-0.1:1/EEG.srate:1.1]; exS = zeros(1,length(tex)); exR = zeros(1,length(tex));
    exS(find(tex==0)+[1:length(S)]) = S; [mm,RTsamp]=min(abs(tex-RT(tr(n)))); exR(RTsamp+tR) = R;
    
    if n == 2; xlabel('Time (s)'); ylabel('Simulated signals (a.u.)');
        lt = '--'; else lt = '-';end
    
    plot(tex,exS, 'Color', [0 0.4470 0.7410]); hold on; %if n==1, set(gca,'XTickLabel',[]); end%set(gca,'XColor',[1 1 1]); end
    plot(tex,exR, 'Color', [0.8500 0.3250 0.0980]); xlim(xlex); ylim(yl); plot([0 0],yl,'k','LineWidth',1);  hold on;
    plot([1 1]*tex(RTsamp),yl,'--k','LineWidth',1); hold on;
    text(-0.02,yl(2)*1.12,'S','FontSize',FS); text(tex(RTsamp)-0.04,yl(2)*1.12,['R_' num2str(n)],'FontSize',FS);
    if n == 1; text(-0.02,yl(2)*1.4,['Time Invariant S and R components'], 'fontsize', 11);
        l=legend({'S', 'R'}, 'interpreter', 'latex', 'location', 'ne', 'box', 'off')
        l.ItemTokenSize = [10 8];
    end
    
end

for n=1:2
    if n == 1; nexttile([1,2]); end
    tex = [-0.1:1/EEG.srate:1.1]; ex = zeros(1,length(tex));
    ex(find(tex==0)+[1:round(EEG.srate*RT(tr(n)))]) = [1:round(EEG.srate*RT(tr(n)))]/round(EEG.srate*RT(tr(n)))*a(tr(n));
    
    if n == 2; xlabel('Time (s)');
        lt = '--'; else lt = '-';end
    
    [mm,RTsamp]=min(abs(tex-RT(tr(n))));
    plot(tex,ex,'Color',[0 .5 0]); if n==1, set(gca,'YTickLabel', ''); end ; hold on;%set(gca,'XColor',[1 1 1]); end
    xlim(xlex); ylim(yl); plot([0 0],yl,'k','LineWidth',1); hold on;
    plot([1 1]*tex(RTsamp),yl,'--k','LineWidth',1); hold on;
    set(gca,'YTickLabel', '');
    text(tex(RTsamp)-0.04,yl(2)*1.12,['R_' num2str(n)],'FontSize',FS);
    text(-0.02,yl(2)*1.12,'S','FontSize',FS);
    if n == 1; text(-0.02,yl(2)*1.4,['Ramping Evidence Accumulation Signal'], 'fontsize', 11);
        l=legend({'EA'}, 'interpreter', 'latex', 'location', 'ne', 'box', 'off')
        l.ItemTokenSize = [10 8];
    end
end

% SR uncorrected:
nexttile; %Stimulus-locked
plot(plt.times,mean(erpSR(find(RT<median(RT)),:)),'Color',colors(1,:)); hold on;
plot(plt.times,mean(erpSR(find(RT>=median(RT)),:)),'Color',colors(2,:)); hold on;
xlim(xlS); ylim(yl); plot([0 0],yl,'k','LineWidth',1);
ylabel('ERP amplitude (a.u.)'); set(gca,'XTickLabel',[]);
text(-0.02,yl(2)*1.12,'S','FontSize',FS);
nexttile; %Response-locked
plot(plt.times,mean(erprSR(find(RT<median(RT)),:)),'Color',colors(1,:)); hold on;
plot(plt.times,mean(erprSR(find(RT>=median(RT)),:)),'Color',colors(2,:));   hold on;
xlim(xlR); ylim(yl);  plot([0 0],yl,'--k','LineWidth',1);   set(gca,'XTickLabel',[]);
text(-0.04,yl(2)*1.12,'R','FontSize',FS);
set(gca,'YTickLabel','');
l=legend({'Fast', 'Slow'}, 'interpreter', 'latex', 'location', 'nw', 'box', 'off')
l.ItemTokenSize = [10 8];

% RAMP uncorrected:
nexttile;
plot(plt.times,mean(erpRAMP(find(RT<median(RT)),:)),'Color',colors(1,:)); hold on;
plot(plt.times,mean(erpRAMP(find(RT>=median(RT)),:)),'Color',colors(2,:)); hold on; xlim(xlS);
ylim(yl); plot([0 0],yl,'k','LineWidth',1);   set(gca,'XTickLabel',[]);set(gca,'YTickLabel','');
xlim(xlS); ylim(yl); plot([0 0],yl,'k','LineWidth',1);
text(-0.02,yl(2)*1.12,'S','FontSize',FS);

nexttile;
plot(plt.times,mean(erprRAMP(find(RT<median(RT)),:)),'Color',colors(1,:)); hold on;
plot(plt.times,mean(erprRAMP(find(RT>=median(RT)),:)),'Color',colors(2,:)); hold on;
xlim(xlR); ylim(yl);
plot([0 0],yl,'--k','LineWidth',1);   set(gca,'XTickLabel',[]);
set(gca,'YTickLabel',''); set(gca,'XTickLabel',[]);
xlim(xlR); ylim(yl);  plot([0 0],yl,'--k','LineWidth',1);   set(gca,'XTickLabel',[]);
text(-0.04,yl(2)*1.12,'R','FontSize',FS);
l=legend({'Fast', 'Slow'}, 'interpreter', 'latex', 'location', 'nw', 'box', 'off')
l.ItemTokenSize = [10 8];

% SR corrected:
% yl=[-0.2 0.6];
nexttile; %Stimulus-locked
plot(plt.times,ufresultSR.beta(:,:,1)); hold on; plot([0 0],yl,'k','LineWidth',1);
xlim(xlS); ylim(yl); hold on;   % eSated S-compt
hold on;  set(gca,'XTickLabel',[]);
l = legend({'$$\hat{S}$$'}, 'interpreter', 'latex', 'location', 'ne', 'box', 'off')
l.ItemTokenSize = [10 8];
xlim(xlS); ylim(yl); plot([0 0],yl,'k','LineWidth',1);
ylabel('Unfold ERP (a.u.)')
text(-0.3,yl(2)*1.25,'                    S ~ 1, R ~ 1 + RT')


nexttile; %Response-locked
plot(plt.times,mean(erprcSR(find(RT<median(RT)),:)),'Color',colors(1,:)); hold on;
plot(plt.times,.015+mean(erprcSR(find(RT>=median(RT)),:)),'Color',colors(2,:)); xlim(xlR); ylim(yl); hold on;
plot([0 0],yl,'--k','LineWidth',1);    hold on;
set(gca,'YTickLabel','','XTickLabel',[]);
xlim(xlR); ylim(yl);  plot([0 0],yl,'--k','LineWidth',1);  % set(gca,'XTickLabel',[]);
l=legend({'$$\hat{R}$$ fast', '$$\hat{R}$$ slow'}, 'interpreter', 'latex', 'location', 'nw', 'box', 'off')

% RAMP corrected:
nexttile;
plot(plt.times,ufresultRAMP.beta(:,:,1));hold on;
xlim(xlS); ylim(yl);
plot(plt.times,mean(erpcRAMP2(find(RT<median(RT)),:)),'--','Color',colors(1,:)); hold on;
plot(plt.times,mean(erpcRAMP2(find(RT>=median(RT)),:)),'--','Color',colors(2,:)); hold on;
plot([0 0],yl,'k','LineWidth',1);   xlim(xlS); ylim(yl);  % eSated S-compt
set(gca,'YTickLabel','','XTickLabel',[]);
l = legend({'$$\hat{S}$$'}, 'interpreter', 'latex', 'location', 'ne', 'box', 'off')
l.ItemTokenSize = [10 8];
text(-0.3,yl(2)*1.25,'                    S ~ 1, R ~ 1 + RT')

nexttile;
plot(plt.times,mean(erprcRAMP(find(RT<median(RT)),:)),'Color',colors(1,:)); hold on;
plot(plt.times,mean(erprcRAMP(find(RT>=median(RT)),:)),'Color',colors(2,:)); hold on;
plot(plt.times,mean(erprcRAMP2(find(RT<median(RT)),:)),'--','Color',colors(1,:)); hold on;
plot(plt.times,mean(erprcRAMP2(find(RT>=median(RT)),:)),'--','Color',colors(2,:));
xlim(xlR); ylim(yl); plot([0 0],yl,'--k','LineWidth',1)
set(gca,'YTickLabel','','XTickLabel',[]);
l=legend({'$$\hat{R}$$ fast', '$$\hat{R}$$ slow'}, 'interpreter', 'latex', 'location', 'nw', 'box', 'off')
l.ItemTokenSize = [10 8];
xlim(xlR); ylim(yl);  plot([0 0],yl,'--k','LineWidth',1);  


% Equivalent of Fig. S6
% yl = [-0.5 0.6];

nexttile;set(gca,'YTickLabel','','XTickLabel',[]);
plot(plt.times,mean(erprcSR_S(find(RT<median(RT)),:)),'-','Color',colors(1,:)); hold on;
plot(plt.times,mean(erprcSR_S(find(RT>=median(RT)),:)),'-','Color',colors(2,:));
xlim(xlS); ylim(yl); hold on;
plot([0 0],yl,'k','LineWidth',1);
l = legend({'$$\hat{S}$$ fast', '$$\hat{S}$$ slow'}, 'interpreter', 'latex', 'location', 'ne', 'box', 'off')
l.ItemTokenSize = [10 8];
text(-0.3,yl(2)*1.25,'                S ~ 1 + RT, R ~ 1 + RT')

ylabel('Unfold ERP (a.u.)')

nexttile;set(gca,'YTickLabel','','XTickLabel',[]);
plot(plt.times,mean(erprcSR_R(find(RT<median(RT)),:)),'-','Color',colors(1,:)); hold on;
plot(plt.times,mean(erprcSR_R(find(RT>=median(RT)),:)),'-','Color',colors(2,:));
plot([0 0],yl,'--k','LineWidth',1);    hold on; % eSated S-compt
xlim(xlR);
ylim(yl); hold on;
set(gca,'YTickLabel','');
l=legend({'$$\hat{R}$$ fast', '$$\hat{R}$$ slow'}, 'interpreter', 'latex', 'location', 'nw', 'box', 'off')
l.ItemTokenSize = [10 8];

nexttile;
plot(plt.times,mean(erprcRAMP_S(find(RT<median(RT)),:)),'-','Color',colors(1,:)); hold on;
plot(plt.times,mean(erprcRAMP_S(find(RT>=median(RT)),:)),'-','Color',colors(2,:));
xlim(xlS); ylim(yl); hold on;
plot([0 0],yl,'k','LineWidth',1);
set(gca,'YTickLabel','');
l = legend({'$$\hat{S}$$ fast', '$$\hat{S}$$ slow'}, 'interpreter', 'latex', 'location', 'ne', 'box', 'off')
l.ItemTokenSize = [10 8];
text(-0.3,yl(2)*1.25,'                S ~ 1 + RT, R ~ 1 + RT')

nexttile;
plot(plt.times,mean(erprcRAMP_R(find(RT<median(RT)),:)),'-','Color',colors(1,:)); hold on;
plot(plt.times,mean(erprcRAMP_R(find(RT>=median(RT)),:)),'-','Color',colors(2,:));
plot([0 0],yl,'--k','LineWidth',1);    hold on; % eSated S-compt
xlim(xlR);
ylim(yl); hold on;
set(gca,'YTickLabel','');
l=legend({'$$\hat{R}$$ fast', '$$\hat{R}$$ slow'}, 'interpreter', 'latex', 'location', 'nw', 'box', 'off')
l.ItemTokenSize = [10 8];


xlabel(t, 'Time(s)')

f.Units = 'centimeters';
f.OuterPosition = [0 0 20 22];