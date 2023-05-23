%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% JEB 1444 - Project-II
%
% In this part of the project, non-parameteric modeling of the neuronal 
% system dynamics is performed. To learn the non-prameteric weights from
% the input-output relationships, the data is being generated using the
% couplued FZN oscillators from Project I for different system response 
% regions. 
%
% Author - Kalana Gayal Abeywardena
% Date   - April 07th, 2023
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; close all; clear;

load("../Project 01/data/part3.mat");
load("project_data.mat");               % data of selected system regions

typeind = 1;   % regions: 1: oscillatory | 2: non-oscillatory | 3: chaotic
region_name = region_data.name{typeind};
region_c = region_data.c_values{typeind}; % c value for the selected region
region_z = region_data.z_values{typeind}; % z values of the selected region

fprintf("Running simulations for %s region!\n", region_name);

Ts = 0.01;
tspan = 0 : Ts : 250; npoints = length(tspan);

x0 = [0.5; 0.1]; y0 = [0.1; 0.5]; xy0 = [x0; y0];   % initial values
c0 = region_c(1); k0 = [z(1), region_z(1)];
fprintf('c0 = %.4f | k1 = %.4f | k2 = %.4f\n', c0, k0);

% Low-Pass Filtering of noise (for aliasing issues)
fp = 50; 
noise_filt = lowpass(ntrain, fp, 2 / Ts);  
noise_filt_test = lowpass(ntest, fp, 2 / Ts);

gwn0 = @(t) interp1(tspan, noise_filt, t, "cubic"); % pseudo random noise

[t, xy] = ode15s(@(t, xy) couplingfhn(t, xy, k0, c0, gwn0), tspan, xy0,...
                  odeset('BDF', 'on'));

%% Part 1: Wiener-Bose Model Estimation
x1 = xy(:,1); 

X1 = fftshift(fft(x1)) / npoints; Nfft = fftshift(fft(noise_filt)) / npoints;
fspan = (-(npoints - 1) / 2 : (npoints - 1) / 2) / (npoints * Ts);

figure(1)
subplot(2,1,1); plot(tspan, noise_filt); ylabel('Volatge'); xlabel('Time(s)'); title('Input Signal');
subplot(2,1,2); plot(tspan, x1); ylabel('Volatge'); xlabel('Time(s)'); title('Output Signal');

figure(2)
subplot(2,1,1); stem(fspan, abs(Nfft), 'filled', '.'); ylabel('Magnitude'); xlabel('Hz'); title('Spectrum of Input Signal');
subplot(2,1,2); stem(fspan, abs(X1), 'filled', '.'); ylabel('Magnitude'); xlabel('Hz'); title('Spectrum of Output Signal');

figure(3);
plot(tspan, x1); ylabel("Voltage (mV)"); xlabel('Time (s)');
title(sprintf('System Response for c = %.4f | k_1 = %.4f | k_2 = %.4f', c0, k0(1), k0(2)))

% Estimating the kernel K1
P = mean(noise_filt .^ 2);        % Power of the input signal
G0 = mean(x1);                    % 0th Volterra Kernal
[K1, Tau] = xcorr(x1 - G0, noise_filt, [], 'coeff');

K1 = K1(Tau >= 0); Tau = Tau(Tau >= 0);
K1abs = abs(K1) / P; Tau = Tau * Ts;

figure(4);
plot(Tau, K1abs, 'k',  'HandleVisibility','off'); hold on;
plot(exp(-4.5) * ones(npoints, 1), 'LineWidth', 1.2, 'DisplayName', 'exp(-4.5)')
ylabel('|Magnitude|'); xlabel('Tau'); legend(); xlim([0, tspan(end)])
title('K_1(\tau)'); 

% Estimate L and alpha
BWx1 = obw(x1) / (2 * pi);  % System Bandwidth estimation

idU = find(K1abs< exp(-4.5));    % Find the samples where the magnitude is below 3dB
idU_diff = idU(2 : end) - idU(1 : end - 1);
SysMemId = find(idU_diff > 1);
SysMemId = min(npoints, SysMemId(end));

SysMem = idU(SysMemId);

fprintf("System Bandwidth = %.4f per sample | System Memory = %d sample\n", BWx1, SysMem);

M_nms = SysMem * BWx1;

LThresh = 25; L = min(max(1, ceil(M_nms / LThresh)), 15);
alphThresh = 45; alpha = exp(-alphThresh / (SysMem  * Ts - 1));

fprintf("L = %d | alpha = %.4f\n", L, alpha);

[Cest, Kest, yWBout, WB_mse] = LET_1(noise_filt, x1, alpha, L, 2, 5);

laguerres = function_generate_laguerre(alpha, L, tspan(end));

figure(7);
plot(laguerres, 'LineWidth', 1.2);
title(sprintf('Laguerre Functions with \\alpha = %.4f | L = %d', alpha, L));

%% Part 2: Principle Dynamic Modes (Modular Model Estimation)

Qsize = size(Kest.k2) + 1; Q = zeros(Qsize);
Q(1,1) = Kest.k0; 
Q(2 : end, 1) = Kest.k1 / 2; Q(1, 2 : end) = Kest.k1 / 2;
Q(2 : end, 2 : end) = Kest.k2;

[~, qlambda, ~] = svd(Q);
qlambda = diag(qlambda) .^ 2;

Totaleigen = sum(qlambda); epsilon = 0.01;
eigPercent = qlambda / Totaleigen;

nPMDs = sum(eigPercent > epsilon) + 2;
fprintf("No. PMDs = %d\n", nPMDs);

figure;
[Npdms, PDMs, ANFs, yPDMout, M_nms] = PDM_1(noise_filt, x1, alpha, L, nPMDs, 8);

y_pdm = convn(PDMs, noise_filt);
yn = ANFs.const + ANFs.pdm1(1) * y_pdm + ANFs.pdm1(2) * y_pdm .^2 + ANFs.pdm1(3) * y_pdm .^ 3;

%% Binarizing FZN Model Output

origThresh = input("Enter threshold value for FZN Model Output : "); %0.2/0/0.1
x1t = 0.5 + 0.5 * sign(x1 - origThresh); 
x1Binary = binarize(x1t, Ts);
x1Binary = refractorniess(x1Binary, x1Binary, region_name);

figure(13)
plot(tspan, x1Binary); ylabel('Voltage'); xlabel('Time (s)');
title('Binarized FZN Model')

%% Part 3(a): Comparison between parametric and Wiener-Bose non-parametric

tolerance = 1;
global fig_count;
fig_count = 0;

[FPR_WB, TPR_WB] = aur_eval(x1Binary, yWBout, tspan, Ts, tolerance, region_name, 'Wiener-Bose', 14);

trigThresh = (min(yWBout) + max(yWBout)) / 2 : 0.00001 : max(yWBout);

[FPR_WB, TPR_WB, trigThresh] = aur_clean(FPR_WB, TPR_WB, trigThresh);

[FPR_WB_sorted, idx_WB] = sort(FPR_WB); TPR_WB_sorted = TPR_WB(idx_WB);

Int = cumtrapz(FPR_WB_sorted, TPR_WB_sorted);
Intv = @(a,b) max(Int(FPR_WB_sorted<=b)) - min(Int(FPR_WB_sorted>=a));
AUC = Intv(FPR_WB_sorted(1), FPR_WB_sorted(end));

fprintf("AUC = %.5f\n", AUC)

Intrand = cumtrapz(FPR_WB_sorted, FPR_WB_sorted);
Intvrand = @(a,b) max(Intrand(FPR_WB_sorted<=b)) - min(Intrand(FPR_WB_sorted>=a));
AUCrand = Intvrand(FPR_WB_sorted(1), FPR_WB_sorted(end));

fprintf("AUC_rand = %.5f\n", AUCrand)

figure(14 + fig_count);
plot(FPR_WB, TPR_WB, '-o', 'MarkerSize', 2, ...
            'MarkerFaceColor', 'b', 'LineWidth', 1.2); 
hold on; plot(FPR_WB, FPR_WB, 'LineWidth', 1.2); ylabel('TPR'); xlabel('FPR');
hold on; plot(FPR_WB, -FPR_WB + 1, '--', 'LineWidth', 1.2); 
legend('ROC Curve', 'Random System', 'Location', 'northwest')
title(sprintf('ROC Curve - FZN Model vs. Wiener-Bose Model\nAUC = %.4f', AUC)); 
fig_count = fig_count + 1;

wb_maxId = find(TPR_WB < -FPR_WB + 1);
wb_maxId = wb_maxId(end) + 1;
opt_trigger_WB = trigThresh(wb_maxId);

fprintf("Optimal trigger threshold for WB Model output for test noise = %.4f\n", ...
            opt_trigger_WB);

y1WB = 0.5 + 0.5 * sign(yWBout - opt_trigger_WB); 
y1WBBinary = binarize(y1WB, Ts);
[y1WBrefactory, Tr] = refractorniess(x1Binary, y1WBBinary, region_name);

figure(14 + fig_count)
subplot(3, 1, 1); plot(tspan, x1Binary); title('Parametric Model Output')
subplot(3, 1, 2); plot(tspan, y1WBBinary); xlabel('Time(s)'); ylabel('Voltage (Binary)')
title(sprintf(['Wiener-Bose Model (No Refractoriness) Output \n' ...
    '\\beta = %.4f | FPR = %.4f | TPR = %.4f'],...
     opt_trigger_WB, FPR_WB(wb_maxId), TPR_WB(wb_maxId)))
subplot(3, 1, 3); plot(tspan, y1WBrefactory); xlabel('Time(s)'); ylabel('Voltage (Binary)')
title(sprintf(['Wiener-Bose Model (with Refractoriness Tr = %.2f) Output \n' ...
    '\\beta = %.4f | FPR = %.4f | TPR = %.4f'],...
     Tr* Ts, opt_trigger_WB, FPR_WB(wb_maxId), TPR_WB(wb_maxId)))

fig_count = fig_count + 1;

%% Part 3(b): Comparison between parametric and PDM non-parametric

[FPR_PDM, TPR_PDM] = aur_eval(x1Binary, yPDMout, tspan, Ts, tolerance, region_name, 'PDM Modular Model', 14);

trigThresh = (min(yPDMout) + max(yPDMout)) / 2 : 0.00001 : max(yPDMout);

[FPR_PDM, TPR_PDM, trigThresh] = aur_clean(FPR_PDM, TPR_PDM, trigThresh);

[FPR_PDM_sorted, idx_PDM] = sort(FPR_PDM); TPR_PDM_sorted = TPR_PDM(idx_PDM);

Int = cumtrapz(FPR_PDM_sorted, TPR_PDM_sorted);
Intv = @(a,b) max(Int(FPR_PDM_sorted<=b)) - min(Int(FPR_PDM_sorted>=a));
AUC = Intv(FPR_PDM_sorted(1), FPR_PDM_sorted(end));

fprintf("AUC = %.5f\n", AUC)

Intrand = cumtrapz(FPR_PDM_sorted, FPR_PDM_sorted);
Intvrand = @(a,b) max(Intrand(FPR_PDM_sorted<=b)) - min(Intrand(FPR_PDM_sorted>=a));
AUCrand = Intvrand(FPR_PDM_sorted(1), FPR_PDM_sorted(end));

fprintf("AUC_rand = %.5f\n", AUCrand)

figure(14 + fig_count);
plot(FPR_PDM, TPR_PDM, '-o', 'MarkerSize', 2, ...
            'MarkerFaceColor', 'b', 'LineWidth', 1.2); ylabel('TPR'); xlabel('FPR');
hold on; plot(FPR_PDM, FPR_PDM, 'LineWidth', 1.2); 
hold on; plot(FPR_PDM, 1 - FPR_PDM, '--', 'LineWidth', 1.2); 
legend('ROC Curve', 'Random System', 'Location', 'northwest');
title(sprintf('ROC Curve - FZN Model vs. PDM Modular Model\nAUC = %.4f', AUC));
fig_count = fig_count + 1;

pdm_maxId = find(TPR_PDM < -FPR_PDM + 1);
pdm_maxId = pdm_maxId(end) + 1;
opt_trigger_PDM = trigThresh(pdm_maxId);

fprintf("Optimal trigger threshold for PMD Model output for test noise = %.4f\n", ...
            opt_trigger_PDM);

y1PDM = 0.5 + 0.5 * sign(yPDMout - opt_trigger_PDM); 
y1PDMBinary = binarize(y1PDM, Ts);
[y1PDMrefactory, Tr] = refractorniess(x1Binary, y1PDMBinary, region_name);

figure(14 + fig_count)
subplot(3, 1, 1); plot(tspan, x1Binary); title('Parametric Model Output')
subplot(3, 1, 2); plot(tspan, y1PDMBinary); xlabel('Time(s)'); ylabel('Voltage (Binary)')
title(sprintf('PDM Modular Model (No Refractoriness) Output \n\\beta = %.4f | FPR = %.4f | TPR = %.4f',...
                    opt_trigger_PDM, FPR_PDM(pdm_maxId), TPR_PDM(pdm_maxId)))
subplot(3, 1, 3); plot(tspan, y1PDMrefactory); xlabel('Time(s)'); ylabel('Voltage (Binary)')
title(sprintf('PDM Modular Model (with Refractoriness Tr = %.2f) Output \n\\beta = %.4f | FPR = %.4f | TPR = %.4f',...
                    Tr* Ts, opt_trigger_PDM, FPR_PDM(pdm_maxId), TPR_PDM(pdm_maxId)))

%% Saving useful parameters for test noise

save(sprintf('params_c_%.2f_k1_%.2f_k2_%.2f_nPDMs_%d.mat', c0, k0, nPMDs), ...
            "ANFs", "PDMs", "Kest", "noise_filt_test", "nPMDs", ...
            "npoints", "c0", "k0", "xy0", "Ts", "tspan", "tolerance", ...
            "opt_trigger_PDM", "opt_trigger_WB");

function out_sig = binarize(in_sig, Ts)
    out_sig = zeros(size(in_sig));
    grad = gradient(in_sig, Ts); posID = find(grad > 0); 
    if ~isempty(posID)
        deltposID = diff(posID);
        deltposID = find(deltposID ~= 1); deltposID = [1; deltposID + 1];
        posID = posID(deltposID);
        out_sig(posID) = 1;
    end
end

function [out_sig, Tr] = refractorniess(fzn_sig, model_sig, region_name)
    if strcmp(region_name, 'oscillatory')
        Tr = mean(diff(find(fzn_sig == 1))); 
    elseif strcmp(region_name, 'chaotic')
        Tr = max(diff(find(fzn_sig == 1)));
    else
        fprintf('Error in estimating Tr!!')
    end

    npoints = length(model_sig);
    out_sig = zeros(size(model_sig));

    start_idx = find(fzn_sig == 1); start_idx = start_idx(1); 
    modelOnes = find(model_sig == 1); 
    diffArray = abs(modelOnes - start_idx); mindiff = min(diffArray); 
    startPoint = find(diffArray == mindiff);
    startPoint = startPoint(1); 

    scount = 0;
    for t = modelOnes(startPoint) : npoints
        if model_sig(t) == 1 && scount == 0
            out_sig(t) = 1; scount = scount + 1;
        elseif model_sig(t) == 0 && scount == 0
            out_sig(t) = 0; 
        else
            out_sig(t) = 0; scount = scount + 1;
        end
        if scount >= round(Tr * 0.8)
            scount = 0;
        end
    end
end

function [fpr_new, tpr_new, triger_new] = aur_clean(fpr_old, tpr_old, triger_old)
    uniqfpr = unique(fpr_old);

    fpr_new = zeros(size(uniqfpr));
    tpr_new = zeros(size(uniqfpr));
    triger_new = zeros(size(uniqfpr));
    
    for nfpr = 1 : length(uniqfpr)
        fpr_val = uniqfpr(nfpr); fpr_index = find(fpr_old == fpr_val);
        tpr_val = tpr_old(fpr_index);

        [maxTpr, maxTprid] = max(tpr_val);
        fpr_new(nfpr) = fpr_val; tpr_new(nfpr) = maxTpr;
        triger_new(nfpr) = triger_old(fpr_index(maxTprid));
    end
end

function dxydt = couplingfhn(t, xy, k0, c0, gwn)
    a = 0.7; b = 0.8; 
    alpha = 3; omega2 = 1;

    x1 = xy(1); y1 = xy(3); k1 = k0(1);
    x2 = xy(2); y2 = xy(4); k2 = k0(2);

    dx1dt = alpha * (y1 + x1 - (x1^3)/3 + k1 + c0 * x2 + gwn(t));
    dy1dt = -(omega2 * x1 - a + b * y1) / alpha;

    dx2dt = alpha * (y2 + x2 - (x2^3)/3 + k2 + c0 * x1);
    dy2dt = -(omega2 * x2 - a + b * y2) / alpha;
    
    dxydt = [dx1dt ; dx2dt ; dy1dt ; dy2dt];
end

function [FPR, TPR] = aur_eval(gtBrinary, model_pred, tspan, Ts, tolerance, sys_region, model_name, figno)
    global fig_count

    trigThresh = (min(model_pred) + max(model_pred)) / 2 : 0.00001 : max(model_pred);
    FPR = zeros(length(trigThresh), 1); TPR = zeros(length(trigThresh), 1);
    
    for ntrig = 1 : length(trigThresh)
        trigThreshold = trigThresh(ntrig);
    
        y1 = 0.5 + 0.5 * sign(model_pred - trigThreshold); 
        y1Binary = binarize(y1, Ts);
        if strcmp(sys_region, 'oscillatory') || strcmp(sys_region, 'chaotic')
            [~, Tr] = refractorniess(gtBrinary, y1Binary, sys_region);
        end

        [TP, TN, FP, FN] = evaluation(gtBrinary, y1Binary, Tr * Ts, Ts);
        TPR(ntrig) = TP / (TP + FN); FPR(ntrig) = FP / (FP + TN);
    
        if (rem(ntrig, 10000) == 0)
            fprintf("Threshold = %.4f | TPR = %.4f | FPR = %.4f | TP = %d | FN = %d | FP = %d | TN = %d\n", ...
                    trigThreshold, TPR(ntrig), FPR(ntrig), TP, FN, FP, TN);
%             figure(figno + fig_count);
%             subplot(2, 1, 1); plot(tspan, gtBrinary); title('FZN Output')
%             subplot(2, 1, 2); plot(tspan, y1Binary); title([model_name, ' Output'])
            fig_count = fig_count + 1;
        end
    end
end