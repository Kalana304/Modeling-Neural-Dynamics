%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% JEB 1444 - Project-I (Part-II)
%
% In this part of the project, two FitzHugh-Nagumo oscillators coupled with
% a linear symmetric coupling factor is implemented. To quantify the system
% behaviour, the phase-synchronization index (R-Index) was used. 
%
% Author - Kalana Gayal Abeywardena
% Date   - Feb 17th, 2023
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; close all; clear; 

load data/part1.mat

Ts = 0.01;
tspan = 0 : Ts : 250;

x0 = [0; 0]; y0 = [0; 0]; xy0 = [x0; y0]; % initial values
c = [0 : 0.01 : 0.5, 0.6 : 0.2 : 4]; 
wavelet = 'morse';                  % Complex Wavelet function

nFreq = length(z); indk1 = floor(nFreq / 2);
k1 = z(indk1); f1 = f0array(indk1);
deltak = (k1 - z) / k1; 

% Response Maps
response = zeros(length(z) - 1, length(c));

for ic = 1 : length(c)
    c0 = c(ic);
    fprintf('Running :: c = %.3f\n', c0);
    for iz = 2 : length(z)
        k2 = z(iz); f2 = f0array(iz);
        k0 = [k1; k2]; f0 = [f1; f2];

        R = phasesync(c0, k0, f0, tspan, xy0, wavelet, Ts);
        response(iz - 1, ic) = R;
    end
end

save(sprintf('data/part2_%s_wavlet.mat', wavelet), "response", "deltak", "c", "z", "f0array");

load data/part2_morse_wavlet.mat
wavelet = 'morse';  

% Plotting the response map

figure;
imagesc(c, flipud(deltak), response);
set(gca, 'YDir', 'normal'); grid on; 
colormap(jet); colorbar; 
xlabel('Symmetric Coupling Strength (C)'); ylabel('\Delta k / k_1');
title('Response Map (R - Index)')
saveas(gcf, sprintf('data/part2_%s_wavlet_R_index.png', wavelet));


function r = phasesync(c0, k0, f0, tspan, xy0, wavelet, Ts)
    [t, xy] = ode15s(@(t, xy) couplingfhn(t, xy, k0, c0), tspan, xy0, odeset('BDF', 'on'));

    x1 = xy(:, 1); x2 = xy(:, 2);

    % Performing the CWT
    [Cx1, fx1] = cwt(x1, wavelet, 1 / Ts); 
    [Cx2, fx2] = cwt(x2, wavelet, 1 / Ts);

    [~, ix1] = min(abs(fx1 - f0(1)));
    [~, ix2] = min(abs(fx2 - f0(2)));
    
    delta_phi = angle(Cx1(ix1, :)) - angle(Cx2(ix2, :));
    r = abs(mean(exp(1i*delta_phi)));   % Take the mean over all frequencies
end

function dxydt = couplingfhn(~, xy, k0, c0)
    a = 0.7; b = 0.8; 
    alpha = 3; omega2 = 1;
 
    x1 = xy(1); y1 = xy(3); k1 = k0(1);
    x2 = xy(2); y2 = xy(4); k2 = k0(2);

    dx1dt = alpha * (y1 + x1 - (x1^3)/3 + k1 + c0 * x2);
    dy1dt = -(omega2 * x1 - a + b * y1) / alpha;

    dx2dt = alpha * (y2 + x2 - (x2^3)/3 + k2 + c0 * x1);
    dy2dt = -(omega2 * x2 - a + b * y2) / alpha;
    
    dxydt = [dx1dt ; dx2dt ; dy1dt ; dy2dt];
end

