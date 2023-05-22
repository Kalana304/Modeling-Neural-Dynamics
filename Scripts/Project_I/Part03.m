%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% JEB 1444 - Project-I (Part-III)
%
% In this part of the project, two FitzHugh-Nagumo oscillators coupled with
% a linear symmetric coupling factor is implemented. To quantify the system
% behaviour, the max. Lyapunov Exponent was used. 
%
% Author - Kalana Gayal Abeywardena
% Date   - Feb 17th, 2023
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; close all; clear; 

clc; close all; clear; 

load data/part1.mat

Ts = 0.01;
tspan = 0 : Ts : 250;

x0 = [0; 0]; y0 = [0; 0]; xy0 = [x0; y0];   % initial values
c = [0 : 0.01 : 0.5, 0.6 : 0.2 : 2];        % Coupling coefficients

k1 = z(1); deltak = (k1 - z(2:end)) / k1; 

% Response Maps
responsex1 = zeros(length(z) - 1, length(c));
responsex2 = zeros(length(z) - 1, length(c));

for ic = 1 : length(c)
    c0 = c(ic);
    for iz = 2 : length(z)
        k2 = z(iz); k0 = [k1; k2]; 
        [lambx1, lambx2] = lyapunov(c0, k0, tspan, xy0, Ts);
        responsex1(iz - 1, ic) = lambx1;
        responsex2(iz - 1, ic) = lambx2;
        fprintf('c = %.4f | (k1 - k2) / k1 = %.4f | l1 = %.4f, l2 = %.4f\n', c0, deltak(iz-1), lambx1, lambx2);
    end
end

save('data/part3.mat', "responsex1", "responsex2", "deltak", "c", "z");

load data/part3.mat

figure;
imagesc(c, flipud(deltak), responsex1);
set(gca, 'YDir', 'normal'); grid on; 
colormap(jet); colorbar; 
xlabel('Symmetric Coupling Strength (C)'); ylabel('\Delta k / k_1');
title('Response Map of x_1(t) (Max Lyapunov Exponents)')
saveas(gcf, 'data/part3_x1orig.png')

% Calibration
responsex1 = responsex1 - responsex1(:, 1);
responsex2 = responsex2 - responsex2(:, 1);

% Plotting the response map
figure;
imagesc(c, flipud(deltak), responsex1);
set(gca, 'YDir', 'normal'); grid on; 
colormap(jet); colorbar; 
xlabel('Symmetric Coupling Strength (C)'); ylabel('\Delta k / k_1');
title('Response Map of x_1(t) (Max Lyapunov Exponents)')
saveas(gcf, 'data/part3_x1t.png')

figure;
imagesc(c, flipud(deltak), responsex2);
set(gca, 'YDir', 'normal'); grid on; 
colormap(jet); colorbar; 
xlabel('Symmetric Coupling Strength (C)'); ylabel('\Delta k / k_1');
title('Response Map of x_2(t) (Max Lyapunov Exponents)')
saveas(gcf, 'data/part3_x2t.png')

% Binarizing
thresh = 0.01; 

responsex1(responsex1 > thresh) = max(responsex1, [], 'all');
responsex1(responsex1 < -thresh) = min(responsex1, [], 'all');
responsex1(responsex1 < thresh & responsex1 > -thresh) = 0;

figure;
imagesc(c, flipud(deltak), responsex1);
set(gca, 'YDir', 'normal'); grid on; 
colormap(jet);  
xlabel('Symmetric Coupling Strength (C)'); ylabel('\Delta k / k_1');
title('Response Map of x_1(t) (Max Lyapunov Exponents)')
saveas(gcf, 'data/part3_x1t_mask.png')

responsex2(responsex2 > thresh) = max(responsex2, [], 'all');
responsex2(responsex2 < -thresh) = min(responsex2, [], 'all');
responsex2(responsex2 < thresh & responsex2 > -thresh) = 0;

figure;
imagesc(c, flipud(deltak), responsex2);
set(gca, 'YDir', 'normal'); grid on; 
colormap(jet); 
xlabel('Symmetric Coupling Strength (C)'); ylabel('\Delta k / k_1');
title('Response Map of x_1(t) (Max Lyapunov Exponents)')
saveas(gcf, 'data/part3_x2t_mask.png')

function [lambdax1, lambdax2, x1, x2] = lyapunov(c0, k0, tspan, xy0, Ts)
    [t, xy] = ode15s(@(t, xy) couplingfhn(t, xy, k0, c0), tspan, xy0, odeset('BDF', 'on'));

    x1 = xy(:, 1); x2 = xy(:, 2);
    x1 = x1(50 : end); x2 = x2(50 : end);
    
    [~, eLagx1, eDimx1] = phaseSpaceReconstruction(x1);
    lambdax1 = lyapunovExponent(x1, 1 / Ts, eLagx1, eDimx1, 'ExpansionRange', 200);

    [~, eLagx2, eDimx2] = phaseSpaceReconstruction(x2);
    lambdax2 = lyapunovExponent(x2, 1 / Ts, eLagx2, eDimx2, 'ExpansionRange', 200);
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

