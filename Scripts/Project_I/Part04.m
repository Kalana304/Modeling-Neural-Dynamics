%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% JEB 1444 - Project-I (Part-IV)
%
% In this part of the project, two FitzHugh-Nagumo oscillators coupled with
% a linear symmetric coupling factor is implemented. To quantify the system
% behaviour, the correlation dimension was used. 
%
% Author - Kalana Gayal Abeywardena
% Date   - Feb 17th, 2023
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; close all; clear; 

load data/part1_intrinsic_freq.mat

Ts = 0.01;
tspan = 0 : Ts : 250;

x0 = [0; 0]; y0 = [0; 0]; xy0 = [x0; y0]; % initial values
c = [0 : 0.01 : 1.0, 1.2 : 0.2 : 4];      % Coupling coefficients

k1 = z(1); deltak = (k1 - z) / k1; 

% Response Maps
responsex1_cordim = zeros(length(z) - 1, length(c));
responsex2_cordim = zeros(length(z) - 1, length(c));

for ic = 1 : length(c)
    c0 = c(ic);
    for iz = 1 : length(z)
        k2 = z(iz); k0 = [k1; k2]; 
        [cordimx1, cordimx2] = corrdim(c0, k0, tspan, xy0, Ts);
        responsex1_cordim(iz, ic) = cordimx1;
        responsex2_cordim(iz, ic) = cordimx2;
        fprintf('c = %.4f | (k1 - k2) / k1 = %.4f | dim1 = %.2f, dim2 = %.2f\n',...
            c0, deltak(iz), cordimx1, cordimx2);
    end
end

save('data/part4.mat', "responsex1_cordim", "responsex2_cordim", "deltak", "c", "z");

load data/part4.mat 

% Plotting the response map
figure;
imagesc(c, flipud(deltak), responsex1_cordim);
set(gca, 'YDir', 'normal'); grid on; 
colormap(flipud(hot)); colorbar;  
xlabel('Symmetric Coupling Strength (C)'); ylabel('\Delta k / k_1');
title('Response Map of x_1(t) (D_c)')
saveas(gcf, 'data/part4_x1.png')

figure;
imagesc(c, flipud(deltak), responsex2_cordim);
set(gca, 'YDir', 'normal'); grid on; 
colormap(flipud(hot)); colorbar;  
xlabel('Symmetric Coupling Strength (C)'); ylabel('\Delta k / k_1');
title('Response Map of x_2(t) (D_c)')
saveas(gcf, 'data/part4_x2.png')

function [corDimx1, corDimx2] = corrdim(c0, k0, tspan, xy0, Ts)
    [t, xy] = ode15s(@(t, xy) couplingfhn(t, xy, k0, c0), tspan, xy0, odeset('BDF', 'on'));

    x1 = xy(:, 1); x2 = xy(:, 2);

    [~, eLagx1, eDimx1] = phaseSpaceReconstruction(x1);
    corDimx1 = correlationDimension(x1, eLagx1, eDimx1);

    [~, eLagx2, eDimx2] = phaseSpaceReconstruction(x2);
    corDimx2 = correlationDimension(x2, eLagx2, eDimx2);
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
