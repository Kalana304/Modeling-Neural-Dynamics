%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% JEB 1444 - Project-I (Part-I)
%
% In this part of the project, the FitzHugh-Nagumo model was implemented
% using differential equations for a single neuron dynamics. First, for a
% given system parameters, the intrinsic frequency bandwidth was found
% where the oscillatory rhythms are generated. 

% Author - Kalana Gayal Abeywardena
% Date   - Feb 17th, 2023
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; close all; clear;

Ts = 0.0001;
tspan = 0 : Ts : 250;

x0 = 0; y0 = 0; xy0 = [x0; y0]; % initial values
z = -1.43 : 0.01 : -0.33; 
f0array = []; 

parfor k = 1 : length(z)
    disp(k)
    z0 = z(k);

    [t, xy, tz, yz, ~] = ode15s(@(t, xy) fhnsolver(t, xy, z0), tspan, xy0, odeset('BDF', 'on', 'Events', @zerocross));
    
    % Extract individual solution values
    x = xy(:,1);
    y = xy(:,2);
    
    if length(tz) > 1
        f0 = 1 / mean(tz(3 : end) - tz(2 : end - 1));
    else
        f0 = 0;
    end
    f0array(k) = f0; 
end

z(f0array == 0) = [];
f0array(f0array == 0) = [];

figure;
plot(z, f0array, 'LineWidth', 1.2); ylabel('Intrinsic Frequency');
xlabel('Z Values'); 
saveas(gcf, 'data/part1_intrinsic_freq.png')

% Saving the intrinsic freqs and the corresponding input stimuli
save('data/part1.mat', "f0array", "z");

function dxydt = fhnsolver(t, xy, z)
    a = 0.7; b = 0.8; 
    alpha = 3; omega2 = 1;
    
    x = xy(1); y = xy(2);

    dxdt = alpha * (y + x - (x^3)/3 + z);
    dydt = -(omega2 * x - a + b * y) / alpha;
    dxydt = [dxdt ; dydt];        
end

function [position,isterminal,direction] = zerocross(t, xy)
  position = xy(1); % The value that we want to be zero
  isterminal = 0;   % Continue integration 
  direction = -1;   % Negative direction only
end