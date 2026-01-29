%% 

% Frequency range
f = logspace(-2, 5, 800);   % Hz
w = 2*pi*f;

% -------- R || CPE #1 (High-frequency arc) --------
R1 = 2;          % Ohm
Q1 = 5e-3;       % S*s^alpha
alpha1 = 0.85;

Z1 = 1 ./ (1/R1 + Q1*(1j*w).^alpha1);

% -------- R || CPE #2 (Low-frequency arc) --------
R2 = 6;          % Ohm
Q2 = 2e-5;       % S*s^alpha
alpha2 = 0.75;

Z2 = 1 ./ (1/R2 + Q2*(1j*w).^alpha2);

% -------- Total impedance (series) --------
Z_total = Z1 + Z2;

% -------- Nyquist plot --------
figure
plot(real(Z_total), -imag(Z_total), 'LineWidth', 2)
grid on
axis equal
xlabel('Z'' (\Omega)')
ylabel('-Z'''' (\Omega)')
title('Nyquist Plot: (R_1 || CPE_1) + (R_2 || CPE_2)')

