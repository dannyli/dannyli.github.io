clear all
close all

% Frequency range
f_exponent_h = 3;
f_exponent_l = -1;

f = logspace(f_exponent_l, f_exponent_h, 500);   % Hz
w = 2*pi*f;

% different a

alphas = [0.1 0.25 0.5  0.75 1];
Q = 1;

figure
set(gcf, 'Position', [50 50 1200 500])
subplot(1,2,1)

hold on
for a = alphas
    Z = 1 ./ (Q * (1j*w).^a);
    plot(real(Z), -imag(Z), 'LineWidth', 2, ...
         'DisplayName', ['$\alpha$ = ' num2str(a)])
end


% LF 

a = linspace(0.01,1,100);
f = 10^(f_exponent_l);
w = 2*pi*f;

Z = 1 ./ (Q * (1j*w).^a);
plot(real(Z), -imag(Z), 'k--', 'LineWidth', 2, ...
         'DisplayName', ['$\omega = 2 \pi \times 10^{' num2str(f_exponent_l) '}$'])
     
% HF 
f = 10^(f_exponent_h);
w = 2*pi*f;

Z = 1 ./ (Q * (1j*w).^a);
plot(real(Z), -imag(Z), 'g--', 'LineWidth', 2, ...
         'DisplayName', ['$\omega = 2 \pi \times 10^{' num2str(f_exponent_h) '}$'])
     
legend show
grid on
xlabel('$Z'' (\Omega)$')
ylabel('$-Z'''' (\Omega)$')
legend 
axis equal
title('Nyquist Plot for Different CPE Exponents')

%%

% Frequency range
f = logspace(-2, 5, 500);   % Hz
w = 2*pi*f;

% CPE parameters

% CPE impedance
% Z = 1 ./ (Q * (1j*w).^alpha);
% alphas = [0.75 1];

% figure

subplot(2,2,2)
ax = gca;
ax.XScale = 'log';
ax.YScale = 'log';
hold on
for a = alphas
    Z = 1 ./ (Q * (1j*w).^a);
    loglog(f, abs(Z), 'LineWidth', 2, ...
         'DisplayName', ['$\alpha$ = ' num2str(a)])
end

get(gca,'XScale'), get(gca,'YScale')

grid on
ylabel('$|Z|$ ($\Omega$)')
title('Bode Plot of CPE')

subplot(2,2,4)
ax = gca;
ax.XScale = 'log';
ax.YScale = 'log';
hold on
for a = alphas
    Z = 1 ./ (Q * (1j*w).^a);
    semilogx(f, angle(Z)*180/pi, 'LineWidth', 2, ...
         'DisplayName', ['$\alpha$ = ' num2str(a)])
end

grid on
xlabel('Frequency (Hz)')
ylabel('Phase (deg)')
