clear all
close all

% Frequency range
f_exponent_h = 3;
f_exponent_l = -2;

f = logspace(f_exponent_l, f_exponent_h, 500);   % Hz
w = 2*pi*f;
s = 1j*w;

% different a

a = 0.5;
Q = 1;

figure(1)
axis equal
set(gcf, 'Position', [50 50 500 500])
% subplot(1,2,1)
grid on

hold on
% for a = alphas
%     Z = 1 ./ (Q * (1j*w).^a);
%     plot(real(Z), -imag(Z), 'LineWidth', 2, ...
%          'DisplayName', ['$\alpha$ = ' num2str(a)])
% end
Z0 = 1 ./ (Q * (1j*w).^a);
Z1 = tanh(sqrt(s))./sqrt(s);
Z2 = 2*tanh(2*sqrt(s))./(2*sqrt(s));
Z3 = 3*tanh(3*sqrt(s))./(3*sqrt(s));

plot(real(Z1), -imag(Z1), 'LineWidth', 2, ...
         'DisplayName', ['$L = 1$'])
plot(real(Z2), -imag(Z2), 'LineWidth', 2, ...
         'DisplayName', ['$L = 2$'])
plot(real(Z3), -imag(Z3), 'LineWidth', 2, ...
         'DisplayName', ['$L = 3$'])
plot(real(Z0), -imag(Z0), 'LineWidth', 2, ...
         'DisplayName', ['$L = \infty$'])
plot(real(Z4), -imag(Z4), 'LineWidth', 2, ...
         'DisplayName', ['$L = 1$'])   
     
xlabel('$Z'' (\Omega)$','interpreter','latex')
ylabel('$-Z'''' (\Omega)$','interpreter','latex')
axis([0 3 0 3])
legend show
legend('location','best')
% legend('interpreter','latex') 
% figure(2)
% subplot(2,1,1)
% ax = gca;
% ax.XScale = 'log';
% ax.YScale = 'log';
% hold on
% loglog(f, abs(Z0))
% loglog(f, abs(Z1))
% loglog(f, abs(Z2))
% loglog(f, abs(Z3))
% 
% subplot(2,1,2)
% ax = gca;
% ax.XScale = 'log';
% ax.YScale = 'log';
% hold on
% semilogx(f, angle(Z0)*180/pi)
% semilogx(f, angle(Z1)*180/pi)
% semilogx(f, angle(Z2)*180/pi)
% semilogx(f, angle(Z3)*180/pi)

