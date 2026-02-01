close all

% a1 = 0.96
% a2 = 0.03
% figure(1)
% hold on

% ezplot('y = x*log(x) + (1-x)*log(1-x)')
% %  ezplot('y =  ((2*x-0.96-0.03)/(0.96-0.03)).^3 - (2*x-0.96-0.03)/(0.96-0.03)')
% %ezplot('y =  x.^3 - x')
% % ezplot('y = log(x)')
% ezplot('y = log(x/(1-x))')
% 
%  figure(2)
% % ezplot('y = 1/(4)*x^4 - (1/2)*x^2')
% 
% ezplot('y = log(x/(1-x))')
% 
% %((2*x-a1-a2)/(a1-a2))^3 -

x = linspace(0.001,0.999,500);

R = 8.134;
T = 25+273;
cmax = 1;
Omega1 = 0;
Omega2 = 4000;
Omega3 = 6000;

fmix = R*T*cmax*(x.*log(x) + (1-x).*log(1-x));
fint1 = Omega1*cmax * x .* (1 - x);
fint2 = Omega2*cmax * x .* (1 - x);
fint3 = Omega3*cmax * x .* (1 - x);

mu1 = R*T*log(x./(1-x)) + Omega1*(1 - 2*x);
mu2 = R*T*log(x./(1-x)) + Omega2*(1 - 2*x);
mu3 = R*T*log(x./(1-x)) + Omega3*(1 - 2*x);

figure(1)
subplot(1,2,1)
hold on
plot(x,fmix+fint1)
plot(x,fmix+fint2)
plot(x,fmix+fint3)
xlabel('$c/c_{\max}$')
ylabel('$f_{\rm hom} (\textrm{J/m}^3)$')

subplot(1,2,2)
hold on
plot(x,mu1)
plot(x,mu2)
plot(x,mu3)
xlabel('$c/c_{\max}$')
ylabel('$\mu$ (J/mol)')
legend('$\Omega = 0$ kJ/mol','$\Omega = 4$ kJ/mol','$\Omega = 6$ kJ/mol')
