% Infinite-Horizon Continuous-Time LQR Control Simulation
clear; clc;

% --------------------------------------
% 1. System definition (continuous-time)
% --------------------------------------
A = [0 1; -2 -3];       % State matrix
B = [0; 1];             % Input matrix
Q = eye(2);             % State weighting
R = 1;                  % Input weighting

% --------------------------------------
% 2. Solve continuous-time LQR via CARE
% --------------------------------------
[K, P, eigVals] = lqr(A, B, Q, R);  % Compute optimal feedback gain

disp('LQR feedback gain K:');
disp(K);

% --------------------------------------
% 3. Simulation setup
% --------------------------------------
x0 = [1; 0];           % Initial state
t_span = [0 10];       % Simulation time interval

% --------------------------------------
% 4. Define closed-loop system dynamics
% dx/dt = (A - B*K)*x
% --------------------------------------
A_cl = A - B * K;      % Closed-loop system matrix

f = @(t, x) A_cl * x;  % Anonymous function for ODE solver

% --------------------------------------
% 5. Simulate using ODE45
% --------------------------------------
[t, X] = ode45(f, t_span, x0);  % Solve ODE

% --------------------------------------
% 6. Compute control input u(t) = -K * x(t)
% --------------------------------------
U = -X * K';           % Each row of X multiplied by -K

% --------------------------------------
% 7. Plot results
% --------------------------------------
figure;

subplot(2,1,1);
plot(t, X(:,1), 'r-', 'LineWidth', 2); hold on;
plot(t, X(:,2), 'b--', 'LineWidth', 2);
xlabel('Time t'); ylabel('State $x(t)$');
legend('$x_1$','$x_2$');
title('LQR Closed-Loop State Response');
grid on;

subplot(2,1,2);
plot(t, U, 'k-', 'LineWidth', 2);
xlabel('Time $t$'); ylabel('Control input $u(t)$');
title('LQR Control Input');
grid on;