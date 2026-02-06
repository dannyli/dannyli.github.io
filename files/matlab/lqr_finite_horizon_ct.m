% Finite-Horizon Continuous-Time LQR Simulation
clear; clc;

% --------------------------------------
% 1. System definition (continuous-time)
% --------------------------------------
A = [0 1; -2 -3];       % State matrix
B = [0; 1];             % Input matrix
Q = eye(2);             % State weighting
R = 1;                  % Input weighting


[K, P_inf, eigVals] = lqr(A, B, Q, R);  % Compute optimal feedback gain


S = 10 * eye(2);        % Terminal state cost matrix
% S = P_inf;


t0 = 0; tf = 10;        % Time interval
x0 = [1; 0];            % Initial state

% --------------------------------------
% 2. Solve Riccati Differential Equation backward in time
% --------------------------------------
% Convert matrix P to vector for ODE solver (column-stacked)
P_final = S;
p0 = reshape(P_final, [], 1);  % Vectorize P(t_f)

% Riccati ODE: dP/dt = -A'*P - P*A + P*B*inv(R)*B'*P - Q
riccati_ode = @(t, p) ...
    -reshape(A'*reshape(p,2,2) + reshape(p,2,2)*A ...
    - reshape(p,2,2)*B*(R\B')*reshape(p,2,2) + Q, [], 1);

% Integrate backward in time
[t_back, p_sol] = ode45(riccati_ode, linspace(tf, t0, 100), p0);

% Flip time to increasing order
t_vec = flipud(t_back);
P_t = flipud(p_sol);

% Interpolate K(t) at arbitrary time
K_func = @(t) ...
    (B' * reshape(interp1(t_vec, P_t, t, 'linear', 'extrap'), 2, 2)) / R;


% --------------------------------------
% 3. Define closed-loop dynamics with time-varying feedback
% dx/dt = (A - B*K(t)) * x
% --------------------------------------
f = @(t, x) (A - B * K_func(t)) * x;

% --------------------------------------
% 4. Simulate system dynamics
% --------------------------------------
[t_sim, X] = ode45(f, [t0 tf], x0);

% Compute u(t) = -K(t) * x(t)
U = zeros(length(t_sim),1);
for i = 1:length(t_sim)
    Kt = K_func(t_sim(i));
    U(i) = -Kt * X(i,:)';
end

% --------------------------------------
% 5. Plot results
% --------------------------------------
figure;

subplot(2,1,1);
plot(t_sim, X(:,1), 'r-', 'LineWidth', 2); hold on;
plot(t_sim, X(:,2), 'b--', 'LineWidth', 2);
xlabel('Time $t$'); ylabel('State $x(t)$');
legend('$x_1$','$x_2$');
title('Finite-Horizon LQR State Response');
grid on;

subplot(2,1,2);
plot(t_sim, U, 'k-', 'LineWidth', 2);
xlabel('Time $t$'); ylabel('Control input $u(t)$');
title('Finite-Horizon LQR Control Input');
grid on;
