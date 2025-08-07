% Finite-Horizon Discrete-Time LQR Simulation
clear; clc;

% --------------------------------------
% 1. System definition
% --------------------------------------
A = [1 1; 0 1];       % State transition matrix
B = [0; 1];           % Input matrix

Q = eye(2);           % State cost
R = 1;                % Control cost
S = 10 * eye(2);      % Terminal state cost

N = 20;               % Time horizon
x0 = [2; 0];          % Initial state

% --------------------------------------
% 2. Backward Riccati recursion
% --------------------------------------
P = cell(N+1,1);      % Store P matrices
K = cell(N,1);        % Store feedback gains

P{N+1} = S;           % Terminal cost

for k = N:-1:1
    Pk1 = P{k+1};
    Kk = (R + B' * Pk1 * B) \ (B' * Pk1 * A);
    K{k} = Kk;
    P{k} = Q + A' * Pk1 * A - A' * Pk1 * B * Kk;
end

% --------------------------------------
% 3. Forward simulation with time-varying K_k
% --------------------------------------
X = zeros(2, N+1);
U = zeros(1, N);

x = x0;
X(:,1) = x;

for k = 1:N
    u = -K{k} * x;
    x = A * x + B * u;
    X(:,k+1) = x;
    U(k) = u;
end

% --------------------------------------
% 4. Plot results
% --------------------------------------
figure;

% State trajectories
subplot(2,1,1);
plot(0:N, X(1,:), 'r-s', 'LineWidth', 1); hold on;
plot(0:N, X(2,:), 'b-*', 'LineWidth', 1);
xlabel('Time step $k$'); ylabel('State');
legend('$x_1$','$x_2$');
title('Finite-Horizon LQR State Trajectory');
grid on;

% Control input
subplot(2,1,2);
stairs(0:N-1, U, 'k-', 'LineWidth', 2);
xlabel('Time step $k$'); ylabel('Control input $u_k$');
title('Finite-Horizon LQR Control Input');
grid on;
