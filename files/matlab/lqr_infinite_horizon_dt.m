% Discrete-Time Infinite-Horizon LQR Simulation
clear; clc;

% --------------------------------------
% 1. System definition (discrete-time)
% --------------------------------------
A = [1 1; 0 1];       % State transition matrix
B = [0; 1];           % Input matrix

Q = eye(2);           % State weighting matrix
R = 1;                % Control weighting scalar

% --------------------------------------
% 2. Solve Discrete-Time Algebraic Riccati Equation (DARE)
% --------------------------------------
[K, P, eigVals] = dlqr(A, B, Q, R);   % Compute LQR gain

disp('LQR gain K:');
disp(K);

% --------------------------------------
% 3. Simulation setup
% --------------------------------------
x = [2; 0];           % Initial state
N = 20;               % Number of time steps

X = zeros(2, N+1);    % State trajectory storage
U = zeros(1, N);      % Control input storage
X(:,1) = x;           % Set initial state

% --------------------------------------
% 4. Closed-loop simulation
% x_{k+1} = (A - B*K)*x_k
% --------------------------------------
for k = 1:N
    u = -K * x;                      % Compute control input
    x = A * x + B * u;               % Apply closed-loop dynamics
    X(:,k+1) = x;                    % Store new state
    U(k) = u;                        % Store control input
end

% --------------------------------------
% 5. Plot results
% --------------------------------------
figure;

% Plot state trajectories
subplot(2,1,1);
plot(0:N, X(1,:), 'r-s', 0:N, X(2,:), 'b-*');
legend('$x_1$','$x_2$');
xlabel('Time step $k$');
ylabel('State $x_k$');
title('LQR Closed-Loop State Trajectory');
grid on;

% Plot control input
subplot(2,1,2);
stairs(0:N-1, U, 'k-', 'LineWidth', 2);
xlabel('Time step $k$');
ylabel('Control input $u_k$');
title('LQR Control Input');
grid on;
