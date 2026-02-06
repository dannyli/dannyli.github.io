% Continuous-time Infinite-Horizon Differential LQ Games Simulation
clear; clc;

% --------------------------------------
% 1. System definition (continuous-time)
% --------------------------------------

A  = [0 1; -1 -0.5];

B1 = [0; 1];   % controller 1
B2 = [0; 1];   % controller 2

dt = 0.01;
T  = 10;
N  = T/dt;

x0 = [1; 0];   % initial condition


% --------------------------------------
% 2. Cooperative games
% --------------------------------------
R1 = 1;
R2 = 1;

% Cooperative Riccati iteration
Q  = eye(2);
P = eye(2);
alpha = 0.001;

for k = 1:4000
    G = B1*(1/R1)*B1' + B2*(1/R2)*B2';
    Pdot = -(A'*P + P*A - P*G*P + Q);
    P = P - alpha*Pdot;
end

%[P, L, G] = care(A, [B1 B2], Q, diag([R1 R2]))

K1c = (1/R1)*B1'*P;
K2c = (1/R2)*B2'*P;

% Closed-loop simulation
x = x0;
Xc = zeros(2,N);

for k = 1:N
    u1 = -K1c*x;
    u2 = -K2c*x;
    x  = x + dt*(A*x + B1*u1 + B2*u2);
    Xc(:,k) = x;
end

% --------------------------------------
% 3. Non-cooperative games
% --------------------------------------
% Q1 = eye(2);
% Q2 = eye(2);

Q1 = diag([20, 1]);
Q2 = diag([1, 20]);
R1 = 1;
R2 = 1;

P1 = eye(2);
P2 = eye(2);
alpha = 0.001;

% Solve P1 and P2
for k = 1:4000
    K1 = (1/R1)*B1'*P1;
    K2 = (1/R2)*B2'*P2;

    Acl = A - B1*K1 - B2*K2;

    P1dot = -(Acl'*P1 + P1*Acl + Q1 + K1'*R1*K1);
    P2dot = -(Acl'*P2 + P2*Acl + Q2 + K2'*R2*K2);

    P1 = P1 - alpha*P1dot;
    P2 = P2 - alpha*P2dot;
end

K1n = (1/R1)*B1'*P1;
K2n = (1/R2)*B2'*P2;

% Closed-loop simulation
x = x0;
Xn = zeros(2,N);

for k = 1:N
    u1 = -K1n*x;
    u2 = -K2n*x;
    x  = x + dt*(A*x + B1*u1 + B2*u2);
    Xn(:,k) = x;
end


% --------------------------------------
% 4. Plot results
% --------------------------------------
t = (0:N-1)*dt;

figure;
subplot(2,1,1)
plot(t,Xc(1,:), 'b', t,Xn(1,:), 'r--','LineWidth',1.5)
xlabel('Time'); ylabel('$x_1$')
legend('Cooperative','Non-cooperative')
title('State $x_1$')

subplot(2,1,2)
plot(t,Xc(2,:), 'b', t,Xn(2,:), 'r--','LineWidth',1.5)
xlabel('Time'); ylabel('$x_2$')
legend('Cooperative','Non-cooperative')
title('State $x_2$')

