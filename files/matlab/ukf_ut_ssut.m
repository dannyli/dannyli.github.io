clear all
close all


x = [1 -1]'; 
P = [.9 .4; .4 .3];
r = mvnrnd(x, P, 100);
% [sqrt_Sigma, p] = chol(Sigma,'lower'); % Cholesky decomposition, lower triangular matrix
[V,D] = eig(P);
sqrt_P = V*sqrtm(D);


alfa = 0.5;
kappa = 0;
beta = 2;

nx = 2;

lambda = alfa^2*(nx + kappa) - nx;           % scaling parameter
w_m = lambda/(nx + lambda);
w_c = lambda/(nx + lambda) + (1 - alfa^2 + beta);
for i = 2:(2*nx + 1)
    w_m(i) = 0.5/(nx+lambda);
    w_c(i) = w_m(i);
end

L = 5;

eta = sqrt(nx + lambda);
X_P1 =    eta*sqrt_P;
X_P2 =  - eta*sqrt_P;
Xi = repmat(x,[1 L]) + [zeros(nx,1) X_P1 X_P2];

x_prd       = Xi*w_m'; % weighted averaging
dXi         = Xi - repmat(x_prd,[1 L]);
P_prd       = dXi*diag(w_c)*dXi'; 

        W0 = 0.5;
        W = W0;
        for i = 2:(nx + 2)
            W(i) = (1-W0)/(nx + 1);
        end
        
        w(1) = 1 + (W0 - 1)/(alfa^2);
        for i = 2:(nx + 2)
            w(i) = W(i)/(alfa^2);
        end
        
        w_m = w(1);
        w_c = w(1) + (1 - alfa^2 + beta);
        for i = 2:(nx + 2)
            w_m(i) = w(i);
            w_c(i) = w(i);
        end
        
        W1 = W(2);
        w1 = w(2);

        Xi1 = [0 -1/sqrt(2*w1) 1/sqrt(2*w1)];

        for j = 2:nx
            for i = 0:j+1
                switch i
                    case 0
                        Xi_new(:,i+1) = [Xi1(:,i+1); 0 ];
                    case j+1
                        Xi_new(:,i+1) = [zeros(j-1,1); j/sqrt(j*(j+1)*w1) ];
                    otherwise
                        Xi_new(:,i+1) = [Xi1(:,i+1); -1/sqrt(j*(j+1)*w1) ];
                end
            end

            Xi1 = Xi_new;
            clear Xi_new

        end

    Xi1 = repmat(x,[1 nx + 2]) + sqrt_P*Xi1;




y = exp(r)
Yi = exp(Xi)
Yi1 = exp(Xi1)


figure(1)
subplot(1,2,1)
hold on
plot(r(:,1),r(:,2),'.');
plot(Xi(1,:),Xi(2,:),'r*')
plot(Xi1(1,:),Xi1(2,:),'bo')
xlabel('$x_1$')
ylabel('$x_2$')

subplot(1,2,2)
hold on
plot(y(:,1),y(:,2),'.');
plot(Yi(1,:),Yi(2,:),'r*')
plot(Yi1(1,:),Yi1(2,:),'bo')
xlabel('$\exp(x_1)$')
ylabel('$\exp(x_2)$')
legend({'Monte Carlo','UT','SSUT'},'box','on')
