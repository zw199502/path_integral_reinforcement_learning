

N = 450;

k_all = importdata('K_all.txt');
x1 = 0;
y1 = 0;
theta1 = 0;

x2 = -1.5;
y2 = -0.6;
theta2 = -55.0 / 180.0 * pi;



x_target = zeros(N + 1);
y_target = zeros(N + 1);
theta_target = zeros(N + 1);



x = zeros(N + 1);
y = zeros(N + 1);
theta = zeros(N + 1);



x_t = x2;
y_t = y2;
theta_t = theta2;

x_target(1) = x2;
y_target(1) = 0;

x(1) = x2;
y(1) = y2;
theta(1) = theta2;

index_d = 0;
index_phi = 0;

K1 = 75.3;
K2 = 14.8;

u = 0.22;
u_real = zeros(N);
d_real = zeros(N);
phi_real = zeros(N);
run_time = zeros(N + 1);
r_last = 0;
max_r = 1.58;
max_r_delta = 0.5;

dt = 0.035;

K1_ref = 50.0;
K2_ref = 20.20;
d_ref = 0;
phi_ref = 0;
r_ref = 0;
r_last_ref = 0;
x_ref = zeros(N + 1);
y_ref = zeros(N + 1);
theta_ref = zeros(N + 1);
x_ref(1) = x2;
y_ref(1) = y2;
theta_ref(1) = theta2;
x_t_ref = x2;
y_t_ref = y2;
theta_t_ref = theta2;

K1_ref2 = 98.253;
K2_ref2 = 14.447;
d_ref2 = 0;
phi_ref2 = 0;
r_ref2 = 0;
r_last_ref2 = 0;
x_ref2 = zeros(N + 1);
y_ref2 = zeros(N + 1);
theta_ref2 = zeros(N + 1);
x_ref2(1) = x2;
y_ref2(1) = y2;
theta_ref2(1) = theta2;
x_t_ref2 = x2;
y_t_ref2 = y2;
theta_t_ref2 = theta2;
tic
for tt = 1:1:N

    d = y_t;
    phi = theta_t;

    if d < -0.795
        index_d = 1;
    end
    if d > 0.795
        index_d = 161;
    end
    if phi < -82.5 / 180.0 * pi
        index_phi = 1;
    end
    if phi > 82.5 / 180.0 * pi
        index_phi = 35;
    end
    if d >= -0.795 && d <= 0.795
        for index=1:1:159
            if d >= -0.795 + (index-1) * 0.01 && d < -0.795 + (index) * 0.01    
                index_d = index + 1;
                break;
            end
        end
    end
    if (phi >= -82.5 / 180.0 * pi) && (phi <= 82.5 / 180.0 * pi)
        for index=1:1:33
            if (phi >= -82.5 / 180.0 * pi + (index-1) * 5 / 180.0 * pi) && (phi < -82.5 / 180.0 * pi + (index) * 5 / 180.0 * pi)
                index_phi = index + 1;
                break;
            end
        end
    end
    if mod(tt, 10) == 0
        K1 = k_all(index_d * 35 + index_phi, 1);
        K2 = k_all(index_d * 35 + index_phi, 2);
    end
    d_real(tt) = d;
    phi_real(tt) = phi;

    % temp_u = u + np.random.uniform(-0.2, 0.2, 1)
    temp_u = u;
    r = -K1 * temp_u * d * sin(phi) / phi - K2 * temp_u * phi;
    if r > max_r
        r = max_r;
    end
    if r - r_last > max_r_delta
        r = r_last + max_r_delta;
    end
    if r < -max_r
        r = -max_r;
    end
    if r - r_last < -max_r_delta
        r = r_last - max_r_delta;
    end
    r_last = r;
    x_t = x_t + temp_u * cos(theta_t) * dt;
    y_t = y_t + temp_u * sin(theta_t) * dt;
    theta_t = theta_t + r * dt;

    x(1 + tt) = x_t;
    y(1 + tt) = y_t;
    theta(1 + tt) = theta_t;

%     d_ref = y_t_ref;
%     phi_ref = theta_t_ref;
% 
%     % temp_u = u + random.uniform(-0.2, 0.2, 1)
%     temp_u = u;
%     r_ref = -K1_ref * temp_u * d_ref * sin(phi_ref) / phi_ref - K2_ref * temp_u * phi_ref;
%     if r_ref > max_r
%         r_ref = max_r;
%     end
%     if r_ref - r_last > max_r_delta
%         r_ref = r_last + max_r_delta;
%     end
%     if r_ref < -max_r
%         r_ref = -max_r;
%     end
%     if r_ref - r_last < -max_r_delta
%         r_ref = r_last - max_r_delta;
%     end
%     r_last_ref = r_ref;
% 
% 
%     x_t_ref = x_t_ref + temp_u * cos(theta_t_ref) * dt;
%     y_t_ref = y_t_ref + temp_u * sin(theta_t_ref) * dt;
%     theta_t_ref = theta_t_ref + r_ref * dt;
% 
%     x_ref(1 + tt) = x_t_ref;
%     y_ref(1 + tt) = y_t_ref;
%     theta_ref(1 + tt) = theta_t_ref;
% 
%     d_ref2 = y_t_ref2;
%     phi_ref2 = theta_t_ref2;
% 
%     % temp_u = u + np.random.uniform(-0.2, 0.2, 1)
%     temp_u = u;
%     r_ref2 = -K1_ref2 * temp_u * d_ref2 * sin(phi_ref2) / phi_ref2 - K2_ref2 * temp_u * phi_ref2;
%     if r_ref2 > max_r
%         r_ref2 = max_r;
%     end
%     if r_ref2 - r_last > max_r_delta
%         r_ref2 = r_last + max_r_delta;
%     end
%     if r_ref2 < -max_r
%         r_ref2 = -max_r;
%     end
%     if r_ref2 - r_last < -max_r_delta
%         r_ref2 = r_last - max_r_delta;
%     end
%     r_last_ref2 = r_ref2;
% 
%     x_t_ref2 = x_t_ref2 + temp_u * cos(theta_t_ref2) * dt;
%     y_t_ref2 = y_t_ref2 + temp_u * sin(theta_t_ref2) * dt;
%     theta_t_ref2 = theta_t_ref2 + r_ref2 * dt;
% 
%     x_ref2(1 + tt) = x_t_ref2;
%     y_ref2(1 + tt) = y_t_ref2;
%     theta_ref2(1 + tt) = theta_t_ref2;

%     x_target(1 + tt) = x_t;
%     y_target(1 + tt) = 0;

    run_time(tt + 1) = dt * (tt + 1);
end
 toc
 plot(run_time,y)
