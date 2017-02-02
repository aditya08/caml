%modify fname and libsvmread command as needed.

%each plot is automatically saved as .pdf and .fig.
%you may have to modify legend locations, if bad placement.

fname = 'a9a';
[y, X] = libsvmread(strcat('caml/libsvm_sdca/ca_sdca_code/data/', fname, '.txt'));

X = X';
[m, n] = size(X);



OPTS = optimset('TolFun', 1e-4);
lmeig = eigs(@(x) (X'*(X*x)), n, 1, 'LM', OPTS)
smeig = eigs(@(x) (X'*(X*x)), n, 1, 'SM', OPTS)
nnz(X)/(m*n)

lambda = 1e3*smeig;
seed = 100;

%modify maxit and freq, as needed. Might want to keep tol as is.

%maxit is the maximum number of iterations.
%freq is the frequency that rel_error and objective function are computed.

%Ensure that mod(maxit, freq) = 0.
%maxit/freq is approximately the number of points on the plots.

%Visually, 10 <= maxit/freq < 20, works best.


tol = 1e-16;
maxit = 20000;
freq = 2000;


mat = @(x) (X*(X'*x)).*(1/n) + lambda.*x;
rhs = (X*y)./n;
w   = pcg(mat,rhs,1.0e-15,1000);
disp('Opt Objective function')
objopt = 1/(2*n)*norm(X'*w-y)^2 + lambda/2*norm(w)^2;
disp(objopt)

tic;
%b = 1, is dual coordinate descent. Do not change.
b =1;
s = 1;
results = cabdcd_conv(X, y, lambda, s, b, maxit, tol, 1, freq, w);
toc;

pres = -lambda*results.w-((1/n)*X*(X'*results.w)) + ((1/n)*X*y);
norm(pres)
pause(2)

tic;
% change b1 as needed.
b1 = 16;
results_1 = cabdcd_conv(X, y, lambda, s, b1, maxit, tol, 1, freq, w);
toc;

pres_1 = -lambda*results_1.w-((1/n)*X*(X'*results_1.w)) + ((1/n)*X*y);
norm(pres_1)
pause(2)

tic;
% change b2 as needed.
b2 = 32;
results_2 = cabdcd_conv(X, y, lambda, s, b2, maxit, tol, 1, freq, w);
toc;

pres_2 = -lambda*results_2.w-((1/n)*X*(X'*results_2.w)) + ((1/n)*X*y);
norm(pres_2)
pause(2)

tic;
% change b3 as needed.
b3 = 32;
results_3 = cabdcd_conv(X, y, lambda, s, b3, maxit, tol, 1, freq, w);
toc;

pres_3 = -lambda*results_3.w-((1/n)*X*(X'*results_3.w)) + ((1/n)*X*y);
norm(pres_3)


%Ensure that lengths of measurement vectors are equal (for short vectors,
%replicate last entry until you reach desired length).

diff1 = length(results.rel_error) - length(results_1.rel_error) -1;
diff2 = length(results.rel_error) - length(results_2.rel_error) -1;
diff3 = length(results.rel_error) - length(results_3.rel_error) -1;

results_1.rel_error(end+1:end+diff1) = kron(results_1.rel_error(end), ones(diff1,1))';
results_2.rel_error(end+1:end+diff2) = kron(results_2.rel_error(end), ones(diff2,1))';
results_3.rel_error(end+1:end+diff3) = kron(results_3.rel_error(end), ones(diff3,1))';

results_1.obj(end+1:end+diff1) = kron(results_1.obj(end), ones(diff1,1))';
results_2.obj(end+1:end+diff2) = kron(results_2.obj(end), ones(diff2,1))';
results_3.obj(end+1:end+diff3) = kron(results_3.obj(end), ones(diff3,1))';

results_1.gramcond(end+1:end+diff1+1) = kron(results_1.gramcond(end), ones(diff1+1,1))';
results_2.gramcond(end+1:end+diff2+1) = kron(results_2.gramcond(end), ones(diff2+1,1))';
results_3.gramcond(end+1:end+diff3+1) = kron(results_3.gramcond(end), ones(diff3+1,1))';

results_1.iters(end+1:end+diff1) = results.iters(end-diff1+1:end);
results_2.iters(end+1:end+diff2) = results.iters(end-diff2+1:end);
results_3.iters(end+1:end+diff3) = results.iters(end-diff3+1:end);


figure;
hold on;
grid on;
% 
plot(results.iters, log10(results.rel_error), '-k', 'LineWidth', 2, 'MarkerSize', 10)
plot(results_1.iters, log10(results_1.rel_error), 'o--k', 'LineWidth', 2, 'MarkerSize', 10)
plot(results_2.iters, log10(results_2.rel_error), '+--k', 'LineWidth', 2, 'MarkerSize', 10)
plot(results_3.iters, log10(results_3.rel_error), 'x--k', 'LineWidth', 2, 'MarkerSize', 10)
H = refline(0,log10(eps));
H.Color = 'k';
H.LineWidth = 2;
H.MarkerSize = 10;
H.LineStyle = ':';


ylabel('log10(relative solution error)', 'FontSize', 18);
xlabel('Iterations (H)', 'FontSize', 18);
leg = legend(strcat('BDCD b'' =  ', int2str(b)), strcat('BDCD b'' =  ', int2str(b1)),...
    strcat('BDCD b'' =  ', int2str(b2)), strcat('BDCD b'' = ', int2str(b3)), '\epsilon_{mach}');
set(leg, 'FontSize', 18);
set(gca, 'FontSize', 16);
ylim([-17 5]);
print(strcat(fname, '_solerr'),'-dpdf','-r300');
savefig(strcat(fname, '_solerr.fig'));


figure;
hold on;
grid on;
% 
plot(results.iters, log10((results.obj - objopt)/objopt  + eps), '-k', 'LineWidth', 2, 'MarkerSize', 10)
plot(results_1.iters, log10((results_1.obj - objopt)/objopt  + eps), 'o--k', 'LineWidth', 2, 'MarkerSize', 10)
plot(results_2.iters, log10((results_2.obj - objopt)/objopt  + eps), '+--k', 'LineWidth', 2, 'MarkerSize', 10)
plot(results_3.iters, log10((results_3.obj - objopt)/objopt  + eps), 'x--k', 'LineWidth', 2, 'MarkerSize', 10)
H = refline(0,log10(eps));
H.Color = 'k';
H.LineWidth = 2;
H.MarkerSize = 10;
H.LineStyle = ':';


ylabel('log10(relative objective error)', 'FontSize', 18);
xlabel('Iterations (H)', 'FontSize', 18);
leg = legend(strcat('BDCD b'' = ', int2str(b)), strcat('BDCD b'' = ', int2str(b1)),...
    strcat('BDCD b'' = ', int2str(b2)), strcat('BDCD b'' = ', int2str(b3)), '\epsilon_{mach}');
set(leg, 'FontSize', 18);
set(gca, 'FontSize', 16);
ylim([-17 5]);
print(strcat(fname, '_objerr'),'-dpdf','-r300');
savefig(strcat(fname, '_objerr.fig'));

figure;
hold on;
grid on;
% 
plot(log10(results.iters*m + 1), log10((results.obj - objopt)/objopt  + eps), '-k', 'LineWidth', 2, 'MarkerSize', 10)
plot(log10(results_1.iters*(b1^2*m + b1^3)+1), log10((results_1.obj - objopt)/objopt  + eps), 'o--k', 'LineWidth', 2, 'MarkerSize', 10)
plot(log10(results_2.iters*(b2^2*m + b2^3)+1), log10((results_2.obj - objopt)/objopt  + eps), '+--k', 'LineWidth', 2, 'MarkerSize', 10)
plot(log10(results_3.iters*(b3^2*m + b3^3)+1), log10((results_3.obj - objopt)/objopt  + eps), 'x--k', 'LineWidth', 2, 'MarkerSize', 10)
H = refline(0,log10(eps));
H.Color = 'k';
H.LineWidth = 2;
H.MarkerSize = 10;
H.LineStyle = ':';


ylabel('log10(relative objective error)', 'FontSize', 18);
xlabel('log10(Flops (F))', 'FontSize', 18);
leg = legend(strcat('BDCD b'' = ', int2str(b)), strcat('BDCD b'' = ', int2str(b1)),...
    strcat('BDCD b'' = ', int2str(b2)), strcat('BDCD b'' = ', int2str(b3)), '\epsilon_{mach}');
set(leg, 'FontSize', 18);
set(gca, 'FontSize', 16);
ylim([-17 5]);
print(strcat(fname, '_flops'),'-dpdf','-r300');
savefig(strcat(fname, '_flops.fig'));


figure;
hold on;
grid on;
% 
plot(log10(results.iters*1 + 1), log10((results.obj - objopt)/objopt  + eps), '-k', 'LineWidth', 2, 'MarkerSize', 10)
plot(log10(results_1.iters*(b1^2)+1), log10((results_1.obj - objopt)/objopt  + eps), 'o--k', 'LineWidth', 2, 'MarkerSize', 10)
plot(log10(results_2.iters*(b2^2)+1), log10((results_2.obj - objopt)/objopt  + eps), '+--k', 'LineWidth', 2, 'MarkerSize', 10)
plot(log10(results_3.iters*(b3^2)+1), log10((results_3.obj - objopt)/objopt  + eps), 'x--k', 'LineWidth', 2, 'MarkerSize', 10)
H = refline(0,log10(eps));
H.Color = 'k';
H.LineWidth = 2;
H.MarkerSize = 10;
H.LineStyle = ':';


ylabel('log10(relative objective error)', 'FontSize', 18);
xlabel('log10(Bandwidth (W))', 'FontSize', 18);
leg = legend(strcat('BDCD b'' = ', int2str(b)), strcat('BDCD b'' = ', int2str(b1)),...
    strcat('BDCD b'' = ', int2str(b2)), strcat('BDCD b'' = ', int2str(b3)), '\epsilon_{mach}');
set(leg, 'FontSize', 18);
set(gca, 'FontSize', 16);
ylim([-17 5]);
print(strcat(fname, '_bw'),'-dpdf','-r300');
savefig(strcat(fname, '_bw.fig'));


figure;
hold on;
grid on;
% 
plot(log10(results.iters + 1), log10((results.obj - objopt)/objopt  + eps), '-k', 'LineWidth', 2, 'MarkerSize', 10)
plot(log10(results_1.iters+1), log10((results_1.obj - objopt)/objopt  + eps), 'o--k', 'LineWidth', 2, 'MarkerSize', 10)
plot(log10(results_2.iters+1), log10((results_2.obj - objopt)/objopt  + eps), '+--k', 'LineWidth', 2, 'MarkerSize', 10)
plot(log10(results_3.iters+1), log10((results_3.obj - objopt)/objopt  + eps), 'x--k', 'LineWidth', 2, 'MarkerSize', 10)
H = refline(0,log10(eps));
H.Color = 'k';
H.LineWidth = 2;
H.MarkerSize = 10;
H.LineStyle = ':';


ylabel('log10(relative objective error)', 'FontSize', 18);
xlabel('log10( Number of Messages (L))', 'FontSize', 18);
leg = legend(strcat('BDCD b'' = ', int2str(b)), strcat('BDCD b'' = ', int2str(b1)),...
    strcat('BDCD b'' = ', int2str(b2)), strcat('BDCD b'' = ', int2str(b3)), '\epsilon_{mach}');
set(leg, 'FontSize', 18);
set(gca, 'FontSize', 16);
ylim([-17 5]);
print(strcat(fname, '_msg'),'-dpdf','-r300');
savefig(strcat(fname, '_msg.fig'));