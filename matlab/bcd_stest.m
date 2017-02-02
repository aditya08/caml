%modify fname and libsvmread command as needed.

%each plot is automatically saved as .pdf and .fig.
%you may have to modify legend locations, if bad placement.



fname = 'abalone';
[y, X] = libsvmread(strcat('caml/libsvm_sdca/ca_sdca_code/data/', fname, '.txt'));
%  X = sprand(7000, 200, .05);
%  y = rand(7000,1);
X = X';
[m, n] = size(X);



OPTS = optimset('TolFun', 1e-4);
lmeig = eigs(@(x) (X'*(X*x)), n, 1, 'LM', OPTS)
smeig = eigs(@(x) (X'*(X*x)), n, 1, 'SM', OPTS)
nnz(X)/(m*n)

lambda = 1e3*smeig;
seed = 100;
maxit = 500;
freq = 50;

tol = 1e-16;

mat = @(x) (X*(X'*x)).*(1/n) + lambda.*x;
rhs = (X*y)./n;
w   = pcg(mat,rhs,1.0e-15,1000);
disp('Opt Objective function')
objopt = 1/(2*n)*norm(X'*w-y)^2 + lambda/2*norm(w)^2;
disp(objopt)

tic;
b =4;
s = 1;
results = cabcd_conv(X', y, lambda, s, b, maxit, tol, 1, freq, w);
toc;

pres = -lambda*results.w-((1/n)*X*(X'*results.w)) + ((1/n)*X*y);
norm(pres)
pause(2)

tic;
s1 = 10;
results_1 = cabcd_conv(X', y, lambda, s1, b, maxit, tol, 1, freq, w);
toc;

pres_1 = -lambda*results_1.w-((1/n)*X*(X'*results_1.w)) + ((1/n)*X*y);
norm(pres_1)
pause(2)

tic;
s2 = 100;
results_2 = cabcd_conv(X', y, lambda, s2, b, maxit, tol, 1, freq, w);
toc;

pres_2 = -lambda*results_2.w-((1/n)*X*(X'*results_2.w)) + ((1/n)*X*y);
norm(pres_2)
pause(2)

tic;
s3 = 200;
results_3 = cabcd_conv(X', y, lambda, s3, b, maxit, tol, 1, freq, w);
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
leg = legend(strcat('BCD s =  ', int2str(s)), strcat('BCD s =  ', int2str(s1)),...
    strcat('BCD s =  ', int2str(s2)), strcat('BCD s= ', int2str(s3)), '\epsilon_{mach}');
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
leg = legend(strcat('BCD s = ', int2str(s)), strcat('BCD s = ', int2str(s1)),...
    strcat('BCD s = ', int2str(s2)), strcat('BCD s = ', int2str(s3)), '\epsilon_{mach}');
set(leg, 'FontSize', 18);
set(gca, 'FontSize', 16);
ylim([-17 5]);
print(strcat(fname, '_objerr'),'-dpdf','-r300');
savefig(strcat(fname, '_objerr.fig'));


figure;
hold on;
grid on;
% 
plot(log10(results.iters*(s*b^2*m + b^3) + 1), log10((results.obj - objopt)/objopt  + eps), '-k', 'LineWidth', 2, 'MarkerSize', 10)
plot(log10(results_1.iters*(s1*b^2*m + b^3)+1), log10((results_1.obj - objopt)/objopt  + eps), 'o--k', 'LineWidth', 2, 'MarkerSize', 10)
plot(log10(results_2.iters*(s2*b^2*m + b^3)+1), log10((results_2.obj - objopt)/objopt  + eps), '+--k', 'LineWidth', 2, 'MarkerSize', 10)
plot(log10(results_3.iters*(s3*b^2*m + b^3)+1), log10((results_3.obj - objopt)/objopt  + eps), 'x--k', 'LineWidth', 2, 'MarkerSize', 10)
H = refline(0,log10(eps));
H.Color = 'k';
H.LineWidth = 2;
H.MarkerSize = 10;
H.LineStyle = ':';

ylabel('log10(relative objective error)', 'FontSize', 18);
xlabel('log10(Flops (F))', 'FontSize', 18);
leg = legend(strcat('BCD s = ', int2str(s)), strcat('BCD s = ', int2str(s1)),...
    strcat('BCD s = ', int2str(s2)), strcat('BCD s = ', int2str(s3)), '\epsilon_{mach}');
set(leg, 'FontSize', 18);
set(gca, 'FontSize', 16);
ylim([-17 5]);
print(strcat(fname, '_flops'),'-dpdf','-r300');
savefig(strcat(fname, '_flops.fig'));


figure;
hold on;
grid on;
% 
plot(log10(results.iters*(s*b^2) + 1), log10((results.obj - objopt)/objopt  + eps), '-k', 'LineWidth', 2, 'MarkerSize', 10)
plot(log10(results_1.iters*(s1*b^2)+1), log10((results_1.obj - objopt)/objopt  + eps), 'o--k', 'LineWidth', 2, 'MarkerSize', 10)
plot(log10(results_2.iters*(s2*b^2)+1), log10((results_2.obj - objopt)/objopt  + eps), '+--k', 'LineWidth', 2, 'MarkerSize', 10)
plot(log10(results_3.iters*(s3*b^2)+1), log10((results_3.obj - objopt)/objopt  + eps), 'x--k', 'LineWidth', 2, 'MarkerSize', 10)
H = refline(0,log10(eps));
H.Color = 'k';
H.LineWidth = 2;
H.MarkerSize = 10;
H.LineStyle = ':';

ylabel('log10(relative objective error)', 'FontSize', 18);
xlabel('log10(Bandwidth (W))', 'FontSize', 18);
leg = legend(strcat('BCD s = ', int2str(s)), strcat('BCD s = ', int2str(s1)),...
    strcat('BCD s = ', int2str(s2)), strcat('BCD s = ', int2str(s3)), '\epsilon_{mach}');
set(leg, 'FontSize', 18);
set(gca, 'FontSize', 16);
ylim([-17 5]);
print(strcat(fname, '_bw'),'-dpdf','-r300');
savefig(strcat(fname, '_bw.fig'));


figure;
hold on;
grid on;
% 
plot(log10(results.iters/s + 1), log10((results.obj - objopt)/objopt  + eps), '-k', 'LineWidth', 2, 'MarkerSize', 10)
plot(log10(results_1.iters/s1+1), log10((results_1.obj - objopt)/objopt  + eps), 'o--k', 'LineWidth', 2, 'MarkerSize', 10)
plot(log10(results_2.iters/s2+1), log10((results_2.obj - objopt)/objopt  + eps), '+--k', 'LineWidth', 2, 'MarkerSize', 10)
plot(log10(results_3.iters/s3+1), log10((results_3.obj - objopt)/objopt  + eps), 'x--k', 'LineWidth', 2, 'MarkerSize', 10)
H = refline(0,log10(eps));
H.Color = 'k';
H.LineWidth = 2;
H.MarkerSize = 10;
H.LineStyle = ':';



ylabel('log10(relative objective error)', 'FontSize', 18);
xlabel('log10( Number of Messages (L))', 'FontSize', 18);
leg = legend(strcat('BCD s = ', int2str(s)), strcat('BCD s = ', int2str(s1)),...
    strcat('BCD s = ', int2str(s2)), strcat('BCD s = ', int2str(s3)), '\epsilon_{mach}');
set(leg, 'FontSize', 18);
set(gca, 'FontSize', 16);
ylim([-17 5]);
print(strcat(fname, '_msg'),'-dpdf','-r300');
savefig(strcat(fname, '_msg.fig'));


figure;
hold on;
grid on;

h = boxplot(log10([results.gramcond(2:end)', results_1.gramcond(2:end)', results_2.gramcond(2:end)', results_3.gramcond(2:end)']),'Label',... 
    { strcat('BCD s = ', int2str(b)),...
    strcat('BCD s = ', int2str(s1)),...
    strcat('BCD s = ', int2str(s2)),...
    strcat('BCD s = ', int2str(s3))},...
    'Colors', 'k');

set(gca, 'FontSize', 18,'XTickLabelRotation',-10);
ylabel('log10(\kappa(G))', 'FontSize', 18);
set(h, 'LineWidth', .8)
h=findobj(gca,'tag','Outliers');
set(h,'MarkerEdgeColor','k');

print(strcat(fname, '_gram'),'-dpdf','-r300');
savefig(strcat(fname, '_gram.fig'));