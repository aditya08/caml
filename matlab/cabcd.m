 function [ results ] = cabcd_conv(X, y, lambda,s, b, maxit, tol, seed, freq, wopt)
%SDCA_RES Computes L2 regression using the residual form of SDCA
%   Detailed explanation goes here
[m , n] = size(X');
rng(seed);
alpha = zeros(n, 1);
w = zeros(m, 1);
res = zeros(s*b, 1);
del_w = zeros(b, 1);
gamma = zeros(1, 1);
I = speye(m,m);
dwsamp = zeros(1,s*b);
index = zeros(1,s*b);
idx = 0;
alpha(:,1) = zeros(n,1);
%res(:,1) = (1/n)*y; 
w(:,1) = zeros(m,1);

cnt = 0;
idd = 0;
out_nrm = zeros(m,1);

%results.out_dalp = zeros(1,maxit/freq+1);
%results.rel_error = zeros(1, maxit/freq+1);
%results.obj = zeros(1, maxit/freq+1);

results.obj(1) = (1/(2*n))*norm(X*w(:,1)-y)^2 + (lambda/2)*norm(w(:,1))^2;
results.rel_error(1) = norm(wopt - w)/norm(wopt);
results.gramcond(1) = 0;
results.iters = 0;
Isb = eye(s*b);

iter = 1;
outiter = 0;

true_res = 0;
comp_res = 0;

while (iter <= maxit)

    for j = 1:s
        [~, index((j-1)*b + 1 : j*b)] = datasample(X,b,2,'Replace', false);
    end
    Y = X(:,index);
    Yt = I(:,index);
    G = ((1/n)*(Y'*Y)) + (lambda*Isb);
    tG = Yt'*Yt;
      
    talpha = ((1/n)*Y'*alpha);
    ty = ((1/n)*Y'*y);
    for j = 1:s
        idxr = (j-1)*b + 1: j*b;
        idxc = 1:(j-1)*b;
        dres(idxr) = - (lambda*w(index(idxr)))... 
            - talpha(idxr) + ty(idxr);
        dw = (-G(idxr,idxc)*dwsamp(idxc)') + dres(idxr)';
            
        dw = dw - (lambda*tG(idxr,idxc)*dwsamp(idxc)');
        
        dwsamp(idxr) = G(idxr,idxr)\dw;
        
        if (mod(iter, freq) == 0 || iter == 1)
            
            wp = w;
            for k = 1:j
                idxr = (k-1)*b + 1: k*b;
                wp(index(idxr)) = wp(index(idxr)) + dwsamp(idxr)';
                del_w = norm(dwsamp(idxr));
            end
            results.obj(end+1) = 1/(2*n)*norm(X*wp-y)^2 + lambda/2*norm(wp)^2;
            results.gramcond(end+1) = cond(full(G));
            results.rel_error(end+1) = norm(wopt - wp)/norm(wopt);
            results.iters(end+1) = iter;
            
            if(iter == maxit)
                results.w = wp;
                return;
            end


        end
        
        iter = iter + 1;
    end
    
    alpha = alpha + Y*dwsamp';
    for k = 1:s
        idxr = (k-1)*b + 1: k*b;
        w(index(idxr)) = w(index(idxr)) + dwsamp(idxr)';
    
    end
        
    out_nrm(index) = dres';
        
    
    if(norm(out_nrm) <= tol)
        if(norm(-lambda*w-((1/n)*X'*(X*w)) + ((1/n)*X'*y)) <= tol)
            %out_nrm = zeros(m,1);
            results.w = w;
            results.out_nrm = norm(out_nrm);
            results.iter = iter;
            
            results.obj(end+1) = 1/(2*n)*norm(X*w-y)^2 + lambda/2*norm(w)^2;
            results.gramcond(end+1) = cond(full(G));
            results.rel_error(end+1) = norm(wopt - w)/norm(wopt);
            %results.out_dalp(end+1) = norm(del_w);
            results.iters(end + 1) = iter;
            disp(strcat(string('At iteration: '), num2str(iter)));
            disp(strcat(string('Converged to solution with estimated residual: '),num2str(norm(out_nrm))));

            return;
        end
    end
    
    if(mod(iter,freq/s -1 ) == 0)
            norm(out_nrm)
    end
    dwsamp = zeros(1,s*b);
end

results.iter = maxit;
results.w = w;
results.out_nrm = norm(out_nrm);
    
if(iter ~= maxit + 1)
    results.obj(end+1) = 1/(2*n)*norm(X*w-y)^2 + lambda/2*norm(w)^2;
    results.gramcond(end+1) = cond(full(G));
    results.rel_error(end+1) = norm(wopt - w)/norm(wopt);
    results.iters(end+1) = maxit;

    %results.out_dalp(end+1) = norm(del_w);
end
disp('WARNING: CABCD reached maxit without converging')

end