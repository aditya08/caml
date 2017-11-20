function [ results ] = cabdcd_conv(X, y, lambda,s, b, maxit, tol, seed, freq, wopt, objopt)
%SDCA_RES Computes L2 regression using the residual form of SDCA
%   Detailed explanation goes here
[m , n] = size(X);
rng(seed);
alpha = zeros(n, 1);
w = zeros(m, 1);
res = zeros(s*b, 1);
del_alpha = zeros(b, 1);
gamma = zeros(1, 1);
I = speye(n,n);
dalsamp = zeros(1,s*b);
index = zeros(1,s*b);

alpha(:,1) = zeros(n,1);
%res(:,1) = (1/n)*y; 
w(:,1) = zeros(m,1);
results.dres(:,1) = norm(1/n*y);
results.tres(:,1) = norm(1/n*X*y);

cnt = 0;
out_nrm = zeros(n,1);



results.obj(1) = 1/(2*n)*norm(X'*w(:,1)-y)^2 + lambda/2*norm(w(:,1))^2;
results.rel_error(1) = norm(wopt - w)/norm(wopt);
results.gramcond(1) = 0;
results.iters = 0;
results.flops = 0;



Isb = eye(s*b);

iter = 1;
idd = 0;
while (iter <= maxit)

    for j = 1:s
        [~, index((j-1)*b + 1 : j*b)] = datasample(X,b,2,'Replace', false);
    end
    %index(1) = 15880;
    Y = X(:,index);
    G = ((1/(lambda*(n^2)))*(Y'*Y)) + ((1/n)*Isb);
    Yt = I(:,index);
    
    flops = ssmultsym(Y', Y);
    results.flops(end+1) = results.flops(end) + flops.flops  + b^3;

    tG = Yt'*Yt;
    %cond(full(G))
        
    tw = ((1/n)*Y'*w);
    %cond(full(G))
        
    for j = 1:s
        idxr = (j-1)*b + 1: j*b;
        idxc = 1:(j-1)*b;
        dres(idxr) = - (tw(idxr))... 
            + ((1/n)*alpha(index(idxr))) + ((1/n)*y(index(idxr)));
        dalp = (-G(idxr,idxc)*dalsamp(idxc)') + dres(idxr)';
        
        dalp = dalp - ((1/n)*tG(idxr,idxc)*dalsamp(idxc)');

        
%         for k = 1:j-1
%                 dalp = dalp...
%                     + (1/n)*I(:,index(idxr))'*I(:,index(idxr - k*b))*dalsamp(idxr-k*b)';
%         end
        
        dalsamp(idxr) = G(idxr,idxr)\dalp;
        
        if (mod(iter, freq) == 0 || iter == 1)
            
            wp = w;
            for k = 1:j
                idxr = (k-1)*b + 1: k*b;
                wp = wp + (1/(lambda*n))*X(:,index(idxr))*dalsamp(idxr)';
            end
            results.obj(end+1) = 1/(2*n)*norm(X'*wp-y)^2 + lambda/2*norm(wp)^2;
            results.gramcond(end+1) = cond(full(G));
            results.iters(end+1) = iter;
            %cond(full(G))
            %condest(G)
            results.rel_error(end+1) = norm(wopt - wp)/norm(wopt);
            if(iter == maxit)
                results.w = wp;
                %return;
            end
        end
        
        iter = iter + 1;
    end
    
%     alpha(index) = alpha(index) + dalsamp';

    w = w + (1/(lambda*n))*Y*dalsamp';

    for k = 1:s
        idxr = (k-1)*b + 1: k*b;
        %w = w - (1/(lambda*n))*X(:,index(idxr))*dalsamp(idxr)';
        alpha(index(idxr)) = alpha(index(idxr)) - dalsamp(idxr)';
    end
    
%     if(mod(iter, freq) == 0)
%     results.dres(:,end+1) = norm(1/n*X'*w - 1/n*alpha - 1/n*y);
%     results.dres(end)
%     results.tres(:,end+1) = norm(-lambda*w - 1/n*X*(X'*w) + 1/n*X*y);
%     end
%     
    out_nrm(index) = dres';
    
    %if(norm(out_nrm) <= tol)
        
        if(results.dres(end) <= tol)
            results.w = w;
            results.iter = iter;
            results.out_nrm = norm(out_nrm);
            
            %results.obj(end+1) = 1/(2*n)*norm(X'*w-y)^2 + lambda/2*norm(w)^2;
            %results.gramcond(end+1) = cond(full(G));
            %results.rel_error(end+1) = norm(wopt - w)/norm(wopt);
            %results.iters(end + 1) = iter;
            
            disp(strcat(string('At iteration: '), num2str(iter)));
            disp(strcat(string('Converged to solution with dual estimated residual: '),num2str(results.dres(end))));
            return;
            %out_nrm = zeros(m,1);
        end
    %end
    %if(mod(iter,5000) == 0)
     %   norm(out_nrm)
%     end
   % out_nrm = zeros(n,1);

    dalsamp = zeros(1,s*b);

end

results.iter = maxit;
results.w = w;
results.out_nrm = norm(out_nrm);
    
if(iter ~= maxit + 1)
    results.obj(end+1) = 1/(2*n)*norm(X'*w-y)^2 + lambda/2*norm(w)^2;
    results.gramcond(end+1) = cond(full(G));
    results.rel_error(end+1) = norm(wopt - w)/norm(wopt);
    results.iters(end+1) = maxit;
    %results.out_dalp(end+1) = norm(del_w);
end

disp('WARNING: CABDCD reached maxit without converging')

end

