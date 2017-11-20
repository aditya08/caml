function [ results ] = casvm_cd(A, b, lambda,s, maxit, tol, seed, freq, Atest, btest, ref)
%UNTITLED Summary of this function goes here
%   Soft-margins dual SVM using coordinate descent, based on the algorithm by 
%   Cho-Jui Hsieh, et. al. 
tstart = tic;
rng(seed);
[m n] = size(A');
alpha = zeros(m,1);
w = zeros(n,1);

results.x = w;
results.alpha = alpha;


f_c_print = w'*w/2;
tmp = A'*w;
for j = 1:m
    f_c_print = f_c_print + lambda*max(1-b(j)*tmp(j), 0);   
end
results.obj(1) = f_c_print;

iter = 1;
i = zeros(1, s);
del_alp = zeros(1,s);
r = zeros(1,s);
tb = zeros(1,s);
talp = zeros(1,s);
theta = zeros(1,s);
results.grad(1) = 0;
results.grad2(1) = 0;
 results.truegrad(1) = 0;
results.del_alp(1) = 0;

tdiag = zeros(1,s);
tstep = zeros(1,s-1);
tG = zeros(1,s);
G = zeros(s,s);
dual = -Inf;
primal = Inf;

correct = 0;
tmp = Atest*w;

for k = 1:length(btest)
    if(tmp(k) > 0 && btest(k) == 1)
        correct = correct + 1;
    elseif(tmp(k) <= 0 && btest(k) == -1)
        correct = correct + 1;
    end
end
results.acc(1) = (correct/length(btest))*100;

    while(iter <= maxit)
        for j=1:s
            if(mod(iter + j - 2, m) == 0)
                perm = randperm(m);
                %perm = randi(m,1,m);
                nexti = 1;
            end
            %i(j) = mod(iter+j-2,m) + 1;
            i(j) = perm(nexti);
            nexti = nexti + 1;
            tb(j) = b(i(j));
            talp(j) = alpha(i(j));
        end
        tA = A(:, i);
        %tS = sparse(i, 1:s, ones(1,s));
        %Gs = tS'*tS - speye(100);
        G = tA'*tA;
        r = w'*tA;
        tdiag = 1./diag(G);
        tstep = zeros(1,s);
        theta = zeros(1,s);
        %tb = b(i);
        %talp = alpha(i);
        for j = 1:s
            tbb = tb(j);
            %tAA = tA(:,j);
            grad = tbb*r(j) - 1;
            gradcorr = 0;
            %gradcorr3 = 0;
            %for k = 1:j-1
            %    gradcorr = gradcorr + G(j,k)*tstep(k);
                %grad = grad + tbb*G(j,k)*del_alp(k)*tb(k);
            %end
            %if(j > 1)
            %    gradcorr = gradcorr + G(j, 1:j-1)*tstep(1:j-1)';
            %end
            %talpup = talp(j);
            if(j > 1)
                gradcorr = gradcorr + tstep*G(:, j);
                %talpup = talpup + theta*Gs(:,j);
            end
            
            grad = grad + tbb*gradcorr;
            %for large s, this is more efficient
            %grad2 = tbb*r(j) - 1+ tbb*sum(G(j,1:j-1).*del_alp(1:j-1).*tb(1:j-1));
            %grad + tbb*sum(G(j,1:j-1).*del_alp(1:j-1).*tb(1:j-1)');
            
            %if(j > 1)
            %    grad = grad + tbb*sum(G(j,1:j-1).*del_alp(1:j-1).*tb(1:j-1)');
            %end
            %results.grad(end+1) = grad;
            %results.grad2(end+1) = grad2;

            talpup = talp(j);
            for k=1:j-1
                if(i(j) == i(k))
                    talpup = talpup + theta(k);
                end
            end
            
            proj_grad = abs(min(max( talpup - grad, 0.0), lambda) - talpup);
            %tdiag = 1/G(j,j);
            if(proj_grad ~= 0)
                ttheta = min(max(talpup - grad*tdiag(j), 0.0), lambda) - talpup;
            else
                ttheta = 0;
            end

                del_alp(j) = talpup;
                alpha(i(j)) = alpha(i(j)) + ttheta;
                del_alp(j) = alpha(i(j)) - del_alp(j);
                theta(j) = ttheta;
                
                tstep(j) = del_alp(j)*tbb;
                %results.del_alp(end + 1) = del_alp(j);
    %             
    %             talp = alpha(i);
    %             alpha(i) = min(max(alpha(i) - grad/eta, 0.0),lambda);
                %results.truegrad(end+1) = tbb*w'*tAA - 1;

                %results.truegrad(end) - ref.grad(iter+1)
                %w = w + del_alp(j)*tbb*tAA;
                
                
            if(mod(iter, freq) == 0)
                tw = w + tA(:,1:j)*tstep(1:j)';
                
                f_c_print = tw'*tw;
                for k = 1:m
                    f_c_print = f_c_print + (-2*alpha(k));
                end
                
                results.x(:,end+1) = tw;
                results.alpha(:,end+1) = alpha;
                
                f_c_print = f_c_print/2;
                fprintf('D(x) %e\t', f_c_print);
                results.obj(end + 1) = f_c_print;
                dual = f_c_print;
                tmp = A'*tw;
                f_c_print = 0;
                for k = 1:m
                    d = (1 - tmp(k)*b(k));
                    if(d > 0)
                        f_c_print = f_c_print + d;
                    end
                end
                f_c_print = f_c_print + (tw'*tw/2);
                fprintf('P(x) %e\t', f_c_print);
                primal = f_c_print;

                correct = 0;
                tmp = Atest*tw;

                for k = 1:length(btest)
                    if(tmp(k) > 0 && btest(k) == 1)
                        correct = correct + 1;
                    elseif(tmp(k) <= 0 && btest(k) == -1)
                        correct = correct + 1;
                    end
                end
                fprintf('Gap %e\t', primal+dual);
                results.gap(iter/freq+1) = primal + dual;
                results.acc(iter/freq+1) = (correct/length(btest))*100;

                fprintf('accuracy %.4f\t', (correct/length(btest))*100);
                fprintf('elapsed time %.4f\n', toc(tstart));
                %fprintf('Cond(G) %.4f\n', condest(G));

            end

            iter = iter + 1;
            
            if(iter > maxit || abs(dual + primal) <= tol)
                tw = w + tA(:,1:j)*tstep(1:j)';

                results.w = tw;
                results.alpha = alpha;
                results.time = toc(tstart);
                return;
            end
            
        end
        w = w + tA*(tstep)';
        %fprintf('w[1] = %f\n', w(1))
        %w = w + del_alp*tb*tA;
        
        %i = zeros(1, s);
        %del_alp = zeros(1,s);
        %theta = zeros(1,s);
    end
    
    results.w = w;
    results.x(:,end+1) = w;
    results.alpha(:,end+1) = alpha;
    results.time = toc(tstart);


end

