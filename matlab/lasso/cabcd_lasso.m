function [ results ] = cablock_lasso(A, b, lambda, tau, blocksize, s, L, vi, maxit, tol, seed, freq, opts, prevresults)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

%results_opt = block_lasso3(A, b, lambda, tau, blocksize, s, L, vi, maxit, tol, seed, freq, opts);

rng(seed);


[m, n] = size(A);
z = opts.z;
y = opts.y;
theta = zeros(s+1,1);
len = zeros(s+1,1);
c = zeros(s,1);
sample_block = zeros(s,1);
block_start = zeros(s,1);
block_end = zeros(s,1);

G = zeros(blocksize, ((s*(s-1))/2)*blocksize);
samp_r = zeros(s*blocksize, 1);

%fold = lambda*norm(x,1) + 0.5*norm(A*x-b)^2;

nblks = ceil(n/blocksize);
theta(1) = tau/nblks;
if(opts.theta == 1)
    theta(1) = 1;
end

c(1) = (1 - (nblks*theta(1))/tau)/theta(1)^2;
len(1) = 0;
iter = 0;


%lmax = eigs(A'*A, 1, 'lm');

ty = A*y;
tz = A*z - b;
r = -b;

delz = zeros(s*blocksize,1);

results.ty(:,1) = zeros(m,1);
results.tz(:,1) = zeros(m,1);
results.y(:,1) = zeros(n,1);
results.z(:,1) = zeros(n,1);
results.x(:,1) = zeros(n,1);
results.obj(:,1) = lambda*norm(theta(1)^2*y + z,1) + 0.5*norm(theta(1)^2*ty + tz)^2;


results.grad_fy = zeros(blocksize,1);
results.delz = zeros(blocksize, 1);
results.samp_r = zeros(blocksize,1);
results.sample = 0;
%results.step_size = 0;

results.theta = theta(1);
results.c = c(1);

    while(iter < maxit)
        
        G_start = 1;
        G_end = blocksize;
        %sample_block = [6 0 5 0 0 6 5 2 5 5 4 1 2 5 4 0];
        %sample_block = sample_block + ones(1,s);
        for i = 1:s
            sample_block(i) = randi(nblks);
            
            while(vi(sample_block(i)) == 0)
                sample_block(i) = randi(nblks);
            end
            
            %results.sample(end + 1) = sample_block(i);
            block_start(i) = (sample_block(i)-1)*blocksize + 1;
            block_end(i) = sample_block(i)*blocksize;
            
            if block_end(i) > n
                block_end(i) = n;
            end
            len(i) = block_end(i) - block_start(i) + 1;

            Y(:,(i-1)*blocksize + 1: (i)*blocksize) = [full(A(:, block_start(i):block_end(i))) zeros(m, blocksize - len(i))];
            
            samp_r((i-1)*blocksize + 1: i*blocksize) = Y(:,(i-1)*blocksize + 1: (i)*blocksize)'*(theta(i)^2*ty + tz);
            results.samp_r(:,end + 1) = samp_r((i-1)*blocksize + 1: i*blocksize);
            
            theta(i+1) = (sqrt(theta(i)^4 + 4*theta(i)^2) - theta(i)^2)/2;
            c(i+1) = (1 - (nblks*theta(i+1))/tau)/theta(i+1)^2;
                
            %results.theta(end + 1) = theta(i);
            %results.c(end + 1) = c(i);
            if i > 1
                %Compute the cross-terms of Gram matrix Y'*Y (don't need block
                %diagonals since lipschitz constants are pre-computed).
                G(:, G_start: G_end) = full(Y(:,(i-1)*blocksize + 1: i*blocksize)'*Y(:, 1:(i-1)*blocksize));
                G_start = G_end + 1;
                G_end = ((i+1)*i)/2*blocksize;
            end   

        end
        
        theta(end) = (sqrt(theta(end-1)^4 + 4*theta(end-1)^2) - theta(end-1)^2)/2;

                
        for i = 1:tau
            
            step_size = tau/nblks/theta(1)/vi(sample_block(1));
            %results.step_size(end+1) = step_size;
            %At the first iteration, gradient of f(y) is the first block of
            %samp_r. No computation needed.
            %results.grad_fy(:,end+1) = samp_r(1:blocksize);
            

            %results_opt.grad_fy(iter+2) - %results.grad_fy(iter + 2)
            
            grad_update = [z(block_start(1):block_end(1)); zeros(blocksize - len(1), 1)] - step_size*samp_r(1:blocksize);
            delz(1:blocksize) = sign(grad_update).*max(abs(grad_update) - lambda*step_size,0) - [z(block_start(1):block_end(1)); zeros(blocksize - len(1),1)];
            %delz(1:blocksize)
            %results.delz(:,end+1) = delz(1:blocksize);
            
            %disp(results_opt.step_size(iter+1) - %results.step_size(iter+1))

            
            G_start = 1;
            G_end = blocksize;
            %delz(1:blocksize)

            for j = 2:s
                step_size = tau/nblks/theta(j)/vi(sample_block(j));
                
                grad_fy = samp_r((j-1)*blocksize + 1: j*blocksize);% - ((c(j-1)*theta(j)^2 - 1)*G(:,G_start:G_end)*delz(1:G_end));
                
                %Need to ensure that we subtract off Gram matrix terms.
                
                for k = 1:j-1
                    offset = ((j-2)*(j-1))/2;
                    grad_fy = grad_fy - ((c(k)*theta(j)^2 - 1)*G(:,(k-1 + offset)*blocksize +1:(k+offset)*blocksize)*delz((k-1)*blocksize +1:k*blocksize));
                end
                
                grad_update = [z(block_start(j):block_end(j)); zeros(blocksize - len(j), 1)] - step_size*grad_fy;
                
                %results.grad_fy(:,end+1) = grad_fy;
                
                %If we picked some blocks multiple times in the s-steps,
                %then we have to update grad_update with the delz's from
                %previous iterations.
                
                for k = 1:j-1
                    if (sample_block(j) == sample_block(k))
                        grad_update = grad_update + delz((k-1)*blocksize + 1: k*blocksize);
                    end
                end
                
                delz((j-1)*blocksize + 1: j*blocksize) = sign(grad_update).*max(abs(grad_update) - lambda*step_size,0) - [z(block_start(j):block_end(j)); zeros(blocksize - len(j), 1)];
%                 results.delz(:,end+1) = delz((j-1)*blocksize + 1: j*blocksize);
                for k = 1:j-1
                    if (sample_block(j) == sample_block(k))
                        delz((j-1)*blocksize + 1: j*blocksize) = delz((j-1)*blocksize + 1: j*blocksize) - delz((k-1)*blocksize + 1: k*blocksize);
                    end
                end
                
                %delz((j-1)*blocksize + 1: j*blocksize)
                
            end
            
            %update y, ty, z, and tz.
           
            for j = 1:s
                
                tmp = [y(block_start(j):block_end(j)); zeros(blocksize - len(j), 1)] - c(j)*delz((j-1)*blocksize + 1: j*blocksize);
                y(block_start(j):block_end(j)) = tmp(1:block_end(j) - block_start(j) + 1);
                
                ty = ty - c(j)*Y(:,(j-1)*blocksize + 1: j*blocksize)*delz((j-1)*blocksize + 1: j*blocksize);
                
                tmp = [z(block_start(j):block_end(j)); zeros(blocksize - len(j), 1)] + delz((j-1)*blocksize + 1: j*blocksize);
                z(block_start(j):block_end(j)) = tmp(1:block_end(j) - block_start(j) + 1);
                
                tz = tz + Y(:,(j-1)*blocksize + 1: j*blocksize)*delz((j-1)*blocksize + 1: j*blocksize);
                
                f_c_print = lambda*norm(theta(j+1)^2*y + z,1) + 0.5*norm(theta(j+1)^2*ty + tz)^2;
                fprintf('F(x) %d \n', f_c_print);
                 results.obj(:,end+1) = f_c_print;
%                 results.ty(:,end+1) = ty;
%                 results.tz(:,end+1) = tz;
%                 results.y(:,end+1) = y;
%                 results.z(:,end+1) = z;
%                 results.x(:,end+1) = theta(j)^2*y + z;

                iter = iter + 1;
                
                %disp(norm(results_opt.y(:,iter + 1) - y))
            end
            
            delz = zeros(s*blocksize,1);
            samp_r = zeros(s*blocksize,1);
            G = zeros(blocksize, ((s*(s-1))/2)*blocksize);
            
            theta(1) = theta(end);
            c(1) = (1 - (nblks*theta(1))/tau)/theta(1)^2;
            
            %results.theta(end+1) = theta(1);
            %results.c(end+1) = c(1);
            

            
        end
    end
    
results.x = theta(1)^2*y + z;
%results.z = z;
%results.y = y;
results.iter = iter;


end

