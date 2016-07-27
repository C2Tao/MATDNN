%
% author Yang tongfeng@shandong university
%
%
function [pz,pzd,pwz,pzdw]=yang_plsa(data,n_z)
%  data is the document-word co-occurence matrix. data(i,j), word j number in  document i 
%  n_z topic number
%n_d document number , n_w word number
[n_d n_w]=size(data);
% 随机初始化
pz=rand(1,n_z);  % p(z)
pzd=rand(n_z,n_d); % p(z|d) 
pwz=rand(n_w,n_z); % p(w|z)
pzdw=rand(n_z,n_d,n_w);        %initialize
%p(d_i)固定为  p(d_i) \propto \sum_j n(d_i,w_j) 
pd  = sum(data,2)/sum(sum(data));
%结束条件
cov = 1;
err=0.0001;
pz_old = zeros(n_z,1);
iter = 1;
while cov > err
    % E step
    % p(z_k|d_i,w_j) \propto  p(w_j|_k)p(z_k|d_i)
    for k=1:n_z
        for i = 1:n_d
            for j= 1:n_w
                pzdw(k,i,j) = pwz(j,k)*pzd(k,i);
            end
        end
    end
    % 正比算法 ： 对于一个文档中的一个词，求所有topic的概率的和 做分母
    fenmu = squeeze(sum(pzdw)); 
    % 标准化 
    for i = 1:n_d
        for j = 1:n_w
            pzdw(:,i,j) = pzdw(:,i,j)/fenmu(i,j);
        end
    end
    % M step  
    % p(w_j|z_k) \propto \sum_i n(d_i,w_j)p(z_k|d_i,w_j)
    pwz = zeros(n_w,n_z);
    for j = 1:n_w
        for k = 1:n_z
            for i=1:n_d
                pwz(j,k) = pwz(j,k) + data(i,j)*pzdw(k,i,j);
            end
        end
    end
    %求分母 , 对一个topic中所有词的概率做加和
    fenmu = squeeze(sum(pwz));
    for k=1:n_z
            pwz(:,k) = pwz(:,k)/fenmu(k);
    end
    % p(z|d)    p(z_k|d_i) \propto \sum_j n(d_i,w_j)p(z_k|d_i,w_j)
    pzd = zeros(n_z,n_d);
    for i = 1:n_d
        for k = 1:n_z
            for j = 1:n_w
                pzd(k,i) = pzd(k,i) + data(i,j)*pzdw(k,i,j);
            end
        end
    end
    %分母 n(d_i)
    fenmu = squeeze(sum(data,2)); % n(d_i)
    for i = 1:n_d
        pzd(:,i) = pzd(:,i)/fenmu(i);
    end
    %p(z_k) = \sum_i p(z_k|d_i)*p(d_i) 
    pz = pzd*pd;
    % 是否收敛 1/n |p(z)-p'(z)|
    cov = mean(abs(pz-pz_old));
    fprintf('iter %d, cov = %3.5f\n',iter,cov);
    %prepare for next iter
    pz_old = pz;
    iter = iter + 1;
end % end while
