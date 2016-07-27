function bndry = myFilter(bndry, thres)

N = length(bndry);
if ~any(bndry > 0)
    bndry = [0, N];
    return; 
end

THRES = thres;
candidate = false(1, N);
while ~any(candidate)
    thres = THRES*mean(bndry(bndry>0));
    candidate = (bndry > thres);
    THRES = THRES - 0.05;
end

for i=1:N
    if candidate(i)
        if i>1 && candidate(i-1) && bndry(i-1) > bndry(i)
            candidate(i) = 0;
        end
        if i<N && candidate(i+1) && bndry(i+1) > bndry(i)
            candidate(i) = 0;
        end
    end
end

for i=1:N-1
    if candidate(i)
        if candidate(i+1)
            w = 2;
            while i+w<=N && candidate(i+w)
                w=w+1;
            end

            for j=i:i+w-1
                candidate(i) = 0;
            end
            candidate(i+round((w-1)/2)) = 1;
        end
    end
end

bndry = find(candidate > 0);
if bndry(end) ~= N
    bndry(end+1) = N;
end
bndry = [0, bndry];

