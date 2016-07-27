function out = word_hash(in, maxNum)

a = 97;
n = 26;

len = 0;
while n^len < maxNum
    len = len+1;
end

out = zeros(1, len);
for i=len:-1:1
    out(i) = mod(in, n) + a;
    in = (in - mod(in, n))/n;
end

out = char(out);
