function out = seg_unique(in)

out = in(1);
if length(in) == 1
    return;
end

for i=2:length(in)
    if in(i) ~= out(end)
        out = [out, in(i)];
    end
end
