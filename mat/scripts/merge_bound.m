function [] = merge_bound(feat_type,nHMM, nState)

zrst_path = '../pattern/';
%feat_type = 'pattern_type';
%nHMM = [50, 100, 300, 500];
%nState = 3:2:7;

output = '../exp/merge/boundary.txt';

pwr = 1;
win = [-1, 0, 1];
mask = [.5, 1, .5];
thres = 1.4;

H = length(nHMM);
S = length(nState);

fins = cell(H, S);
for h=1:H
for s=1:S
    fins{h, s} = fopen(sprintf('%s%s_%d_%d/result/result.mlf', ...
                  zrst_path, feat_type, nHMM(h), nState(s)), 'r');

    line = fgets(fins{h, s}); % header
end
end

fout = fopen(output, 'w');
fprintf(fout, line); % header

for h=1:H, for s=1:S, line = fgets(fins{h, s}); end; end; % read filename
count = 0; fprintf('00000');
while line ~= -1
    boundary = zeros(H*S, 1500);
    fprintf(fout, line); % write filename

    for h=1:H
    for s=1:S
        line = fgets(fins{h, s}); % read first pattern
        while length(line) > 2
            [begin, final, ~, ~] = strread(line, '%d %d %s %f');

            if begin ~= final
                final = final/100000;
                boundary((s-1)*S+h, final+win) = (nState(s)^pwr)*mask;
            end            

            line = fgets(fins{h, s}); % read next pattern or '.'
        end
    end
    end

    boundary(:, final+1:end) = []; 
    Bndry = mean(boundary);
    Bndry = -1*DIFF(DIFF(Bndry));
    Bndry = myFilter(Bndry, thres);

    for b=2:length(Bndry)
        fprintf(fout, '%d %d # #\n', Bndry(b-1), Bndry(b));
    end
    fprintf(fout, '.\n');


    for h=1:H, for s=1:S, line = fgets(fins{h, s}); end; end; % read next filename

    count = count+1;
    fprintf('\b\b\b\b\b%05d', count);
end

fclose all;

end
