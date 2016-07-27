Case = '3';
zrst_path = '../../zrstExps/';
feat_type = 'mfc';
nHMM = [50, 100, 300, 500];
nState = 3:2:9;
nTopic = 100;

bndry_path = ['../' Case '/boundary.txt'];
output = ['../' Case '/corpus.txt'];

H = length(nHMM);
S = length(nState);

vocSize = sum(nHMM)*S;
filenames = cell(5000, 1);
indices = cell(5000, 1);
timesteps = cell(5000, 1);
patterns = sparse(70000, vocSize);


fins = cell(H, S);
for h=1:H
for s=1:S
    fins{h, s} = fopen(sprintf('%s%s_%d_%d/result/result.mlf', ...
                  zrst_path, feat_type, nHMM(h), nState(s)), 'r');

    fgets(fins{h, s}); % header
end
end


bin = fopen(bndry_path, 'r');
header = fgets(bin); % header
count = 0;
t = 0;

fprintf('Reading...  0.00%%');
line = fgets(bin); % read filename
while line ~= -1
    count = count+1;
    filenames{count} = line;
    intervals = zeros(150, 2);
    c = 0;
    line = fgets(bin);
    while length(line) > 2
        c=c+1;
        [b, f, ~, ~] = strread(line, '%d %d %c %c');
        intervals(c, :) = [b+1, f];

        line = fgets(bin);
    end
    intervals(c+1:end, :) = [];
    timesteps{count} = intervals;
    indices{count} = t+1:t+c;


    for h=1:H
    for s=1:S
        frames = zeros(1, intervals(end));
        fgets(fins{h, s}); % read filename
        line = fgets(fins{h, s}); % read first pattern
        while length(line) > 2
            [begin, final, pat, ~] = strread(line, '%d %d %s %f');
            pat = pat{1};

            if begin ~= final && ~strcmp(pat, 'sp') && ~strcmp(pat, 'sil')
                begin = begin/100000 + 1;
                final = final/100000;
                frames(begin:final) = strread(pat(2:end), '%d');
            end            

            line = fgets(fins{h, s}); % read next pattern or '.'
        end

        if h==1
            bias = (s-1)*sum(nHMM);
        else
            bias = (s-1)*sum(nHMM) + sum(nHMM(1:h-1));
        end

        for i=1:c
            pats = seg_unique(frames(intervals(i, 1):intervals(i, 2)));
            for j=1:length(pats)
                if pats(j) ~= 0
                    patterns(t+i, bias+pats(j)) = patterns(t+i, bias+pats(j)) + 1;
                end
            end
        end
    end
    end

    fprintf('\b\b\b\b\b\b\b%6.2f%%', 100*count/4058);

    t = t+c;
    line = fgets(bin); % read next filename
end
fclose all;

filenames(count+1:end) = [];
indices(count+1:end) = [];
patterns(t+1:end, :) = [];


sils = find(sum(patterns, 2) == 0);
not_sils = find(sum(patterns, 2) > 0);
patterns = patterns(not_sils, :);

save(['../' Case '/data.mat']);

fprintf('Writing...  0.00%%');
fout = fopen(output, 'w');

for n=1:size(patterns, 1)
    fprintf(fout, 't%05d X', n);

    for i=find(patterns(n, :) > 0)
        for j=1:patterns(n, i)
            fprintf(fout, ' p%d', i);
        end
    end
    fprintf(fout, '\n');
    fprintf('\b\b\b\b\b\b\b%6.2f%%', 100*n/size(patterns, 1));
end

fclose all;
