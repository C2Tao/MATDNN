%{
zrst_path = '../../zrstExps/';
feat_type = 'mfc';
nHMM = [50, 100, 300, 500];
nState = 3:2:9;
nTopic = 100;

bndry_path = '../1/boundary.txt';
output = '../1/result.mlf';

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
%}

load('../1/data.mat');

sils = find(sum(patterns, 2) == 0);
not_sils = find(sum(patterns, 2) > 0);
patterns = patterns(not_sils, :);

%{
nTopic = 50;
fprintf('\nPLSA starts.\n');
[~, pzd] = yang_plsa(patterns, nTopic);
%}


nClass = 100;
patterns = bsxfun(@rdivide, patterns, sqrt(sum(patterns.^2, 2)));
opts = statset('MaxIter', 1000);
newCl = kmeans(patterns, nClass, 'Replicates', 2, 'Options', opts, 'EmptyAction', 'singleton');
newClasses = zeros(N+length(sils), 1);
newClasses(not_sils) = newCl;

%{
nClass = 100;
[N, vocSize] = size(patterns);
patterns = bsxfun(@rdivide, patterns, sqrt(sum(patterns.^2, 2)));
Winit = rand(N, nClass);
Winit = bsxfun(@rdivide, Winit, sqrt(sum(Winit.^2, 2)));
Hinit = rand(nClass, vocSize);
Hinit = bsxfun(@rdivide, Hinit, sqrt(sum(Hinit.^2, 1)));

[W, ~] = nmf(patterns, Winit, Hinit, 0.001, 7200, 100);
newW = sparse(N+length(sils), nClass);
newW(not_sils, :) = W;
%}

fprintf('Writing...  0.00%%');
fout = fopen(output, 'w');
fprintf(fout, header);

for f=1:count
    fprintf(fout, filenames{f});

    for i=1:length(indices{f})
        b = (timesteps{f}(i, 1)-1)*100000;
        f = timesteps{f}(i, 2)*100000;

        %[prob, z] = max(pzd(:, indices{f}(i)));
        %fprintf(fout, '%d %d p%d %f\n', b, f, z, prob);

        if newClasses(indices{f}(i)) ~= 0
            fprintf(fout, '%d %d p%d 0\n', b, f, newClasses(indices{f}(i)));
        else
            fprintf(fout, '%d %d sil 0\n', b, f);
        end

        %{
        [prob, z] = max(newW(indices{f}(i), :));
        if prob == 0
            fprintf(fout, '%d %d sil 0\n', b, f);
        else
            fprintf(fout, '%d %d p%d %f\n', b, f, z, prob);
        end
        %}
    end
    fprintf(fout, '.\n');
end

fclose all;
