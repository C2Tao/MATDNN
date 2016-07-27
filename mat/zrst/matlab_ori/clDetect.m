clusterNumber = 300;%30

path = '/home/taroball800405_hotmail_com/data2/ZRC/eng/';
dumpfile = [path sprintf('zrstExps/init/eng_%d.txt', clusterNumber)];
mfcPath = [path 'preprocess/eng.mfc'];

fin = fopen(mfcPath, 'r');
textfile = fopen(dumpfile, 'w');

iter=4;%4
powerorder=4;%4
peakdelta=0.2;%0.2

normalize = false;%false
frameThresh = 5%10,5
energyThresh = 0.2%0.3,0.2

featureId = 1;

F =[];
wavNames = cell(0);

fprintf('Start reading file %s\n', mfcPath);
w = 0;
line = fgets(fin);
while line ~= -1
    % read mfcc

    clear M Audio Memph cut I Idist Iwater N Ncut 
    w = w+1;

    wavNames{w} = line(1:end-1);
    M = zeros(39, 2000);
    m = 0;
    
    line = fgets(fin);
    while length(line) > 1
        m = m+1;
        M(:, m) = strread(line(13:end), '%f');
        line = fgets(fin);
    end
    if m < 2000
        M(:, m+1:end) = [];
    end
    M = M';
    line = fgets(fin);  % next filename

    % If it is too short, discard it
    if size(M, 1) < 2
        wavNames(w) = [];
        w = w-1;
        continue;
    end

    % segment

    Memph = M(:,13).^powerorder;

    [MAXTAB, MINTAB] = peakdet(Memph,peakdelta);
    if ~isempty(MINTAB)
        cut = [1; MINTAB(:,1); m];
    else
        cut = [1;  m];
    end
    c = length(cut);
    N = cell(c-1, 1);

    for i =1:c-1
        N{i} = M(cut(i):cut(i+1),:);
    end

    for x=1:c-1
        I = dotplot(N{x},N{x});
        Idist = cell(iter, 1);
        Idist{1}=I;

        for i=2:iter
            Idist{i} = im2bw(Idist{i-1},graythresh(Idist{i-1}));
            Idist{i} = bwdist(~Idist{i});
            Idist{i} = Idist{i}./max(max(Idist{i}));
        end

        Iwater = watershed(1-Idist{iter});
        Isub = find(diag(Iwater)==0);

        Ncut = [1;Isub;length(Iwater)];
        Nc = length(Ncut);
        
        ireal=1;
        for i =1:Nc-1
            fbeg = Ncut(i)   + cut(x)-1;
            fend = Ncut(i+1) + cut(x)-1;
            feature = mean( M(fbeg:fend,:) );
            if normalize
                feature = feature./sqrt(sum(feature.^2));
            end
            if fend-fbeg > frameThresh && feature(13) > energyThresh
                F = [F;  feature];
                FId{w}{x}{ireal}.featureId = featureId;
                featureId = featureId+1;
                ireal=ireal+1;
            end
        end
    end
end

fprintf('%d files are read in.\n', w);
fprintf('Start Clustering.\n');
clusterTable = kmeans(F,clusterNumber, 'MaxIter', 500);

[f temp]=size(F);

for y = 1:w
    for x =1:length(FId{y})
        for i =1:length(FId{y}{x})
            clusterId = clusterTable(FId{y}{x}{i}.featureId);
            FId{y}{x}{i}.clusterId = clusterId;
        end
    end
end

fprintf('Output starts.\n');
textfile = fopen(dumpfile, 'w');
for y = 1:w
    fprintf(textfile,'%s\n', wavNames{y});
    for x =1:length(FId{y})
        for i =1:length(FId{y}{x})
            fprintf(textfile,'%d ', FId{y}{x}{i}.clusterId);
        end
        fprintf(textfile,'\n');
    end
end

fclose('all');
