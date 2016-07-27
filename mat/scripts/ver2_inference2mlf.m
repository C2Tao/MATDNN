function [] = ver2_inference2mlf(hmm)

load(['../exp/merge/data.mat']);
infer = ['../exp/' hmm '/inference'];
output = ['../exp/' hmm '/result/result.mlf'];

topics = zeros(t, 2);
fin = fopen(infer, 'r');
line = fgets(fin); % header
line = fgets(fin);
while line ~= -1
    parse = strread(line, '%s');
    id = strread(parse{1}, '%d') + 1;
    z = strread(parse{3}, '%d') + 1;
    prob = strread(parse{4}, '%f');

    topics(id, :) = [z, prob];
    line = fgets(fin);
end


fprintf('Writing...  0.00%%');
fout = fopen(output, 'w');
fprintf(fout, header);
for j=1:count
    fprintf(fout, filenames{j});
    for i=1:length(indices{j})
        b = (timesteps{j}(i, 1)-1)*100000;
        f = timesteps{j}(i, 2)*100000;
        topic = topics(indices{j}(i), :);
        if topic(1) ~= 0
            fprintf(fout, '%d %d p%d %f\n', b, f, topic(1), topic(2));
        else
            fprintf(fout, '%d %d sil 1\n', b, f);
        end
    end
    fprintf(fout, '.\n');

    fprintf('\b\b\b\b\b\b\b%6.2f%%', 100*j/count);
end
fclose all;

end
