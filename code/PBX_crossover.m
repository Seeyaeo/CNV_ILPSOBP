function [current_fitness, pop, elite_archive] = PBX_crossover(elite_archive, pop, idx_parent1,idx_parent2, inputnum,hiddennum,outputnum,net,ginputn,goutput_train,outputps)
% PBX-交叉操作函数(基于位置的交叉)
% 输入：parent1, parent2  ——两个父代染色体
% 输出：child1,child2     ——两个子代染色体

parent1 = pop(idx_parent1, :);
parent2 = pop(idx_parent2, :);

len = length(parent1);
child1 = NaN(1, len);
child2 = NaN(1, len);

% 随机选择位置集合
positions = randperm(len, randi([1, len-1]));

% 复制选定位置的基因到子代
child1(positions) = parent1(positions);
child2(positions) = parent2(positions);

% 填充子代的其余部分
child1 = fill_remaining_positions(child1, parent2);
child2 = fill_remaining_positions(child2, parent1);

function child = fill_remaining_positions(child, parent)
    len = length(parent);
    parent_pos = 1;
    for i = 1:len
        if isnan(child(i))
            % 寻找父代中不在子代中的基因
            while ismember(parent(parent_pos), child)
                parent_pos = parent_pos + 1;
                if parent_pos > len
                    error('All parent genes are already in the offspring');
                end
            end
            child(i) = parent(parent_pos);
            parent_pos = parent_pos + 1;
        end
    end
end

fitness_parent1 = fun(parent1,inputnum,hiddennum,outputnum,net,ginputn,goutput_train,outputps);
fitness_parent2 = fun(parent2,inputnum,hiddennum,outputnum,net,ginputn,goutput_train,outputps);
fitness_child1 = fun(child1,inputnum,hiddennum,outputnum,net,ginputn,goutput_train,outputps);
fitness_child2 = fun(child2,inputnum,hiddennum,outputnum,net,ginputn,goutput_train,outputps);

% 选择更优的子代替代较差的父代
if fitness_child1 < fitness_child2
    if fitness_parent1 < fitness_parent2
        pop(idx_parent2, :) = child1;
    else
        pop(idx_parent1, :) = child2;
    end
    current_fitness = fitness_child1; %返回适应度值较小子代的适应度值
    elite_archive(end + 1, :) = child1;  %将适应度值较小的子代存入存档集中
else
    if fitness_parent1 < fitness_parent2
        pop(idx_parent2, :) = child2;
    else
        pop(idx_parent1, :) = child1;
    end
    current_fitness = fitness_child2; %返回适应度值较小子代的适应度值
    elite_archive(end + 1, :) = child2;  %将适应度值较小的子代存入存档集中
end

end

