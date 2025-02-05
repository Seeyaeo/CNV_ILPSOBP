function [current_fitness, pop, elite_archive] = OX_crossover(elite_archive, pop, idx_parent1,idx_parent2, inputnum,hiddennum,outputnum,net,ginputn,goutput_train,outputps)
% OX-交叉操作函数(顺序交叉)
% 输入：parent1, parent2  ——两个父代染色体
% 输出：child1,child2     ——两个子代染色体

parent1 = pop(idx_parent1, :);
parent2 = pop(idx_parent2, :);

len = length(parent1);
child1 = NaN(1, len);
child2 = NaN(1, len);

% 生成两个交叉点
crossover_points = sort(randperm(len, 2));
point1 = crossover_points(1);
point2 = crossover_points(2);

% 复制部分父代片段到子代
child1(point1:point2) = parent1(point1:point2);
child2(point1:point2) = parent2(point1:point2);

% 剩余部分按顺序填充
child1 = fill_child(child1, parent2, point1, point2);
child2 = fill_child(child2, parent1, point1, point2);

   function child = fill_child(child, parent, point1, point2)
       len = length(child);
       current_pos = mod(point2, len) + 1;
        for i = 1:len
            candidate = parent(mod(point2 + i - 1, len) + 1);
            if ~ismember(candidate, child)
                child(current_pos) = candidate;
                current_pos = mod(current_pos, len) + 1;
%                 if current_pos == 0
%                     current_pos = len;
%                 end
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

