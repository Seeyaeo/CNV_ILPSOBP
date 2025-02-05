function [current_fitness, pop, elite_archive] = PMX_crossover(elite_archive, pop, idx_parent1,idx_parent2, inputnum,hiddennum,outputnum,net,ginputn,goutput_train,outputps)
% PMX-交叉操作函数(部分映射交叉)
% 输入：parent1, parent2  ——两个父代染色体
% 输出：child1,child2     ——两个子代染色体

parent1 = pop(idx_parent1, :); 
parent2 = pop(idx_parent2, :);

len = length(parent1);
child1 = zeros(1, len);
child2 = zeros(1, len);

% 生成两个交叉点
crossover_points = sort(randperm (len, 2));
point1 = crossover_points(1);
point2 = crossover_points(2);

% 部分映射交叉
child1(point1:point2) = parent1(point1:point2);
child2(point1:point2) = parent2(point1:point2);

% 处理child1
for i = 1:len
    if i >= point1 && i <= point2
        continue;
    end
    val = parent2(i);
    while ismember(val, child1(point1:point2))
        idx = find(parent1 == val);
        val = parent2(idx);
    end
    child1(i) = val;
end

% 处理child2
for i = 1:len
    if i >= point1 && i <= point2
        continue;
    end
    val = parent1(i);
    while ismember(val, child2(point1:point2))
        idx = find(parent2 == val);
        val = parent1(idx);
    end
    child2(i) = val;
end

fitness_parent1 = fun(parent1,inputnum,hiddennum,outputnum,net,ginputn,goutput_train,outputps);
fitness_parent2 = fun(parent2,inputnum,hiddennum,outputnum,net,ginputn,goutput_train,outputps);
fitness_child1 = fun(child1,inputnum,hiddennum,outputnum,net,ginputn,goutput_train,outputps);
fitness_child2 = fun(child2,inputnum,hiddennum,outputnum,net,ginputn,goutput_train,outputps);

% 选择更优的子代替代较差的父代, 并且更新存档集
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
    elite_archive(end + 1, :) = child2;
end

end

