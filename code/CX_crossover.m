function [current_fitness, pop, elite_archive] = CX_crossover(elite_archive, pop, idx_parent1,idx_parent2, inputnum,hiddennum,outputnum,net,ginputn,goutput_train,outputps)
% CX-交叉操作函数(循环交叉)
% 输入：parent1, parent2  ——两个父代染色体
% 输出：child1,child2     ——两个子代染色体

parent1 = pop(idx_parent1, :);
parent2 = pop(idx_parent2, :);

len = length(parent1);
child1 = NaN(1, len);
child2 = NaN(1, len);

% 标记已经处理过的索引
index_map1 = false(1, len);

% 循环查找和填充循环部分
cycle = 1;  %cycle用来标记当前是第几个循环
while ~all(index_map1)
    % 找到一个未标记的起始点
    start = find(~index_map1, 1);
    idx = start;  %start表示一个未标记的起始点
    
    if isempty(start)
        break;
    end
    
    while true
        child1(idx) = parent1(idx);
        child2(idx) = parent2(idx);
        index_map1(idx) = true;
       
        % 找到parent2中与当前parent1(idx)相同元素在parent1中的位置
        idx = find(parent1 == parent2(idx), 1);
        if isempty(idx) || idx == start
            break;
        end
    end
    
    % 判断是否交换交叉父代
    if mod(cycle, 2) == 1   %如果cycle是奇数，则交换父代基因进行填充
                            %如果cycle是偶数，则保持当前父代基因不变
                
            temp = child1;
            child1 = child2;
            child2 = temp; 
    end
    cycle = cycle + 1;
end

%处理NaN的值
for i = 1:len
    if isnan(child1(i))
        child1(i) = parent2(i);
    end
    if isnan(child2(i))
        child2(i) = parent1(i);
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


