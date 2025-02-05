function [current_fitness_Q, pop] = Operator_5(pop, elite_archive, idx_parent,inputnum,hiddennum,outputnum,net,ginputn,goutput_train,outputps)
% 随机选择两个元素块进行交叉，将后面的元素块插入到前面的元素块前面，生成新解

    % 获取解的长度
    parent = elite_archive(idx_parent, :);
    n = length(parent);

    % 检查输入向量的长度是否足够进行块交换
    block_size = 4;
    if n < 2 * block_size
        error('The input vector must be at least twice the block size.');
    end
    
    % 随机选择两个不同的块起始位置
    index1 = randi([1, n - block_size + 1]);
    index2 = randi([1, n - block_size + 1]);
    
    % 确保两个索引不同并且不重叠
    while index1 == index2 || (index1 < index2 && index2 < index1 + block_size) || (index2 < index1 && index1 < index2 + block_size)
        index2 = randi([1, n - block_size + 1]);
    end
    
    % 将较小的索引设为前块的索引
    if index1 > index2
        temp = index1;
        index1 = index2;
        index2 = temp;
    end
    
    % 创建新解并将后面的块插入到前面的块前面
    new_solution = parent;
    block = new_solution(index2:index2 + block_size - 1);
    
    % 删除后面的块
    new_solution(index2:index2 + block_size - 1) = [];
    
    % 插入后面的块到前面的块前面
    new_solution = [new_solution(1:index1-1), block, new_solution(index1:end)];
    
    %计算适应度值
    fitness_new_solution = fun(new_solution,inputnum,hiddennum,outputnum,net,ginputn,goutput_train,outputps);
    current_fitness_Q = fitness_new_solution;
    
    %更新pop
    pop(idx_parent, :) = new_solution;
    
end

