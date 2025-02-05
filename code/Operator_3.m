function [current_fitness_Q, pop] = Operator_3(pop, elite_archive, idx_parent,inputnum,hiddennum,outputnum,net,ginputn,goutput_train,outputps)
% 随机选择两个元素，反转其中间的元素，生成新解

    % 获取解的长度
    parent = elite_archive(idx_parent, :);
    n = length(parent);
    
    % 随机选择两个不同的索引
    index1 = randi(n);
    index2 = randi(n);
    
    % 确保两个元素索引不同
    while index1 == index2
        index2 = randi(n);
    end
    
    % 创建新解并反转两个索引位置之间的元素
    new_solution = parent;
    new_solution(index1:index2) = new_solution(index2:-1:index1);
    
    %计算适应度值
    fitness_new_solution = fun(new_solution,inputnum,hiddennum,outputnum,net,ginputn,goutput_train,outputps);
    current_fitness_Q = fitness_new_solution;
    
    %更新pop
    pop(idx_parent, :) = new_solution;
end

