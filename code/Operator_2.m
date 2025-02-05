function [current_fitness_Q, pop] = Operator_2(pop, elite_archive, idx_parent,inputnum,hiddennum,outputnum,net,ginputn,goutput_train,outputps)
% 随机选择两个元素进行交叉，将后面的元素插入到前面的元素前面，生成新解

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
    
    % 创建新解并将后面的元素插入到前面的元素前面
    new_solution = parent;
    if index1 < index2   
        temp = new_solution(index2);
        new_solution(index2) = [];  %删除后面的元素，给它置为空
        new_solution = [new_solution(1:index1-1), temp, new_solution(index1:end)];
    else
        temp = new_solution(index1);
        new_solution(index1) = [];  %删除后面的元素，给它置为空
        new_solution = [new_solution(1:index2-1), temp, new_solution(index2:end)];
    end
    
    %计算适应度值
    fitness_new_solution = fun(new_solution,inputnum,hiddennum,outputnum,net,ginputn,goutput_train,outputps);
    current_fitness_Q = fitness_new_solution;

    %更新pop
    pop(idx_parent, :) = new_solution;
    
end

