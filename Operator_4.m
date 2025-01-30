function [current_fitness_Q, pop] = Operator_4(pop, elite_archive, idx_parent,inputnum,hiddennum,outputnum,net,ginputn,goutput_train,outputps)

    parent = elite_archive(idx_parent, :);
    n = length(parent);
    
    block_size = 4;
    if n < 2 * block_size
        error('The input vector must be at least twice the block size.');
    end
    
    index1 = randi([1, n - block_size + 1]);
    index2 = randi([1, n - block_size + 1]);

    while index1 == index2 || (index1 < index2 && index2 < index1 + block_size) || (index2 < index1 && index1 < index2 + block_size)
        index2 = randi([1, n - block_size + 1]);
    end

    new_solution = parent;
    block1 = new_solution(index1:index1 + block_size - 1);
    block2 = new_solution(index2:index2 + block_size - 1);

    new_solution(index1:index1 + block_size - 1) = block2;
    new_solution(index2:index2 + block_size - 1) = block1;

    fitness_new_solution = fun(new_solution,inputnum,hiddennum,outputnum,net,ginputn,goutput_train,outputps);
    current_fitness_Q = fitness_new_solution;

    pop(idx_parent, :) = new_solution;
    
end

