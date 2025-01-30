function [current_fitness_Q, pop] = Operator_2(pop, elite_archive, idx_parent,inputnum,hiddennum,outputnum,net,ginputn,goutput_train,outputps)

    parent = elite_archive(idx_parent, :);
    n = length(parent);

    index1 = randi(n);
    index2 = randi(n);

    while index1 == index2
        index2 = randi(n);
    end

    new_solution = parent;
    if index1 < index2   
        temp = new_solution(index2);
        new_solution(index2) = [];  
        new_solution = [new_solution(1:index1-1), temp, new_solution(index1:end)];
    else
        temp = new_solution(index1);
        new_solution(index1) = [];  
        new_solution = [new_solution(1:index2-1), temp, new_solution(index2:end)];
    end

    fitness_new_solution = fun(new_solution,inputnum,hiddennum,outputnum,net,ginputn,goutput_train,outputps);
    current_fitness_Q = fitness_new_solution;

    pop(idx_parent, :) = new_solution;
    
end

