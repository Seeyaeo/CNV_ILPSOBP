function [current_fitness_Q, pop] = Operator_1(pop, elite_archive, idx_parent,inputnum,hiddennum,outputnum,net,ginputn,goutput_train,outputps)

    parent = elite_archive(idx_parent, :);
    n = length(parent);
    
    index1 = randi(n);
    index2 = randi(n);
    
    while index1 == index2
        index2 = randi(n);
    end
    
    new_solution = parent;
    temp = new_solution(index1);
    new_solution(index1) = new_solution(index2);
    new_solution(index2) = temp;
    
    fitness_new_solution = fun(new_solution,inputnum,hiddennum,outputnum,net,ginputn,goutput_train,outputps);
    current_fitness_Q = fitness_new_solution;

    pop(idx_parent, :) = new_solution;
end
