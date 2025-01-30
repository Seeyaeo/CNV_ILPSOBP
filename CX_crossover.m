% CX
function [current_fitness, pop, elite_archive] = CX_crossover(elite_archive, pop, idx_parent1,idx_parent2, inputnum,hiddennum,outputnum,net,ginputn,goutput_train,outputps)

parent1 = pop(idx_parent1, :);
parent2 = pop(idx_parent2, :);

len = length(parent1);
child1 = NaN(1, len);
child2 = NaN(1, len);

index_map1 = false(1, len);


cycle = 1;  
while ~all(index_map1)
    start = find(~index_map1, 1);
    idx = start;  
    
    if isempty(start)
        break;
    end
    
    while true
        child1(idx) = parent1(idx);
        child2(idx) = parent2(idx);
        index_map1(idx) = true;
        idx = find(parent1 == parent2(idx), 1);
        if isempty(idx) || idx == start
            break;
        end
    end

    if mod(cycle, 2) == 1
        temp = child1;
        child1 = child2;
        child2 = temp; 
    end
    cycle = cycle + 1;
end

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

if fitness_child1 < fitness_child2
    if fitness_parent1 < fitness_parent2
        pop(idx_parent2, :) = child1;
    else
        pop(idx_parent1, :) = child2;
    end
    current_fitness = fitness_child1; 
    elite_archive(end + 1, :) = child1;  
else
    if fitness_parent1 < fitness_parent2
        pop(idx_parent2, :) = child2;
    else
        pop(idx_parent1, :) = child1;
    end
    current_fitness = fitness_child2; 
    elite_archive(end + 1, :) = child2;  
end

end


