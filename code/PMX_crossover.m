% PMX
function [current_fitness, pop, elite_archive] = PMX_crossover(elite_archive, pop, idx_parent1,idx_parent2, inputnum,hiddennum,outputnum,net,ginputn,goutput_train,outputps)

parent1 = pop(idx_parent1, :); 
parent2 = pop(idx_parent2, :);

len = length(parent1);
child1 = zeros(1, len);
child2 = zeros(1, len);

crossover_points = sort(randperm (len, 2));
point1 = crossover_points(1);
point2 = crossover_points(2);

child1(point1:point2) = parent1(point1:point2);
child2(point1:point2) = parent2(point1:point2);

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
