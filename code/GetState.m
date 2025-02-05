function state = GetState(delt_fitness)
    if delt_fitness < 0
        state = 1;
    elseif delt_fitness == 0
        state = 2;
    elseif delt_fitness > 0
        state = 3;
    end
end