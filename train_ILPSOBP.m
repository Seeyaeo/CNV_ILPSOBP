%% Clear environment variables
clc
clear
%% Importing training sample data

inputnum = 5;
hiddennum = 4;
outputnum = 4;

data1 = load('data\0.2_4x_mat\sim1_4_4100_read_trains.mat');
data2 = load('data\0.3_4x_mat\sim1_4_4100_read_trains.mat');
data3 = load('data\0.4_4x_mat\sim1_4_4100_read_trains.mat');
data4 = load('data\0.2_6x_mat\sim1_6_6100_read_trains.mat');
data5 = load('data\0.3_6x_mat\sim1_6_6100_read_trains.mat');
data6 = load('data\0.4_6x_mat\sim1_6_6100_read_trains.mat');

data_trains1=load('data\0.2_4x_11.1\sim1_4_4100_trains.txt');
data_trains2=load('data\0.3_4x_11.1\sim1_4_4100_trains.txt');
data_trains3=load('data\0.4_4x_11.1\sim1_4_4100_trains.txt');
data_trains4=load('data\0.2_6x_11.1\sim1_6_6100_trains.txt');
data_trains5=load('data\0.3_6x_11.1\sim1_6_6100_trains.txt');
data_trains6=load('data\0.4_6x_11.1\sim1_6_6100_trains.txt');

data_trains = [data_trains1;data_trains2;data_trains3;data_trains5;data_trains6;];
column = [2,3,4,5,6];
[m1,n1] = size(data_trains);

trainLines = m1;
gdata(1:trainLines,:) = data_trains(1:trainLines,:);

%Random sorting
k = rand(1,trainLines);
[m,n] = sort(k);

%Get input and output data
ginput = gdata(:,column);
goutput1 = gdata(:,7);

goutput = zeros(trainLines,4);
for i = 1:trainLines
    switch goutput1(i)
        case 0
            goutput(i,:) = [1 0 0 0];
        case 1
            goutput(i,:) = [0 1 0 0];
        case 2
            goutput(i,:) = [0 0 1 0];
        case 3
            goutput(i,:) = [0 0 0 1];
    end
end

%Find the training data and prediction data
ginput_train = ginput(n(1:trainLines),:)';
goutput_train = goutput(n(1:trainLines),:)';

%Normalization
[ginputn,ginputps] = mapminmax(ginput_train);
[outputn,outputps] = mapminmax(goutput_train);

%Network Training
net = newff(ginputn,goutput_train,hiddennum);
numsum = inputnum * hiddennum + hiddennum + hiddennum * outputnum + outputnum;


c1 = 1.49445;
c2 = 1.49445;
lr = 0.5;
history = [];
init_fitness = [];
init_index = [];  
maxgen=2;      
sizepop=2;     

Vmax = 1;
Vmin = -1;
popmax = 5;
popmin = -5;

for i = 1:sizepop
    pop(i,:) = 5 * rands(1,numsum);
    V(i,:) = rands(1,numsum);
    fitness(i) = fun(pop(i,:),inputnum,hiddennum,outputnum,net,ginputn,goutput_train,outputps);
end

[bestfitness, bestindex] = min(fitness);

while bestfitness <= 100
    init_fitness = [init_fitness bestfitness];
    init_index = [init_index bestindex];
    history = [history bestfitness];
    pop(bestindex,:) = 5*rands(1,numsum);
    V(bestindex,:) = rands(1,numsum);
    fitness(bestindex) = fun(pop(bestindex,:),inputnum,hiddennum,outputnum,net,ginputn,goutput_train,outputps);
    [bestfitness, bestindex] = min(fitness);
end

init_best = [init_fitness;init_index];
init_pop = pop;

zbest = pop(bestindex,:);  
gbest = pop;    
fitnessgbest = fitness;   
fitnesszbest = bestfitness;   

strategy_num = 3;
mutation_num = 3;  
flag = zeros(1,strategy_num);
p = ones(1,strategy_num) * (1 / strategy_num);
success_mem = zeros(1,strategy_num);
failure_mem = zeros(1,strategy_num);
rk = cumsum(ones(1,strategy_num) ./ strategy_num);
strategy_improve = zeros(1,strategy_num);


alpha = 0.1;  
gamma = 0.8; 
epsilon = 0.95; 
delt_fitness = 0;
Qtable = zeros(4, 4);
current_fitness = 0;
elite_archive = [];  

delt_fitness_Q = 0;
Qtable_Q = zeros(4, 4);
current_fitness_Q = 0 ;


for i = 1:maxgen  
    for j = 1:sizepop
        probility = rand;
        if probility <= rk(1)
            strategy = 1;
            V(j,:) = V(j,:) + (c1 + c2) * rand * (gbest(j,:) - pop(j,:));
            V(j,find(V(j,:) > Vmax)) = Vmax;
            V(j,find(V(j,:) < Vmin)) = Vmin;
        elseif probility <= rk(2)
            strategy = 2;
            V(j,:) = V(j,:) + (c1 + c2) * rand * (zbest - pop(j,:));
            V(j,find(V(j,:) > Vmax)) = Vmax;
            V(j,find(V(j,:) < Vmin)) = Vmin;
        elseif probility <= rk(3)
            strategy = 3;
            V(j,:) = V(j,:) + c1 * rand * (gbest(j,:) - pop(j,:)) + c2 * rand * (zbest - pop(j,:));
            V(j,find(V(j,:) > Vmax)) = Vmax;
            V(j,find(V(j,:) < Vmin)) = Vmin;
        end
        
        pop(j,:) = pop(j,:) + V(j,:);
        pop(j,find(pop(j,:) > popmax)) = popmax;
        pop(j,find(pop(j,:) < popmin)) = popmin;
        
        
        fitness(j) = fun(pop(j,:),inputnum,hiddennum,outputnum,net,ginputn,goutput_train,outputps);
       
        
        if fitness(j) < fitnessgbest(j)
            strategy_improve(strategy) = strategy_improve(strategy) + (fitnessgbest(j) - fitness(j)) / fitnessgbest(j);
            gbest(j,:) = pop(j,:); 
            fitnessgbest(j) = fitness(j);
            success_mem(strategy) = success_mem(strategy) + 1;
        else
            failure_mem(strategy) = failure_mem(strategy) + 1;
        end
    
        
        if fitness(j) < fitnesszbest
            zbest = pop(j,:);
            fitnesszbest = fitness(j);
            history = [history fitnesszbest]; 
        end
    end
    
    
    total = success_mem + failure_mem;
    total(find(total == 0)) = 1;
    strategy_improve = strategy_improve ./ total;
    if isequal(strategy_improve,zeros(1,strategy_num))  %初始化
       strategy_improve = ones(1,strategy_num);
    end
    strategy_improve(find(strategy_improve == 0)) = 0.1 * min(strategy_improve(strategy_improve ~= 0)); %防止某些策略的被选的次数为0
    strategy_improve = strategy_improve ./ sum(strategy_improve);
    
    f = strategy_improve;
    p = p + (f - sum(p .* f)) .* p .* lr;
    
    p(find(p <= 0)) = 0; 
    p = p ./ (sum(p));  
    rk = cumsum(p);   
  
    for j=1:3
        if flag(1,j) >= mutation_num && p(1,j) > 0.5
            flag(1,j) = flag(1,j)+1;
        elseif flag(1,j) < mutation_num && p(1,j) > 0.5
            part_improve = p(1,j) - 0.5;  
            distribe = p;
            distribe(:,j) = 0;
            distribe = part_improve .* (distribe ./ sum(distribe)); 
            p = distribe + p; 
            p(1,j) = 0.5; 
            flag(1,j) = flag(1,j) + 1;
        end
    end
    
    pos = unidrnd(numsum);  
    if rand > 0.90
      pop(j,pos) = 5 * rands(1,1);  
    end
    success_mem = zeros(1,strategy_num);  
    failure_mem = zeros(1,strategy_num);  
    strategy_improve = zeros(1,strategy_num);

    % global search based on SARSA
    last_fitness = current_fitness;
    current_state = GetState(delt_fitness);
    %choose and execute action
    if rand < epsilon || all(Qtable(current_state, :) == 0)
        action = ceil(rand * 4);
    else
        [~, action] = max(Qtable(current_state, :));
    end
    % randomly select two individuals
    selected_individual_1 = randi(size(pop, 1)); 
    selected_individual_2 = randi(size(pop, 1));
    while selected_individual_1 == selected_individual_2
        selected_individual_2 = randi(size(pop, 1));
    end
    switch(action)
        case 0
            [current_fitness, pop, elite_archive] = PMX_crossover(elite_archive, pop, selected_individual_1, selected_individual_2, inputnum,hiddennum,outputnum,net,ginputn,goutput_train,outputps); %PMX_crossover
        case 1
            [current_fitness, pop, elite_archive] = OX_crossover(elite_archive, pop, selected_individual_1, selected_individual_2, inputnum,hiddennum,outputnum,net,ginputn,goutput_train,outputps);  %OX_crossover
        case 2
            [current_fitness, pop, elite_archive] = CX_crossover(elite_archive, pop, selected_individual_1, selected_individual_2, inputnum,hiddennum,outputnum,net,ginputn,goutput_train,outputps);  %CX_crossover
        case 3
            [current_fitness, pop, elite_archive] = PBX_crossover(elite_archive, pop, selected_individual_1, selected_individual_2, inputnum,hiddennum,outputnum,net,ginputn,goutput_train,outputps); %PBX_crossover
    end
    %update Qtable
    %calculate the fitness of the operated individual
    delt_fitness = last_fitness - current_fitness;
    if delt_fitness > 0 % find a nice solution
        reward = 10;
    else
        reward = 0;
        
    end
    next_state = GetState(delt_fitness);
    col_idx = randi(4);
    maxReward = Qtable(next_state, col_idx);
    Qtarget = reward + gamma * maxReward;
    Qtable(current_state, action) = Qtable(current_state, action) + alpha * (Qtarget - Qtable(current_state, action));
    
    % local search based on Q-learning
    if isempty(elite_archive)
    else
        last_fitness_Q = current_fitness_Q;
        current_state_Q = GetState(delt_fitness_Q);
        % choose and execute action
        if rand < epsilon || all(Qtable_Q(current_state_Q, :) == 0)
            action_Q = ceil(rand * 4);
        else
            [~, action_Q] = max(Qtable_Q(current_state_Q, :));
        end
        % randomly select one individual from best_pop
        selected_individual = randi(size(elite_archive, 1)); 
        switch(action_Q)
            case 0
                [current_fitness_Q, pop] = Operator_1(pop, elite_archive, selected_individual, inputnum,hiddennum,outputnum,net,ginputn,goutput_train,outputps);
            case 1
                [current_fitness_Q, pop] = Operator_2(pop, elite_archive, selected_individual, inputnum,hiddennum,outputnum,net,ginputn,goutput_train,outputps);
            case 2
                [current_fitness_Q, pop] = Operator_3(pop, elite_archive, selected_individual, inputnum,hiddennum,outputnum,net,ginputn,goutput_train,outputps);
            case 3
                [current_fitness_Q, pop] = Operator_4(pop, elite_archive, selected_individual, inputnum,hiddennum,outputnum,net,ginputn,goutput_train,outputps);
            case 4
                [current_fitness_Q, pop] = Operator_5(pop, elite_archive, selected_individual, inputnum,hiddennum,outputnum,net,ginputn,goutput_train,outputps);
        end
        %update Qtable
        %calculate the fitness of the operated individual
        delt_fitness_Q = last_fitness_Q - current_fitness_Q;
        if delt_fitness_Q > 0
            reward_Q = 10;
        else
            reward_Q = 0;
        end
        next_state_Q = GetState(delt_fitness_Q);
        maxReward_Q = max(Qtable_Q(next_state_Q, :));
        Qtarget_Q = reward_Q + gamma * maxReward_Q;
        Qtable_Q(current_state_Q, action_Q) = Qtable_Q(current_state_Q, action_Q) + alpha * (Qtarget_Q - Qtable_Q(current_state_Q, action_Q));
        
    end
       
end

if isempty(init_best)   
else    
    [init_min, index] = min(init_best(1,:));
    if init_min < fitnesszbest
        zbest = init_pop(index,:);
        fitnesszbest = init_min;
    end
end

x = zbest;
figure(1)
f1 = plot(history,'rd');
xlabel('time','Fontname','Times New Roman');
ylabel('zbest','Fontname','Times New Roman');


w1 = x(1:inputnum * hiddennum);
B1 = x(inputnum * hiddennum + 1:inputnum * hiddennum + hiddennum);
w2 = x(inputnum * hiddennum + hiddennum + 1:inputnum * hiddennum + hiddennum + hiddennum * outputnum);
B2 = x(inputnum * hiddennum + hiddennum + hiddennum * outputnum + 1:inputnum * hiddennum + hiddennum + hiddennum * outputnum + outputnum);
B2 = B2';

net.iw{1,1} = reshape(w1,hiddennum,inputnum);
net.lw{2,1} = reshape(w2,outputnum,hiddennum);
net.b{1} = reshape(B1,hiddennum,1);
net.b{2} = B2;

%Initializing the network structure
net.trainParam.epochs = 200;
net.trainParam.lr = 0.1;
net.trainParam.goal = 0.00001;


[net,per2] = train(net,ginputn,goutput_train);
save ('PSOBP','net');
