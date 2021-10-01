function [ GBEST ,  cgCurve ] = pso(train_X, train_y, test_X, test_y, fobj, lb, ub, dim)
   tic;
               disp('>>> PSO started>>>'); 

   % PSO parameters
    noP = 50;
    maxIter = 100;
    Stop_mse = 0.0002;        % Stop value of mse

    RunNo  = 50;
    BestSolutions_PSO = zeros(1 , RunNo);

    % Define the details of the objective function
    nVar = dim;

    % Extra variables for data visualization
    average_objective = zeros(1, maxIter);
    cgCurve = zeros(1, maxIter);
    FirstP_D1 = zeros(1 , maxIter);
    position_history = zeros(noP , maxIter , nVar );

    % Define the PSO's paramters
    wMax = 0.9;
    wMin = 0.2;
    c1 = 2;
    c2 = 2;
    vMax = (ub - lb) .* 0.2;
    vMin  = -vMax;

    % The PSO algorithm

    % Initialize the particles
    for k = 1 : noP
        Swarm.Particles(k).X = (ub-lb) .* rand(1,nVar) + lb;
        Swarm.Particles(k).V = zeros(1, nVar);
        Swarm.Particles(k).PBEST.X = zeros(1,nVar);
        Swarm.Particles(k).PBEST.O = inf;

        Swarm.GBEST.X = zeros(1,nVar);
        Swarm.GBEST.O = inf;
    end


    % Main loop
    for t = 1 : maxIter

        % Calcualte the objective value
        for k = 1 : noP

            currentX = Swarm.Particles(k).X;
            position_history(k , t , : ) = currentX;


            Swarm.Particles(k).O = fobj(train_X, train_y, test_X, test_y, currentX);
            average_objective(t) =  average_objective(t)  + Swarm.Particles(k).O;

            % Update the PBEST
            if Swarm.Particles(k).O < Swarm.Particles(k).PBEST.O
                Swarm.Particles(k).PBEST.X = currentX;
                Swarm.Particles(k).PBEST.O = Swarm.Particles(k).O;
            end

            % Update the GBEST
            if Swarm.Particles(k).O < Swarm.GBEST.O
                Swarm.GBEST.X = currentX;
                Swarm.GBEST.O = Swarm.Particles(k).O;
            end
        end

        % Update the X and V vectors
        w = wMax - t .* ((wMax - wMin) / maxIter);

        FirstP_D1(t) = Swarm.Particles(1).X(1);

        for k = 1 : noP
            Swarm.Particles(k).V = w .* Swarm.Particles(k).V + c1 .* rand(1,nVar) .* (Swarm.Particles(k).PBEST.X - Swarm.Particles(k).X) ...
                + c2 .* rand(1,nVar) .* (Swarm.GBEST.X - Swarm.Particles(k).X);


            % Check velocities
            index1 = find(Swarm.Particles(k).V > vMax);
            index2 = find(Swarm.Particles(k).V < vMin);

            Swarm.Particles(k).V(index1) = vMax(index1);
            Swarm.Particles(k).V(index2) = vMin(index2);

            Swarm.Particles(k).X = Swarm.Particles(k).X + Swarm.Particles(k).V;

            % Check positions
            index1 = find(Swarm.Particles(k).X > ub);
            index2 = find(Swarm.Particles(k).X < lb);

            Swarm.Particles(k).X(index1) = ub(index1);
            Swarm.Particles(k).X(index2) = lb(index2);

        end

        outmsg = ['Iteration# ', num2str(t) , ' Swarm.GBEST.O = ' , num2str(Swarm.GBEST.O)];
        disp(outmsg);

        cgCurve(t) = Swarm.GBEST.O;
        average_objective(t) = average_objective(t) / noP;
        
        Swarm.GBEST.X(1) = round(Swarm.GBEST.X(1));
        Swarm.GBEST.X(2) = round(Swarm.GBEST.X(2));
        Swarm.GBEST.X(3) = round(Swarm.GBEST.X(3));
%         Swarm.GBEST.X(5) = round(Swarm.GBEST.X(5));

        GBEST = Swarm.GBEST;

        if GBEST.O <= Stop_mse
            disp('>>> Reach to Stop Condition! >>>');
            break
        end
    end
    toc;

end