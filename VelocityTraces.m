data_folder = 'data/';
data_type = '_withVelocity.csv';
%vids = [2];
vids = [2 25 33 39 49 72 74];

for i = vids
    filename = strcat(data_folder, num2str(i), data_type);
    disp(filename)
    cars = csvread(filename);
    % cog_x	cog_y	frame	height	obj	velocity	width	x	y
    
    maintain_velocity_spec_output = [];
    fast_cars_spec_output = [];
    slow_cars_spec_output = [];
    largest_deceleration_spec_output = [];
    unique_objects = unique(cars(:,5));
    
    avg_velocity = mean(cars(:,6));
    
    for car = unique_objects'
        C = cars(cars(:,5)>car-1 & cars(:,5)<car+1,:);
        C = sortrows(C, 3);
        times = 1:5:length(C(:,3));
        velocity = C(:,6);
        velocity = velocity(1:5:end);
        if length(times) == 1
            continue
        end
        
        S = BreachTraceSystem({'v'},[times', velocity]);
        
        %%% Cars Moving Slower than average (find high c)
        spec2 = STL_Formula('mu', 'ev_[t0,T] ((avg_v - v[t]) > c)');
        spec2 = set_params(spec2, {'t0', 'T', 'avg_v'}, [times(1) times(end) avg_velocity]);
        P = ParamSynthProblem(S, spec2, {'c'}, [-50, 50]);
        c = P.solve();
        slow_cars_spec_output = [slow_cars_spec_output; [i, car, c]];
        
        %%% Cars Moving Faster than average (find high c)
        spec1 = STL_Formula('mu', 'ev_[t0,T] ((v[t] - avg_v) > c)');
        spec1 = set_params(spec1, {'t0', 'T', 'avg_v'}, [times(1) times(end) avg_velocity]);
        P = ParamSynthProblem(S, spec1, {'c'}, [-50, 50]);
        c = P.solve();
        fast_cars_spec_output = [fast_cars_spec_output; [i, car, c]];
        
        %%% Maintain Velocity
        spec3 = STL_Formula('phi', 'ev_[t0,T] ((v[t] > c1) and ev_[0,1.5] (v[t] < c2))');
        spec3 = set_params(spec3, {'t0', 'T'}, [times(1) times(end)]);
        P = ParamSynthProblem(S, spec3, {'c2','c1'}, [0,100 ; 0, 100]);
        P.solver_options.monotony = [1 -1];
        c = P.solve();
        c1 = c(2);
        c2 = c(1);
        maintain_velocity_spec_output = [maintain_velocity_spec_output; [i, car, c1, c2]];
        
%         %%% Largest Deceleration (find low c?)
%         spec4 = STL_Formula('phi', 'ev_[t0,T] (min(diff(v[t])) < c)');
%         spec4 = set_params(spec4, {'t0', 'T'}, [times(1) times(end)]);
%         P = ParamSynthProblem(S, spec4, {'c'}, [-50, 50]);
%         P.solver_options.monotony = -1;
%         c = P.solve();
%         largest_deceleration_spec_output = [largest_deceleration_spec_output; [i, car, c]];
    end
    
    csvwrite(strcat(num2str(i), "_maintain_velocity.csv"), maintain_velocity_spec_output)
    % csvwrite(strcat(num2str(i), "_largest_deceleration.csv"_, largest_deceleration_spec_output)
    csvwrite(strcat(num2str(i), "_fast_cars.csv"), fast_cars_spec_output)
    csvwrite(strcat(num2str(i), "_slow_cars.csv"), slow_cars_spec_output)
end



    