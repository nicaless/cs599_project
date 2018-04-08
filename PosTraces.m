data_folder = 'data/';
data_type = '_byobject.csv';
vids = [2];

for i = vids
    filename = strcat(data_folder, num2str(i), data_type);
    disp(filename)
    cars = csvread(filename);
    % height	width	y	x	cog_y	cog_x	obj	frame
    
    collisions_spec_output = [];
    unique_objects = unique(cars(:,7));
    
    for car = unique_objects'
        C = cars(cars(:,7)>car-1 & cars(:,7)<car+1,:);
        C = sortrows(C, 8);
        times = 2:1:length(C(:,8));
        if length(times) == 1
            continue
        end
        C_x = C(:,6);
        C_y = C(:,5);
        change_x = diff(C_x);
        change_y = diff(C_y);
        no_change = .001;
        
        S_x = BreachTraceSystem({'x'},[times', change_x]);
        S_y = BreachTraceSystem({'y'},[times', change_y]);
        
        %%% Car X Position Does not change for some time
        spec1 = STL_Formula('phi', 'ev_[t0,T] (x[t] <= no_change Until_[0, tau] x[t] > no_change)');
        spec1 = set_params(spec1, {'t0', 'T', 'no_change'}, [times(1) times(end) no_change]);
        P = ParamSynthProblem(S_x, spec1, {'tau'}, [0, 50]);
        c1 = P.solve();

        %%% Car Y Position Does not change for some time
        spec2 = STL_Formula('phi', 'ev_[t0,T] (y[t] == no_change Until_[0, tau] y[t] != no_change)');
        spec2 = set_params(spec2, {'t0', 'T', 'no_change'}, [times(1) times(end) no_change]);
        P = ParamSynthProblem(S_y, spec2, {'tau'}, [0, 50]);
        c2 = P.solve();

        % find all frames this car is in and find other cars in these
        % frames
        unique_frames = unique(C(:,8));
        unique_frames = unique_frames';
        unique_frames = unique_frames(1:5:end);
        
        % for each frame get max difference of x val and y_val and append
        % to trace
        diff_x = [];
        diff_y = [];
        for f = unique_frames
            C_x_f = C_x(f);
            other_cars = cars(cars(:,8)>f-1 & cars(:,8)<f+1,:);
            other_cars_x = other_cars(:,6);
            diff_x = [diff_x, min(abs(C_x_f - other_cars_x))];
            
            C_y_f = C_y(f);
            %other_cars = cars(cars(:,8)>f-1 & cars(:,8)<f+1,:);
            other_cars_y = other_cars(:,5);
            diff_y = [diff_y, min(abs(C_y_f - other_cars_y))];
        end
            
        %%% Possible Collision (find small c ??)
        times = 1:1:(length(unique_frames));
        S_x_diff = BreachTraceSystem({'x_diff'},[times; diff_x]');
        S_y_diff = BreachTraceSystem({'y_diff'},[times; diff_y]');
        
        spec3 = STL_Formula('pi', 'ev_[0,T] (x_diff[t]  >  c)')
        spec3 = set_params(spec3, {'t0', 'T'}, [times(1) times(end)]);
        P = ParamSynthProblem(S_x_diff, spec3, {'c'}, [0, 100]);
        %P.solver_options.monotony = 1;
        c3 = P.solve();
        
        spec4 = STL_Formula('pi', 'ev_[0,T] (y_diff[t]  >  c)')
        spec4 = set_params(spec4, {'t0', 'T'}, [times(1) times(end)]);
        P = ParamSynthProblem(S_y_diff, spec4, {'c'}, [0, 100]);
        %P.solver_options.monotony = 1;
        c4 = P.solve();
        
        break
    end
    
end
    