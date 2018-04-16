data_folder = 'data/';
data_type = '_withLanes.csv';
%vids = [2 25 33 39 49 72 74];
vids = [2];

for i = vids
    filename = strcat(data_folder, num2str(i), data_type);
    disp(filename)
    cars = csvread(filename);
    % height	width	y	x	cog_y	cog_x	obj	frame	lane
    
    shoulder_spec_output = [];
    lane_spec_output = [];
    unique_objects = unique(cars(:,7));
    
    for car = unique_objects'
        C = cars(cars(:,7)>car-1 & cars(:,7)<car+1,:);
        C = sortrows(C, 8);
        %times = 1:5:length(C(:,8));
        times = 1:1:length(C(:,8));
        lane = C(:,9);
        %lane = lane(1:5:end);
        shoulder_val = 0;
        lane_val = 1;
        if length(times) == 1
            continue
        end
        if any(lane(:) == 0)
            S = BreachTraceSystem({'lane'},[times', lane]);

            %%% Car Moves Into Shoulder at some point
            spec = STL_Formula('phi', 'ev_[t0,T] ((lane[t] == lane_val) and ev_[0, tau] (alw_[0, w] (lane[t] == shoulder_val))) ');
            spec = set_params(spec, {'t0', 'T', 'shoulder_val', 'lane_val'}, [times(1) times(end) shoulder_val lane_val]);
            P = ParamSynthProblem(S, spec, {'w', 'tau'}, [0, 30 ; 0, 60]);
            P.solver_options.monotony = [-1, 1];
            c = P.solve();
            w = c(1);
            tau = c(2);
            shoulder_spec_output = [shoulder_spec_output; [i, car, w, tau]];

            %%% Car Starts in Shoulder and Moves into Lane
            spec = STL_Formula('phi', 'ev_[t0,T] ((lane[t] == shoulder_val) and ev_[0, tau] (alw_[0, w] (lane[t] == lane_val))) ');
            spec = set_params(spec, {'t0', 'T', 'shoulder_val', 'lane_val'}, [times(1) times(end) shoulder_val lane_val]);
            P = ParamSynthProblem(S, spec, {'w', 'tau'}, [0, 30 ; 0, 60]);
            P.solver_options.monotony = [-1, 1];
            c = P.solve();
            w = c(1);
            tau = c(2);
            lane_spec_output = [lane_spec_output; [i, car, w, tau]];
        end
        
    end
    
    csvwrite(strcat(num2str(i), "_to_shoulder.csv"), shoulder_spec_output)
    csvwrite(strcat(num2str(i), "_to_lane.csv"), lane_spec_output)
         
end
    