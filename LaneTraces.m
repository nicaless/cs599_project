data_folder = 'data/';
data_type = '_withLanes.csv';
vids = [2];

for i = vids
    filename = strcat(data_folder, num2str(i), data_type);
    disp(filename)
    cars = csvread(filename);
    % height	width	y	x	cog_y	cog_x	obj	frame	lane
    
    shoulder_spec_output = [];
    unique_objects = unique(cars(:,7));
    
    for car = unique_objects'
        C = cars(cars(:,7)>car-1 & cars(:,7)<car+1,:);
        C = sortrows(C, 8);
        times = 1:5:length(C(:,8));
        lane = C(:,9);
        lane = lane(1:5:end);
        shoulder_val = 0;
        lane_val = 1;
        if length(times) == 1
            continue
        end
        
        S = BreachTraceSystem({'lane'},[times', lane]);
        
        %%% Car in Lane
        spec = STL_Formula('phi', 'ev_[t0,T] ((lane[t] < lane_val) and ev_[tau, w] (lane[t] > shoulder_val)) ');
        spec = set_params(spec, {'t0', 'T', 'shoulder_val', 'lane_val'}, [times(1) times(end) shoulder_val lane_val]);
        P = ParamSynthProblem(S, spec, {'w', 'tau'}, [0, 90 ; 0, 90]);
        P.solver_options.monotony = [1, 1];
        c = P.solve();
%        shoulder_spec_output = [shoulder_spec_output; [i, car, c]]
            
        %%% Car Starts in Shoulder and Moves into Lane
%         spec1 = STL_Formula('phi', 'ev_[t0, T] (lane[t]==shoulder Until_[tau,w] (lane[t]!=shoulder))');
%         spec1 = set_params(spec1, {'t0', 'T', 'shoulder'}, [times(1) times(end) shoulder_val]);
%         P = ParamSynthProblem(S, spec1, {'tau','w'}, [0, 90 ; 0, 90]);
%         P.solver_options.monotony = [1 1];
%         c1 = P.solve();
        
        %%% Car Goes Into Shoulder and Stays there for some time
%         spec2 = STL_Formula('mu', 'ev_[t0, T] (lane[t]!=shoulder Until_[0,w] alw_[0,tau] lane[t]==shoulder)');
%         spec2 = set_params(spec2, {'t0', 'T', 'shoulder'}, [times(1) times(end) shoulder_val]);
%         P = ParamSynthProblem(S, spec2, {'tau','w'}, [0, 90 ; 0, 90]);
%         c2 = P.solve();
        
 
    end
        
    
end
    