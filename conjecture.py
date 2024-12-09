from compressions import *
from random_generators import generate_points, get_random_num, set_to_number
from orbit import *
import graphing
import models
from data_generation import *
import models
from Lorenz_qualitative import *
import csv  # Import the csv module for writing to CSV

# Open CSV file in write mode, will create the file if it doesn't exist
with open('model_results.csv', 'w', newline='') as csvfile:
    fieldnames = ['Org', 'S1', 'S2', 'S1_behavior', 'S2_behavior', 'S1_zlib_compress', 'S2_zlib_compress', 'model_S1_score', 'model_S2_score']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # Write the header row to the CSV
    writer.writeheader()

    for num_points in range(700, 1200, 50):
        dt = 10 / num_points
        
        # Generate original train and test datasets
        # org_train = original(num_points, [-8.0, 8.0, 27.0])
        # org_test  = original(num_points, [8.0, 7.0, 15.0])


        # org_train = original_rossler(num_points, [-8.0, 8.0, 27.0])
        # org_test  = original_rossler(num_points, [8.0, 7.0, 15.0])


        org_train = original_henon(num_points, [0.1, 0.1])
        org_test  = original_henon(num_points, [-0.1, -0.1])

        for ratio1 in range(5, 10, 1):
            ratio1 = ratio1 / 10
            print(f"Using ratio1: {ratio1}")
            
            # First sample of points S1 from org_train and org_test
            S1_train = sampling_points(org_train, ratio1)
            S1_test  = sampling_points(org_test, ratio1)
            print(f"S1_train size:", len(S1_train))
            
            # Train model on S1
            model_S1 = models.get_sampled_system(S1_train, dt)
            #model_S1.print()
            S1_pred = simulate_dynamics([8,-8,27], num_points, model_S1, dt)
            
            #print(S1_pred)
            

            for ratio2 in range(5, 10, 1):
                ratio2 = ratio2 / 10
                
                print(f"Using ratio2: {ratio2}")

                # Second sample of points S2 from S1_train and S1_test
                S2_train = sampling_points(S1_train, ratio2)
                S2_test  = sampling_points(S1_test, ratio2)
                print(f"S2_train size:", len(S2_train))
                # Train model on S2
                model_S2 = models.get_sampled_system(S2_train, dt)
                S2_pred = simulate_dynamics([8,-8,27], num_points, model_S2, dt)
                #print(S2_pred)
                # Get the behavior for both S1 and S2
                # behavior1 = analyze_quadrant_behavior_with_loops(S1_pred)
                # behavior2 = analyze_quadrant_behavior_with_loops(S2_pred)

                # Convert behaviors to strings
                # S1_String = behaviors_to_string(behavior1)
                # S2_String = behaviors_to_string(behavior2)
                # print(f"S1 Behavior: {S1_String}, S2 Behavior: {S2_String}")
                S1_String = set_to_number(S1_pred)
                S2_String = set_to_number(S2_pred)
                # Check for equivalency between behavior strings
                same_count = equal_length(S1_String, S2_String)

                S1_zlib_compress = zlib_compression(S1_String)
                S2_zlib_compress = zlib_compression(S2_String)

                # print(f"Same count: {same_count}")

                length = min(len(S1_String), len(S2_String))
                # print(f"Normalized same count: {same_count / length}")

                # Model performance score for S1 and S2
                model_S1_score = model_S1.score(S1_test, t=dt)
                model_S2_score = model_S2.score(S2_test, t=dt)
                
                

                # print(f"Model S1 score: {model_S1_score}")
                # print(f"Model S2 score: {model_S2_score}")
                
                
                # Write the results to the CSV file
                writer.writerow({
                    'Org': len(org_train),
                    'S1': len(S1_train),
                    'S2': len(S2_train),
                    'S1_behavior': S1_String,
                    'S2_behavior': S2_String,
                    'S1_zlib_compress' : S1_zlib_compress,
                    'S2_zlib_compress' : S2_zlib_compress,
                    'model_S1_score': model_S1_score,
                    'model_S2_score': model_S2_score
                })