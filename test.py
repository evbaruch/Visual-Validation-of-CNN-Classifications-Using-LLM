import time
from tqdm import tqdm

# Example nested loops
def nested_loops():
    outer_loops = 10
    middle_loops = 20
    inner_loops = 30
    
    # Total iterations (for progress bar calculation)
    total_iterations = outer_loops * middle_loops * inner_loops
    
    # Start timing
    start_time = time.time()

    # Progress bar
    with tqdm(total=total_iterations, desc="Processing", unit="iteration") as pbar:
        for i in range(outer_loops):
            for j in range(middle_loops):
                for k in range(inner_loops):
 
                    # Simulate some work (e.g., computation or I/O)
                    time.sleep(0.001)  # Simulate a short delay

                    #print(f"Processing iteration {i+1}/{outer_loops}, {j+1}/{middle_loops}, {k+1}/{inner_loops}")

                    pbar.update(1)


    # End timing
    end_time = time.time()
    
    # Calculate elapsed time
    elapsed_time = end_time - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")

# Call the function
nested_loops()
