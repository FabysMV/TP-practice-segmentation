# run_experiments.py
import subprocess

# Define your 5 parameter combinations (Model, Features, Depth)
experiments = [
    {"model": "unet", "features": 32, "depth": 3},
    {"model": "unet", "features": 64, "depth": 4},
    {"model": "unet", "features": 128, "depth": 5},
    {"model": "hourglass", "features": 64, "depth": 4},
    {"model": "hourglass", "features": 128, "depth": 4}
]

for exp in experiments:
    model = str(exp["model"])
    features = str(exp["features"])
    depth = str(exp["depth"])
    
    print(f"\n{'='*50}")
    print(f"Starting experiment: {model} (Features: {features}, Depth: {depth})")
    print(f"{'='*50}")
    
    # Construct the command exactly as you would type it in the terminal
    command = [
        "python", "train.py", 
        "--model", model, 
        "--features", features, 
        "--depth", depth
    ]
    
    # Execute the command
    try:
        # check=True ensures an error is raised if the training script fails
        subprocess.run(command, check=True) 
        print(f"Experiment completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Experiment failed with error: {e}")