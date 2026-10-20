import subprocess
import sys

def run_step(script_path):
    print(f"\n---> Executing {script_path}...")
    result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"Error executing {script_path}:\n{result.stderr}")
        sys.exit(result.returncode)

if __name__ == "__main__":
    pipeline_steps = [
        "scripts/get_data.py",
        "scripts/clean_data.py",
        "scripts/label_data.py",
        "scripts/normalize_data.py"
    ]
    
    for step in pipeline_steps:
        run_step(step)
        
    print("\nPipeline execution complete! Data is ready for modeling.")