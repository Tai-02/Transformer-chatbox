import os, shutil

def reset():
    paths = ['data/processed', 'data/augmented', 'models', 'logs', '__pycache__']
    for p in paths:
        if os.path.isfile(p):
            os.remove(p)
            print(f"Removed file: {p}")
        elif os.path.isdir(p):
            shutil.rmtree(p)
            os.makedirs(p)
            print(f"Cleared directory: {p}")
    print("\nProject has been reset. Ready to start from scratch!")

if __name__ == "__main__":
    confirm = input("Are you sure you want to delete all trained data? (y/n): ")
    if confirm.lower() == 'y': reset()
