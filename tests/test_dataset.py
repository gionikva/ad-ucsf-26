from pathlib import Path
import sys

# Force the project root (parent of tests/) into the search path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.dataset import ADNISegDataset

def test_dataset():
    n = 5
    dataset = ADNISegDataset(
            range=(0, n), 
            split=None # specified to ensure exactly 5 images are loaded
        ) 
    
    print(len(dataset))
    
    for i in range(n):
        print(dataset[i])


def run_tests():
    test_dataset()

if __name__ == "__main__":
    run_tests()